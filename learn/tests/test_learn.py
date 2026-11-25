"""Continuous learning (vegaml_learn) end to end with a stand-in reader: features that carry the answer, plus
noise, so nothing downloads. Starting set, learning from feedback, the guard, undo, save and resume,
followers,
bundles, and the REST API with JWT."""

import datetime as dt
import hashlib
import os
import random

import numpy as np
import pytest

from vegaml_learn import Learner, build, label_of

Q = {
    "team": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "money", "technical": "faults", "sales": "buying"},
    },
    "urgent": {"type": "boolean", "instructions": "Reply today?"},
}
WORDS = {
    "billing": ["refund", "invoice", "charge"],
    "technical": ["crash", "error", "login"],
    "sales": ["quote", "seats", "trial"],
}
FILL = ["please", "hello", "my", "the", "team", "account", "again", "thanks"]


def rows(n, seed, flip=False):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        team = rng.choice(list(WORDS))
        urgent = rng.random() < 0.5
        words = [rng.choice(WORDS[team])] + rng.sample(FILL, 3) + (["today"] if urgent else [])
        rng.shuffle(words)
        label = {"billing": "technical", "technical": "sales", "sales": "billing"}[team] if flip else team
        out.append((" ".join(words) + " %d" % rng.randrange(10**6), {"team": label, "urgent": urgent}))
    return out


class FakeReader:
    """Features carry the topic and the urgency, with noise; vega's own answer is always billing / no."""

    def __init__(self, fp="fakevega00000001"):
        self.fp, self.reads = fp, 0

    def fingerprint(self):
        return self.fp

    def features(self, texts, questions, batch_size=8):
        out = np.zeros((len(texts), len(questions), 32), np.float32)
        for i, t in enumerate(texts):
            rng = np.random.default_rng(int(hashlib.sha1(t.encode()).hexdigest()[:8], 16))
            for j, _q in enumerate(questions):
                v = rng.normal(0, 0.3, 32)
                for k, (_team, ws) in enumerate(WORDS.items()):
                    if any(w in t.split() for w in ws):
                        v[k] += 2.0
                if "today" in t.split():
                    v[5] += 2.0
                v[10 + j] += 1.0
                out[i, j] = v
        self.reads += len(texts)
        return out.astype(np.float16)

    def engine_answers(self, text, questions):
        a = {}
        for q, d in questions.items():
            a[q] = (
                {"boolean": 0.2, "probs": {"true": 0.2, "false": 0.8}}
                if d["type"] == "boolean"
                else {"choice": "billing", "probs": {"billing": 0.5, "technical": 0.3, "sales": 0.2}}
            )
        return a

    def engine_labels(self, text, questions):
        a = self.engine_answers(text, questions)
        return {q: label_of(questions[q], a[q]) for q in questions}


def accuracy(learner, data):
    ok = {"team": 0, "urgent": 0}
    for t, lab in data:
        a = learner.decide(t)["answers"]
        ok["team"] += a["team"]["answer"] == lab["team"]
        ok["urgent"] += a["urgent"]["answer"] == lab["urgent"]
    return {q: v / len(data) for q, v in ok.items()}


TEST = rows(120, 99)


def make(tmp, start_rows=200, **kw):
    r = kw.pop("reader", None) or FakeReader()
    examples = rows(start_rows, 1) if start_rows else []
    L = Learner(Q, examples, reader=r, auto=kw.pop("auto", False), **kw)
    return L, L.report, r


def test_a_starting_set_answers_well_on_day_one(tmp_path):
    L, report, _ = make(tmp_path)
    assert (
        report["team"]["starts_with"] == "a head fitted on the starting set" and report["team"]["rows"] == 200
    )
    acc = accuracy(L, TEST)
    assert acc["team"] > 0.9 and acc["urgent"] > 0.9
    out = L.decide(TEST[0][0])
    assert out["answers"]["team"]["readout"] == "head" and out["version"] == 0
    assert set(out["answers"]["team"]["probs"]) == {"billing", "technical", "sales"}


def test_with_no_starting_set_vega_answers_until_it_has_learned(tmp_path):
    L, report, _ = make(tmp_path, start_rows=0, every=50)
    assert report["team"]["starts_with"] == "vega's own answers"
    assert L.decide(TEST[0][0])["answers"]["team"]["readout"] == "engine"
    before = accuracy(L, TEST)
    for t, lab in rows(60, 5):
        L.feedback(t, lab)
    out = L.step()
    assert out["version"] == 1 and out["readout"]["team"] == "head"
    after = accuracy(L, TEST)
    assert after["team"] > before["team"] + 0.4 and after["team"] > 0.85


def test_the_starting_set_gives_way_to_your_own_answers(tmp_path):
    L, *_ = make(tmp_path, start_rows=100, every=10)
    for t, lab in rows(30, 6):
        L.feedback(t, lab)
    L.step(min_rows=1)
    assert L.status()["questions"]["team"] == dict(
        L.status()["questions"]["team"], starting_set=100, yours=30
    )
    assert len(L.store.rows("team")) == 100  # 70 starting rows + 30 of yours


def test_a_correction_replaces_the_answer_it_corrects(tmp_path):
    L, *_ = make(tmp_path, start_rows=100, every=1)
    first_text, first = rows(100, 1)[0]
    wrong = {"team": "sales" if first["team"] != "sales" else "billing", "urgent": first["urgent"]}
    # a starting-set row corrected: one row for that record, carrying the new answer
    L.feedback(first_text, wrong, correction=True)
    L.step(min_rows=1)
    keys = L.store.k["team"]
    k = keys[-1]
    assert keys.count(k) == 1 and L.store.y["team"][-1] == wrong["team"]
    assert L.store.src["team"][-1] == "yours"
    assert L.status()["questions"]["team"]["starting_set"] == 99
    # one of your own answers corrected later, and the same record twice in one batch: newest wins
    t, lab = rows(1, 77)[0]
    L.feedback(t, lab)
    L.step(min_rows=1)
    fixed = {"team": "technical" if lab["team"] != "technical" else "billing", "urgent": not lab["urgent"]}
    L.feedback(t, lab)
    L.feedback(t, fixed, correction=True)
    out = L.step(min_rows=1)
    assert out["held"] == {}  # a person's correction is learned, not held as drift
    for q in ("team", "urgent"):
        ks = L.store.k[q]
        assert ks.count(ks[-1]) == 1
    assert L.store.y["team"][-1] == fixed["team"] and L.store.y["urgent"][-1] == str(fixed["urgent"]).lower()
    # the keys survive a save and a bundle
    raw = L.store.dump()
    from vegaml_learn.learner import Store

    back = Store.load(raw, list(Q))
    assert back.k["team"] == L.store.k["team"]


def test_a_batch_that_contradicts_what_it_learned_is_held_then_accepted_or_discarded(tmp_path):
    L, *_ = make(tmp_path, every=100)
    for t, lab in rows(100, 7):
        L.feedback(t, lab)
    assert L.step()["held"] == {}  # your first answers set the baseline
    assert L.status()["agreement"]["team"] > 0.9
    for t, lab in rows(100, 8, flip=True):
        L.feedback(t, {"team": lab["team"]})
    out = L.step()
    assert "team" in out["held"] and "team" in L.held()
    assert accuracy(L, TEST)["team"] > 0.9  # nothing learned from the bad batch
    L.discard("team")
    assert L.held() == {}
    for t, lab in rows(100, 9, flip=True):
        L.feedback(t, {"team": lab["team"]})
    L.step()
    before = accuracy(L, TEST)["team"]
    v = L.accept("team")["version"]  # the policy really did change
    assert L.held() == {} and v == L.version and accuracy(L, TEST)["team"] < before - 0.2


def test_undo_goes_back_one_batch(tmp_path):
    L, *_ = make(tmp_path, start_rows=0, every=50)
    for t, lab in rows(60, 9):
        L.feedback(t, lab)
    L.step()
    assert L.decide(TEST[0][0])["answers"]["team"]["readout"] == "head"
    v = L.undo()
    assert v == 2 and L.decide(TEST[0][0])["answers"]["team"]["readout"] == "engine"
    with pytest.raises(RuntimeError):
        L.undo()


def test_state_is_saved_resumed_and_followed(tmp_path):
    work = str(tmp_path / "state")
    L, _, r = make(tmp_path, start_rows=0, every=50, workdir=work)
    for t, lab in rows(60, 10):
        L.feedback(t, lab)
    L.step()
    reads = r.reads
    again = Learner(Q, [], reader=r, auto=False, workdir=work)
    assert again.version == 1 and again.decide(TEST[0][0])["answers"]["team"]["readout"] == "head"
    follower = Learner(Q, [], reader=r, workdir=work, follow=3600)
    with pytest.raises(RuntimeError):
        follower.feedback(*rows(1, 11)[0])
    for t, lab in rows(60, 12):
        again.feedback(t, lab)
    again.step()
    assert follower.version == 1 and follower.refresh() and follower.version == 2
    follower.close()
    assert sorted(os.listdir(work)) == ["v000001", "v000002"]
    assert r.reads >= reads


def test_a_resumed_learner_does_not_reread_its_starting_set(tmp_path):
    work = str(tmp_path / "state")
    L, _, r = make(tmp_path, start_rows=100, every=10, workdir=work)
    for t, lab in rows(10, 15):
        L.feedback(t, lab)
    L.step()
    reads = r.reads
    again = Learner(Q, rows(100, 1), reader=r, auto=False, workdir=work)
    assert again.version == 1 and r.reads == reads
    # other questions, or another starting set, never pick up this state
    other = Learner(Q, rows(50, 2), reader=r, auto=False, workdir=work)
    assert other.version == 0


def test_a_bundle_carries_what_it_learned_to_another_machine(tmp_path):
    L, _, r = make(tmp_path, start_rows=0, every=50)
    for t, lab in rows(60, 16):
        L.feedback(t, lab)
    L.step()
    path = L.save_bundle(str(tmp_path / "acme.zip"), name="acme")
    moved = Learner(bundle=path, reader=r, auto=False)
    assert moved.decide(TEST[0][0])["answers"]["team"]["readout"] == "head"
    assert accuracy(moved, TEST)["team"] > 0.85
    with pytest.raises(ValueError):
        Learner(bundle=path, reader=FakeReader("anothervega00001"), auto=False)
    learn, start, report = build(r, Q, rows(50, 1))
    assert report["team"]["rows"] == 50 and learn and start


def test_it_needs_questions_or_a_bundle(tmp_path):
    with pytest.raises(ValueError):
        Learner(reader=FakeReader(), auto=False)


def test_labels_are_read_the_way_people_write_them():
    assert label_of(Q["urgent"], True) == "true" and label_of(Q["urgent"], "no") == "false"
    assert label_of(Q["team"], "sales") == "sales"
    with pytest.raises(ValueError):
        label_of(Q["team"], "marketing")
    s = {"type": "score", "instructions": "x", "criteria": ["low", "mid", "high"]}
    assert label_of(s, 2) == "2" and label_of(s, {"score": 1}) == "1"


def test_feedback_learns_in_the_background(tmp_path):
    events = []
    L = Learner(Q, [], reader=FakeReader(), every=40, on_event=events.append)
    for t, lab in rows(40, 14):
        L.feedback(t, lab)
    L.flush(timeout=60)
    assert L.version == 1 and events and events[0]["kind"] == "learned"
    L.close()


def test_it_is_exported_from_vegaml_itself():
    import vegaml

    assert vegaml.Learner is Learner


# ---------------------------------------------------------------- the REST API
def client(tmp_path, **kw):
    pytest.importorskip("fastapi")
    pytest.importorskip("jwt")
    from fastapi.testclient import TestClient

    from vegaml_learn.serve import create_app

    L, *_ = make(tmp_path, every=1000)
    return TestClient(create_app(L, **kw)), L


def test_every_route_but_health_needs_a_token_with_the_right_scope(tmp_path):
    c, L = client(tmp_path, secret="s3cret")
    from vegaml_learn.serve import mint

    assert c.get("/healthz").status_code == 200
    body = {"text": TEST[0][0]}
    assert c.post("/v1/decide", json=body).status_code == 401
    bad = mint("not-the-secret", "x", ["decide"])
    assert c.post("/v1/decide", json=body, headers={"authorization": "Bearer " + bad}).status_code == 401
    d = {"authorization": "Bearer " + mint("s3cret", "helpdesk", ["decide"])}
    r = c.post("/v1/decide", json=body, headers=d)
    assert r.status_code == 200 and r.json()["answers"]["team"]["answer"] in WORDS
    assert (
        c.post("/v1/feedback", json={"text": "x", "answers": {"team": "sales"}}, headers=d).status_code == 403
    )
    assert c.get("/v1/status", headers=d).status_code == 403
    f = {"authorization": "Bearer " + mint("s3cret", "crm", ["feedback"])}
    r = c.post(
        "/v1/feedback", json={"text": TEST[1][0], "answers": {"team": "sales", "urgent": True}}, headers=f
    )
    assert r.status_code == 200 and r.json()["pending"] == 1
    assert (
        c.post("/v1/feedback", json={"text": "x", "answers": {"team": "marketing"}}, headers=f).status_code
        == 400
    )
    a = {"authorization": "Bearer " + mint("s3cret", "ops", ["admin"])}
    assert c.get("/v1/status", headers=a).json()["pending"] == 1
    assert c.post("/v1/decide/batch", json={"texts": [t for t, _ in TEST[:5]]}, headers=a).status_code == 200
    assert c.post("/v1/decide", json={"text": "x" * 20001}, headers=d).status_code == 422


def test_an_expired_token_is_refused_and_a_server_without_a_key_will_not_start(tmp_path):
    c, L = client(tmp_path, secret="s3cret")
    import jwt

    from vegaml_learn.serve import create_app

    old = jwt.encode(
        {"sub": "x", "scope": "decide", "exp": dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)},
        "s3cret",
        algorithm="HS256",
    )
    r = c.post("/v1/decide", json={"text": "x"}, headers={"authorization": "Bearer " + old})
    assert r.status_code == 401 and "expired" in r.text
    no_exp = jwt.encode({"sub": "x", "scope": "decide"}, "s3cret", algorithm="HS256")
    assert (
        c.post("/v1/decide", json={"text": "x"}, headers={"authorization": "Bearer " + no_exp}).status_code
        == 401
    )
    with pytest.raises(SystemExit):
        create_app(L)


def test_tokens_from_your_own_identity_provider_are_checked_with_its_public_key(tmp_path):
    pytest.importorskip("cryptography")
    jwt = pytest.importorskip("jwt")
    from cryptography.hazmat.primitives import serialization as S
    from cryptography.hazmat.primitives.asymmetric import ec

    k = ec.generate_private_key(ec.SECP256R1())
    pem = k.public_key().public_bytes(S.Encoding.PEM, S.PublicFormat.SubjectPublicKeyInfo).decode()
    c, _ = client(tmp_path, public_key=pem, audience="vega-learner", issuer="https://id.acme.example")
    claims = {
        "sub": "svc",
        "scope": "decide",
        "aud": "vega-learner",
        "iss": "https://id.acme.example",
        "exp": dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=5),
    }
    tok = jwt.encode(claims, k, algorithm="ES256")
    assert (
        c.post(
            "/v1/decide", json={"text": TEST[0][0]}, headers={"authorization": "Bearer " + tok}
        ).status_code
        == 200
    )
    wrong = jwt.encode(dict(claims, aud="something-else"), k, algorithm="ES256")
    assert (
        c.post("/v1/decide", json={"text": "x"}, headers={"authorization": "Bearer " + wrong}).status_code
        == 401
    )


def test_temperature_softens_an_over_confident_head():
    from vegaml_learn.learner import calibration_error, fit_temperature, temper

    # says 0.99 every time but is right 70% of the time
    rows = [({"a": 0.99, "b": 0.01}, "a" if i % 10 < 7 else "b") for i in range(200)]
    t = fit_temperature(rows)
    assert t > 1.5
    after = temper(rows[0][0], t)["a"]
    assert 0.6 < after < 0.8
    right = [y == "a" for _, y in rows]
    assert calibration_error([after] * 200, right) < calibration_error([0.99] * 200, right)


def test_calibration_attaches_is_applied_and_travels_in_the_bundle(tmp_path):
    from vegaml_learn import Escalator

    L, *_ = make(tmp_path, start_rows=200, workdir=str(tmp_path / "w"))
    raw = L.decide(TEST[0][0])["answers"]["team"]["confidence"]
    report = L.calibrate(rows(200, 42))
    assert set(report) == {"team", "urgent"}
    assert report["team"]["temperature"] > 0 and report["team"]["rows"] == 200
    assert L.calibration["team"]["temperature"] == report["team"]["temperature"]
    t = report["team"]["temperature"]
    cal = L.decide(TEST[0][0])["answers"]["team"]["confidence"]
    assert (abs(t - 1) < 1e-6 and cal == raw) or cal != raw
    # carried by the bundle and applied by the learner that loads it
    path = str(tmp_path / "b.zip")
    L.save_bundle(path)
    M = Learner(bundle=path, reader=L.reader, auto=False)
    assert M.calibration["team"]["temperature"] == t
    assert M.decide(TEST[0][0])["answers"]["team"]["confidence"] == cal
    # and kept with the saved state
    R = Learner(Q, rows(200, 1), reader=L.reader, auto=False, workdir=str(tmp_path / "w"))
    assert R.calibration.get("team", {}).get("temperature") == t
    # an escalator can start from the calibrated lines instead of earning them from the top
    E = Escalator(M, lambda text, qs: None, line="calibrated")
    want = M.calibration["team"]["start_line"]
    assert E.line["team"] == (min(E.max_line, want) if want is not None else E.max_line)
    assert Escalator(M, lambda text, qs: None).line["team"] == E.max_line
