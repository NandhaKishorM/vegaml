"""Trained policies: verified loading, the route on every answer, and the fit that writes one.

Everything here runs on a stand-in engine with synthetic features, so no weights are needed. The
features put each label at its own centre, so heads fit cleanly, and an off-topic record sits in
directions no label ever used, which is what the outbound check measures.
"""
from __future__ import annotations

import base64
import io
import json
import math
import zipfile

import numpy as np
import pytest

import vegaml
from vegaml import Vega
from vegaml import policy as P

D = 256
LABELS = ("billing", "technical", "account")
QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team handles this?",
             "criteria": {"billing": "payments", "technical": "bugs", "account": "logins"}},
    "urgent": {"type": "boolean", "instructions": "Is it urgent?"},
}
_rng = np.random.default_rng(7)
CENTRES = {lab: v for lab, v in zip(LABELS, _rng.normal(size=(len(LABELS), D)) * 4.0, strict=True)}
# directions none of the labels use: an off-topic record lives here
OFF = np.linalg.qr(_rng.normal(size=(D, D)))[0][:, -1] * 40.0


class _Engine:
    cfg = {"max_len": 73728, "backbone_name": "dummy/reader"}
    model_dir = "/nowhere"
    backbone = "dummy/reader"

    def __init__(self, sha="e" * 64):
        self.sha = sha

    def base_sha256(self):
        return self.sha

    def decide(self, state, questions, alpha="0.1", views=1, **kw):
        out = {}
        for qid, q in questions.items():
            labels = vegaml.label_ids(q)
            probs = {lab: 1.0 / len(labels) for lab in labels}
            out[qid] = {"probs": probs, "confidence": max(probs.values()),
                        "prediction_set": labels, "abstain": False, "choice": labels[0]}
        return {"answers": out, "n_tokens": 10}


class FakeVega(Vega):
    """Vega with features computed from the state's text instead of a reader."""

    def features(self, state, question):
        text = str(state)
        if text.startswith("off:"):
            seed = int(text.split(":")[1])
            return (OFF + np.random.default_rng(seed).normal(size=D) * 0.3).astype(np.float32)
        lab, i = text.split(":")
        v = CENTRES[lab] + np.random.default_rng((LABELS.index(lab), int(i))).normal(size=D)
        return v.astype(np.float32)


def _rows(n_per=20):
    rows = []
    for lab in LABELS:
        for i in range(n_per):
            rows.append({"state": "%s:%d" % (lab, i),
                         "gold": {"team": lab, "urgent": "yes" if lab == "technical" else "no"}})
    return rows


@pytest.fixture(scope="module")
def fitted():
    model = FakeVega(_Engine())
    return P.fit_policy(model, _rows(), QUESTIONS)


def _keypair():
    from cryptography.hazmat.primitives import serialization as S
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    k = Ed25519PrivateKey.generate()
    pub = base64.b64encode(k.public_key().public_bytes(S.Encoding.Raw, S.PublicFormat.Raw)).decode()
    return k, pub


def _studio_zip(folder, name="acme", key=None, laya_spelling=True, **fields):
    """A zip the way Studio exports one: files in a top-level folder, questions in laya's spelling,
    a manifest naming the reader and engine, signed when a key is given."""
    files = {n: (folder / n).read_bytes() for n in ("heads.npz", "calibration.json")}
    qs = json.loads((folder / "questions.json").read_bytes())
    if laya_spelling:
        qs = {k: dict(v, type="noul") if v["type"] == "boolean" else v for k, v in qs.items()}
    files["questions.json"] = json.dumps(qs).encode()
    body = dict({"format": "embedding-studio-policy/1", "reader": "nandakishorm/vega-U",
                 "engine_sha256": "e" * 64}, **fields)
    sig = None
    if key is not None:
        k, pub = key
        body["key_id"] = P.key_id(pub)
        raw = P.manifest_for(files, **body)
        sig = base64.b64encode(k.sign(raw))
    else:
        raw = P.manifest_for(files, **body)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n, b in files.items():
            z.writestr("%s/%s" % (name, n), b)
        z.writestr("%s/manifest.json" % name, raw)
        if sig is not None:
            z.writestr("%s/manifest.sig" % name, sig)
    return buf.getvalue()


# ----------------------------------------------------------------------------- fitting
def test_fit_policy_serves_heads_calibrates_and_sets_an_outbound_limit(fitted):
    rep = fitted.fit_report
    assert rep["n_calibrate"] >= 10 and rep["n_fit"] + rep["n_calibrate"] == 60
    assert "team" in rep["serving"]
    assert fitted.calibration is not None and "team" in fitted.calibration.questions
    assert fitted.novelty[0] == "team" and 0 < fitted.novelty[1] < 1


def test_an_in_domain_record_routes_auto_and_an_off_topic_one_goes_to_review(fitted):
    ok = fitted.decide("billing:900")
    assert ok["answers"]["team"]["choice"] == "billing"
    assert ok["outbound"]["outside"] is False
    assert ok["answers"]["team"]["calibrated"] is True
    off = fitted.decide("off:3")
    assert off["outbound"]["outside"] is True
    assert off["route"] == "review"
    assert all(a["route"] == "review" and "unlike anything" in a["route_reason"]
               for a in off["answers"].values())


def test_every_answer_and_the_record_carry_a_route(fitted):
    out = fitted.decide("technical:901")
    assert out["route"] in ("auto", "review")
    for a in out["answers"].values():
        assert a["route"] in ("auto", "review") and a["route_reason"]


def test_too_few_rows_names_the_number():
    model = FakeVega(_Engine())
    with pytest.raises(ValueError, match="at least 25"):
        P.fit_policy(model, _rows(3), QUESTIONS)
    assert P.min_rows(QUESTIONS) == 25
    assert P.min_rows({"u": QUESTIONS["urgent"]}) == 20
    five = {"t": {"type": "choice", "instructions": "?", "criteria": list("abcde")}}
    assert P.min_rows(five) == 35


# ----------------------------------------------------------------------------- save and load
def test_save_then_load_round_trips_with_an_unsigned_manifest(fitted, tmp_path):
    fitted.save(tmp_path / "p")
    loaded = P.load_policy(tmp_path / "p", model=FakeVega(_Engine()))
    assert loaded.verified["status"] == "integrity"
    assert loaded.verified["engine"] == "matches"
    assert loaded.novelty == pytest.approx(fitted.novelty)
    a, b = fitted.decide("account:950"), loaded.decide("account:950")
    assert a["answers"]["team"]["probs"] == b["answers"]["team"]["probs"]
    assert a["route"] == b["route"]


def test_a_signed_studio_zip_loads_verified_with_laya_spelled_questions(fitted, tmp_path):
    fitted.save(tmp_path / "p")
    key = _keypair()
    zp = tmp_path / "acme-policy.zip"
    zp.write_bytes(_studio_zip(tmp_path / "p", key=key))
    pol = vegaml.load_policy(zp, model=FakeVega(_Engine()), trusted={P.key_id(key[1]): key[1]})
    assert pol.verified["status"] == "signed"
    assert pol.questions["urgent"]["type"] == "boolean"     # noul read as boolean
    assert pol.decide("billing:77")["answers"]["team"]["choice"] == "billing"


def test_a_changed_file_is_refused_by_name(fitted, tmp_path):
    fitted.save(tmp_path / "p")
    (tmp_path / "p" / "calibration.json").write_text("{}")
    with pytest.raises(P.PolicyError, match="calibration.json has been changed"):
        P.load_policy(tmp_path / "p", model=FakeVega(_Engine()))


def test_a_foreign_key_and_a_bad_signature_are_refused(fitted, tmp_path):
    fitted.save(tmp_path / "p")
    key = _keypair()
    zp = tmp_path / "x.zip"
    zp.write_bytes(_studio_zip(tmp_path / "p", key=key))
    with pytest.raises(P.PolicyError, match="does not trust"):
        P.load_policy(zp, model=FakeVega(_Engine()))
    other = _keypair()
    with pytest.raises(P.PolicyError, match="signature does not match"):
        P.load_policy(zp, model=FakeVega(_Engine()), trusted={P.key_id(key[1]): other[1]})


def test_signed_mode_refuses_an_unsigned_policy(fitted, tmp_path):
    fitted.save(tmp_path / "p")
    with pytest.raises(P.PolicyError, match="not signed"):
        P.load_policy(tmp_path / "p", model=FakeVega(_Engine()), verify="signed")


def test_the_wrong_engine_is_refused(fitted, tmp_path):
    fitted.save(tmp_path / "p")
    with pytest.raises(P.PolicyError, match="trained on engine"):
        P.load_policy(tmp_path / "p", model=FakeVega(_Engine(sha="f" * 64)))
    P.load_policy(tmp_path / "p", model=FakeVega(_Engine(sha="f" * 64)), verify=False)


def test_the_embedding_space_key_is_trusted_by_default():
    assert "589c4677fa61c61c" in P.TRUSTED_KEYS
    assert P.key_id(P.TRUSTED_KEYS["589c4677fa61c61c"]) == "589c4677fa61c61c"


# ----------------------------------------------------------------------------- the route
def test_an_added_option_is_uncalibrated_and_goes_to_review_instead_of_raising(fitted):
    qs = json.loads(json.dumps(fitted.questions))
    qs["team"]["criteria"]["sales"] = "buying"
    out = fitted.decide("billing:5", qs)
    team = out["answers"]["team"]
    assert team["calibrated"] is False and team["route"] == "review"
    assert "options changed" in team["route_reason"] and "sales" in team["route_reason"]


def test_your_own_line_decides_and_abstain_does_not_route(fitted):
    out = fitted.decide("billing:6", review_below=0.0)
    assert out["route"] == "auto"
    out = fitted.decide("billing:6", review_below=1.01)
    assert out["route"] == "review" and "your line" in out["answers"]["team"]["route_reason"]
    # outbound still applies under your own line
    assert fitted.decide("off:1", review_below=0.0)["route"] == "review"


def test_a_nan_probability_is_refused_not_routed():
    class _NaN(_Engine):
        def decide(self, state, questions, **kw):
            out = super().decide(state, questions)
            for a in out["answers"].values():
                a["probs"] = {k: float("nan") for k in a["probs"]}
            return out

    pol = P.Policy(FakeVega(_NaN()), questions=QUESTIONS)
    with pytest.raises(ValueError, match="finite"):
        pol.decide("billing:1")


def test_a_policy_without_calibration_sends_everything_to_review():
    pol = P.Policy(FakeVega(_Engine()), questions=QUESTIONS)
    out = pol.decide("billing:1")
    assert out["route"] == "review"
    assert all(a["calibrated"] is False for a in out["answers"].values())


# ----------------------------------------------------------------------------- inputs
def test_questions_are_checked_and_laya_spelling_is_read():
    q = P.normalise_questions({"u": {"type": "noul", "instructions": "urgent?",
                                     "labels": {"false": "fine", "true": "now"}},
                               "c": {"type": "choice", "instructions": "which?", "criteria": ["a", "b"]}})
    assert q["u"] == {"type": "boolean", "instructions": "urgent?",
                      "criteria": {"false": "fine", "true": "now"}}
    assert q["c"]["criteria"] == {"a": None, "b": None}
    with pytest.raises(ValueError, match="'x' has type"):
        P.normalise_questions({"x": {"type": "multi", "instructions": "?"}})
    with pytest.raises(ValueError, match="at least two options"):
        P.normalise_questions({"x": {"type": "choice", "instructions": "?", "criteria": ["a"]}})


def test_gold_labels_that_cannot_be_placed_are_refused():
    b, s = {"type": "boolean"}, {"type": "score", "criteria": ["low", "mid", "high"]}
    assert P.gold_label(b, 1.0) == "true" and P.gold_label(b, "No") == "false"
    assert P.gold_label(s, "high") == "2" and P.gold_label(s, 1) == "1"
    for bad in ("maybe", 2):
        with pytest.raises(ValueError):
            P.gold_label(b, bad)
    for bad in (1.5, 3, True, float("nan")):
        with pytest.raises(ValueError):
            P.gold_label(s, bad)
    with pytest.raises(ValueError, match="options"):
        P.gold_label(QUESTIONS["team"], "sales")


def test_rows_load_from_csv(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("state,team,urgent\nI was billed twice,billing,no\nlogin broken,,yes\n,billing,no\n")
    rows = P.load_rows(p, QUESTIONS)
    assert rows == [{"state": "I was billed twice", "gold": {"team": "billing", "urgent": "no"}},
                    {"state": "login broken", "gold": {"urgent": "yes"}}]
    with pytest.raises(ValueError, match="match no question"):
        p.write_text("state,dept\nx,billing\n")
        P.load_rows(p, QUESTIONS)


def test_not_a_policy_is_said_plainly(tmp_path):
    with pytest.raises(P.PolicyError, match="no policy"):
        P.read(tmp_path / "missing.zip")
    (tmp_path / "empty").mkdir()
    with pytest.raises(P.PolicyError, match="no heads.npz"):
        P.read(tmp_path / "empty")


def test_residual_is_small_inside_and_large_outside(fitted):
    head = fitted.model.heads["team"][0]
    inside = P.residual(head, CENTRES["billing"])
    outside = P.residual(head, OFF)
    assert inside < outside and math.isfinite(outside)


# ----------------------------------------------------------------------------- video
def test_a_silent_clip_is_decided_from_its_frames_and_goes_to_review(fitted, tmp_path):
    av = pytest.importorskip("av")
    pytest.importorskip("PIL")
    clip = tmp_path / "clip.mp4"
    with av.open(str(clip), "w") as c:
        s = c.add_stream("mpeg4", rate=5)
        s.width, s.height, s.pix_fmt = 64, 64, "yuv420p"
        for i in range(5):
            img = np.full((64, 64, 3), i * 50, dtype=np.uint8)
            for packet in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(packet)
        for packet in s.encode():
            c.mux(packet)
    seen = {}

    def decide_image(image, questions, **kw):
        seen["size"] = image.size
        return fitted.model.engine.decide("pixels", questions)

    fitted.model.decide_image = decide_image
    try:
        out = fitted.decide(video=str(clip))
    finally:
        del fitted.model.decide_image
    assert out["modality"] == "video" and out["video"]["path"] == "frames"
    assert out["video"]["frames"] == 4 and seen["size"] == (1024, 1024)
    assert all(a["calibrated"] is False for a in out["answers"].values())
    assert out["route"] == "review"


# ----------------------------------------------------------------------------- serving
def test_the_server_answers_with_the_policys_own_questions_and_a_route(fitted):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from vegaml.serve import create_app

    client = TestClient(create_app(policy=fitted))
    r = client.post("/v1/decide", json={"state": "billing:31"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["route"] in ("auto", "review") and set(body["answers"]) == {"team", "urgent"}
    assert body["answers"]["team"]["route_reason"]
    assert body["audit"]["policy"]["verified"] == "fitted here"
    off = client.post("/v1/decide", json={"state": "off:9"}).json()
    assert off["route"] == "review" and off["outbound"]["outside"] is True
    info = client.get("/v1/model").json()
    assert info["policy"]["outbound"]["question"] == "team"


def test_without_a_policy_the_server_still_needs_questions():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from vegaml.serve import create_app

    client = TestClient(create_app(model=FakeVega(_Engine())))
    assert client.post("/v1/decide", json={"state": "x"}).status_code == 422
