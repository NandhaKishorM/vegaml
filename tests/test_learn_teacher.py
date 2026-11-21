"""The Escalator: vega answers, what it is unsure of goes to a teacher, the teacher's answer is served and
learned from, and audits keep a confident but wrong head from shutting the teacher out. A stand-in reader
and teacher, so nothing downloads and nothing is sent anywhere."""

import hashlib
import json
import random
import types

import numpy as np

from vegaml.learn import Escalator, Learner, claude_teacher
from vegaml.learn.teacher import claude_prompt

Q = {
    "team": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "money", "technical": "faults", "sales": "buying"},
    }
}
WORDS = {
    "billing": ["refund", "invoice", "charge"],
    "technical": ["crash", "error", "login"],
    "sales": ["quote", "seats", "trial"],
}
FILL = ["please", "hello", "my", "the", "account", "again", "thanks"]


def stream(n, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        team = rng.choice(list(WORDS))
        words = [rng.choice(WORDS[team])] + rng.sample(FILL, 3)
        rng.shuffle(words)
        out.append((" ".join(words) + " %d" % rng.randrange(10**6), team))
    return out


def truth(text):
    return next(t for t, ws in WORDS.items() if any(w in text.split() for w in ws))


class FlaggingReader:
    """Features carry the team; vega's own engine always says billing and flags every record unbound,
    the way the real engine does on a domain it has not been fitted to."""

    def fingerprint(self):
        return "fakevega00000002"

    def features(self, texts, questions, batch_size=8):
        out = np.zeros((len(texts), len(questions), 16), np.float32)
        for i, t in enumerate(texts):
            rng = np.random.default_rng(int(hashlib.sha1(t.encode()).hexdigest()[:8], 16))
            v = rng.normal(0, 0.3, 16)
            v[list(WORDS).index(truth(t))] += 2.5
            out[i, :] = v
        return out.astype(np.float16)

    def engine_answers(self, text, questions):
        return {
            q: {
                "choice": "billing",
                "confidence": 0.5,
                "abstain": False,
                "unbound": True,
                "probs": {"billing": 0.5, "technical": 0.3, "sales": 0.2},
            }
            for q in questions
        }

    def engine_labels(self, text, questions):
        return {q: "billing" for q in questions}


class Teacher:
    def __init__(self, flip=False):
        self.calls, self.flip = 0, flip

    def __call__(self, text, questions):
        self.calls += 1
        t = truth(text)
        if self.flip:  # the policy changed: what was billing is now technical, and so on
            t = {"billing": "technical", "technical": "sales", "sales": "billing"}[t]
        return {q: t for q in questions}


def desk(**kw):
    learner = Learner(Q, reader=FlaggingReader(), auto=False, every=kw.pop("every", 20))
    return learner, Escalator(learner, kw.pop("teacher", Teacher()), **kw)


def test_unbound_goes_to_the_teacher_and_its_answer_is_served():
    _, d = desk(audit=0)
    text, team = stream(1, 1)[0]
    a = d.decide(text)["answers"]["team"]
    assert a["source"] == "teacher" and a["escalated"] == "unbound"
    assert a["answer"] == team and a["vega_answer"] == "billing"


def test_the_teacher_is_needed_less_and_less_until_almost_never():
    learner, d = desk(audit=0)
    t = d.teacher
    rows = stream(400, 2)
    first = None
    for i, (text, _) in enumerate(rows):
        d.decide(text)
        if i == 19:  # before the first refit (every=20)
            first = t.calls / 20
    late = d.report()["teacher_share_recent"]
    assert first == 1.0  # unfitted: everything is unbound
    assert late < 0.2, d.report()  # fitted on the teacher's answers: the head answers alone
    right = np.mean([learner.decide(x)["answers"]["team"]["answer"] == y for x, y in stream(100, 3)])
    assert right > 0.9


def test_a_teacher_that_fails_leaves_vegas_answer_and_learns_nothing():
    def broken(text, questions):
        raise RuntimeError("down")

    learner, d = desk(teacher=broken, audit=0)
    a = d.decide(stream(1, 4)[0][0])["answers"]["team"]
    assert a["source"] == "vega" and a["answer"] == "billing"
    assert learner.pending() == 0 and d.report()["teacher_errors"] == 1


def test_an_unusable_teacher_answer_is_skipped():
    learner, d = desk(teacher=lambda text, qs: {"team": "marketing"}, audit=0)
    a = d.decide(stream(1, 5)[0][0])["answers"]["team"]
    assert a["source"] == "vega" and learner.pending() == 0


def test_audits_raise_the_line_when_a_confident_head_stops_agreeing():
    learner, d = desk(audit=0.3, min_checked=20, seed=1)
    for text, _ in stream(300, 6):  # learn the policy until the head answers alone
        d.decide(text)
    assert d.report()["teacher_share_recent"] < 0.6
    calm = d.line["team"]
    d.teacher = Teacher(flip=True)  # the policy changes; only checks can notice
    for text, _ in stream(120, 7):
        d.decide(text)
    r = d.report()
    assert r["line"]["team"] > calm
    assert r["audit_agreement"]["team"] < 0.95


def test_the_line_comes_back_down_once_vega_agrees_again():
    # a lockout would leave the line at the top for good once everything goes to the teacher
    learner, d = desk(audit=0.3, min_checked=20, seed=2)
    d.teacher = Teacher(flip=True)
    for text, _ in stream(150, 8):  # learns the flipped policy first
        d.decide(text)
    d.teacher = Teacher()  # then the policy changes back
    for text, _ in stream(150, 9):
        d.decide(text)
    high = d.line["team"]
    for text, _ in stream(400, 10):  # relearns it; its confident answers agree again
        d.decide(text)
    assert d.line["team"] < high or d.line["team"] < 0.9
    assert d.report()["teacher_share_recent"] < 0.6


def test_claude_prompt_lists_every_option_and_constrains_the_answer():
    msg, schema = claude_prompt("charged twice", Q)
    assert "charged twice" in msg and "billing: money" in msg
    assert schema["properties"]["team"]["enum"] == ["billing", "technical", "sales"]
    msg, schema = claude_prompt("x", {"urgent": {"type": "boolean", "instructions": "Reply today?"}})
    assert schema["properties"]["urgent"]["enum"] == ["false", "true"]


class FakeClient:
    def __init__(self, reply, stop="end_turn"):
        self.sent, self.reply, self.stop = [], reply, stop
        self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(create=self._create))
        self.messages = types.SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.sent.append(kw)
        return types.SimpleNamespace(
            stop_reason=self.stop,
            usage=types.SimpleNamespace(input_tokens=900, output_tokens=12),
            content=[types.SimpleNamespace(type="text", text=json.dumps(self.reply))],
        )


def test_claude_teacher_sends_a_structured_request_and_reads_the_answer():
    c, usage = FakeClient({"team": "technical"}), {}
    teach = claude_teacher(client=c, usage=usage)
    assert teach("the app crashes", Q) == {"team": "technical"}
    kw = c.sent[0]
    assert kw["model"] == "claude-opus-5" and kw["fallbacks"] == "default"
    assert kw["output_config"]["format"]["schema"]["properties"]["team"]["enum"] == [
        "billing",
        "technical",
        "sales",
    ]
    assert usage == {"calls": 1, "input_tokens": 900, "output_tokens": 12}


def test_a_refused_record_gets_no_answer():
    teach = claude_teacher(client=FakeClient({}, stop="refusal"))
    assert teach("anything", Q) == {}


def test_a_head_that_only_gives_the_common_answer_never_takes_over():
    # 5% of records are fraud; the features carry no signal, so the best a head can do is "no" to
    # everything: 95% accuracy, and not one fraud caught. vega's own answers catch them.
    from vegaml.learn.learner import fit_head

    rng = np.random.default_rng(0)
    q = {"type": "boolean", "instructions": "Is this fraud?"}
    y = ["true" if i % 20 == 0 else "false" for i in range(400)]
    X = rng.normal(0, 1, (400, 16)).astype(np.float32)
    h = fit_head(q, X, y, list(y))
    assert h["cv"] > 0.9 and not h["usable"]
    assert h["cv_balanced"] < 0.6 and h["engine_balanced"] == 1.0
    # the same rare answer, now carried by the features, and vega always says no: the head earns it
    X[[i for i in range(400) if y[i] == "true"], 0] += 4.0
    h = fit_head(q, X, y, ["false"] * 400)
    assert h["usable"] and h["cv_balanced"] > 0.7
