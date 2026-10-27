"""Calibration on labels, against a fake engine: temperature, coverage, persistence, refusals."""
from __future__ import annotations

import random

import pytest

from vegaml.calibrate import Calibration, calibrate, ece

QB = {"churn": {"type": "boolean", "instructions": "Does the customer threaten to leave?"}}
QC = {"team": {"type": "choice", "instructions": "Which team?",
               "criteria": {"billing": "payments", "technical": "bugs", "sales": "plans"}}}
QS = {"sev": {"type": "score", "instructions": "How severe?",
              "criteria": ["cosmetic", "degraded", "outage"]}}


class _Overconfident:
    """Says 0.97 for the label hidden in the state, but is right only when the state says so."""

    def __init__(self, labels=("false", "true")):
        self.labels = list(labels)

    def decide(self, state, questions, mode=None, alpha="0.1", views=1):
        claimed = state["claim"]
        probs = {k: (0.97 if k == claimed else 0.03 / (len(self.labels) - 1))
                 for k in self.labels}
        return {"answers": {q: {"probs": dict(probs)} for q in questions}}


def _boolean_examples(n=40, accuracy=0.7, seed=5):
    """The engine claims 0.97 but is right `accuracy` of the time: textbook overconfidence."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        claim = rng.choice(["false", "true"])
        truth = claim if rng.random() < accuracy else ("true" if claim == "false" else "false")
        out.append(({"claim": claim}, {"churn": truth}))
    return out


# ---------------------------------------------------------------------- fitting

def test_an_overconfident_engine_gets_a_temperature_above_one():
    cal = calibrate(_Overconfident(), _boolean_examples(), QB)
    r = cal.report["churn"]
    assert r["calibrated"] and r["temperature"] > 1.0
    assert r["ece_after"] < r["ece_before"]


def test_apply_moves_probability_but_never_the_answer():
    cal = calibrate(_Overconfident(), _boolean_examples(), QB)
    raw = _Overconfident().decide({"claim": "true"}, QB, mode="engine")
    out = cal.apply(raw)
    a = out["answers"]["churn"]
    assert a["calibrated"] and a["confidence"] < 0.97
    assert max(a["probs"], key=a["probs"].get) == "true"          # argmax invariant
    assert a["boolean"] == a["probs"]["true"]
    assert abs(sum(a["probs"].values()) - 1.0) < 1e-9
    assert raw["answers"]["churn"]["probs"]["true"] == 0.97        # the input is untouched


def test_the_conformal_set_widens_where_the_engine_was_wrong_often():
    cal = calibrate(_Overconfident(), _boolean_examples(accuracy=0.55), QB, alpha=0.1)
    out = cal.apply(_Overconfident().decide({"claim": "true"}, QB))
    # To cover 90% while being right barely over half the time, the set must hold both options.
    assert set(out["answers"]["churn"]["prediction_set"]) == {"false", "true"}


def test_score_and_choice_fields_are_recomputed():
    labels = ["0", "1", "2"]
    eng = _Overconfident(labels)
    rng = random.Random(7)
    ex = []
    for _ in range(30):
        claim = rng.choice(labels)
        truth = claim if rng.random() < 0.8 else rng.choice(labels)
        ex.append(({"claim": claim}, {"sev": truth}))
    cal = calibrate(eng, ex, QS)
    a = cal.apply(eng.decide({"claim": "2"}, QS))["answers"]["sev"]
    assert a["score"] == 2 and 0.0 <= a["expected_score"] <= 2.0

    eng2 = _Overconfident(["billing", "technical", "sales"])
    ex2 = [({"claim": c}, {"team": c}) for c in ["billing", "technical", "sales"] * 5]
    cal2 = calibrate(eng2, ex2, QC)
    assert cal2.apply(eng2.decide({"claim": "sales"}, QC))["answers"]["team"]["choice"] == "sales"


# ---------------------------------------------------------------------- the edges

def test_too_few_examples_leaves_the_question_alone_and_says_so():
    cal = calibrate(_Overconfident(), _boolean_examples(n=4), QB)
    assert cal.report["churn"]["calibrated"] is False
    assert "at least" in cal.report["churn"]["reason"]
    raw = _Overconfident().decide({"claim": "true"}, QB)
    assert cal.apply(raw)["answers"]["churn"] == raw["answers"]["churn"]


def test_a_label_outside_the_options_is_named():
    ex = _boolean_examples(n=12)
    ex[3] = (ex[3][0], {"churn": "maybe"})
    cal = calibrate(_Overconfident(), ex, QB)
    assert cal.report["churn"]["calibrated"] is False
    assert "'maybe'" in cal.report["churn"]["reason"]


def test_ece_is_zero_for_a_perfectly_honest_oracle():
    assert ece([(1.0, True)] * 10) == 0.0
    assert ece([(0.5, True), (0.5, False)] * 10) == 0.0
    with pytest.raises(ValueError):
        ece([])


# ---------------------------------------------------------------------- persistence

def test_save_load_round_trip(tmp_path):
    cal = calibrate(_Overconfident(), _boolean_examples(), QB)
    p = tmp_path / "calibration.json"
    cal.save(p)
    loaded = Calibration.load(p)
    raw = _Overconfident().decide({"claim": "false"}, QB)
    assert loaded.apply(raw) == cal.apply(raw)
    assert loaded.alpha == cal.alpha


# ---------------------------------------------------------------------- refusals

def test_the_refusals_name_the_problem():
    with pytest.raises(ValueError, match="no examples"):
        calibrate(_Overconfident(), [], QB)
    with pytest.raises(ValueError, match="miscoverage"):
        calibrate(_Overconfident(), _boolean_examples(), QB, alpha=1.5)


# ------------------------------------------------- the engine's own boolean spelling

class _EngineSpelled:
    """Booleans exactly as the real engine answers them: "False"/"True" probs, bool set."""

    def decide(self, state, questions, mode=None, alpha="0.1", views=1):
        p = 0.97 if state["claim"] == "true" else 0.03
        return {"answers": {q: {"probs": {"False": 1 - p, "True": p},
                                "prediction_set": [p >= 0.5], "abstain": False,
                                "boolean": p, "confidence": max(p, 1 - p)}
                            for q in questions}}


def test_the_engines_boolean_spelling_is_understood():
    rng = random.Random(3)
    ex = []
    for _ in range(30):
        claim = rng.choice(["false", "true"])
        truth = claim if rng.random() < 0.7 else ("true" if claim == "false" else "false")
        ex.append(({"claim": claim}, {"churn": truth}))
    cal = calibrate(_EngineSpelled(), ex, QB)                  # KeyError before the fix
    assert cal.report["churn"]["calibrated"]
    a = cal.apply(_EngineSpelled().decide({"claim": "true"}, QB))["answers"]["churn"]
    assert set(a["probs"]) == {"false", "true"}                # re-keyed onto the option ids
    assert a["boolean"] == a["probs"]["true"]


def test_a_probability_key_naming_no_option_is_refused():
    from vegaml.calibrate import canon_probs
    with pytest.raises(ValueError, match="names no option"):
        canon_probs({"yes": 0.9, "no": 0.1}, ["false", "true"])


# ---------------------------------------------------------------------- readout

class _TwoReadouts:
    """Answers differently in "engine" and "auto", so a test can tell which was calibrated."""

    def __init__(self):
        self.modes = []

    def decide(self, state, questions, mode=None, alpha="0.1", views=1):
        self.modes.append(mode)
        # The engine is mildly overconfident; the head is wildly so, which is the real shape and
        # the reason calibrating the served readout matters.
        p = 0.70 if mode == "engine" else 0.99
        claimed = state["claim"]
        probs = {k: (p if k == claimed else 1.0 - p) for k in ("false", "true")}
        answer = {"probs": probs}
        if mode != "engine":
            answer["readout"] = "ttt"
        return {"answers": {q: dict(answer) for q in questions}}


def test_the_readout_decides_which_answers_are_calibrated():
    model = _TwoReadouts()
    cal = calibrate(model, _boolean_examples(n=60), QB, readout="auto")
    assert set(model.modes) == {"auto"}
    assert cal.provenance["readout"] == "auto"
    assert cal.report["churn"]["readout"] == "auto"
    # The answers said they came from a head, and the report records that per question.
    assert cal.report["churn"]["served_by"] == "ttt"


def test_the_engine_is_still_the_default():
    model = _TwoReadouts()
    cal = calibrate(model, _boolean_examples(n=60), QB)
    assert set(model.modes) == {"engine"}
    assert cal.provenance["readout"] == "engine"


def test_calibrating_the_wilder_readout_fits_the_stronger_temperature():
    """A head's raw softmax needs more correction than the engine's, which is the whole point."""
    engine = calibrate(_TwoReadouts(), _boolean_examples(n=120), QB)
    head = calibrate(_TwoReadouts(), _boolean_examples(n=120), QB, readout="auto")
    assert head.report["churn"]["temperature"] > engine.report["churn"]["temperature"]
    assert head.report["churn"]["ece_before"] > engine.report["churn"]["ece_before"]
    # And both end up honest, which is what a band is then allowed to cut on.
    assert head.report["churn"]["ece_after"] < head.report["churn"]["ece_before"]


def test_a_readout_that_is_not_a_mode_is_refused():
    with pytest.raises(ValueError, match="readout must be one of"):
        calibrate(_TwoReadouts(), _boolean_examples(n=20), QB, readout="physics")


def test_both_is_refused_because_there_are_two_answers_to_calibrate():
    with pytest.raises(ValueError, match="no one of them to calibrate"):
        calibrate(_TwoReadouts(), _boolean_examples(n=20), QB, readout="both")
