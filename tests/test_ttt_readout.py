"""The TTT readout: crashes that named nothing, and a cross-validation that could flatter itself.

Three groups of defects, all on the path `mode="ttt"` and the head-selecting half of
`mode="auto"`.

1. Crashes with messages that identified nothing. A head asked for labels it was never fitted on
   raised `max() arg is an empty sequence`; `fit` raised a bare `KeyError('zzz')`; `predict` on an
   unfitted head raised `AttributeError` on `self.mu`.

2. Cross-validation cut folds from a single random permutation, so a class with few examples could
   be absent from a training fold entirely. The head for that fold was then fitted on a label space
   it could not predict, and the resulting accuracy was still reported as if it measured the head.

3. Every head answer was keyed `"choice"`, whatever the question type. `mode="auto"` is documented
   as reproducing the engine's answer shape, but a boolean question answered by a head had no
   `"boolean"` field, so a caller reading a mixed result needed two code paths.
"""
from __future__ import annotations

import numpy as np
import pytest

from vegaml import TTTHead, Vega


class _Engine:
    """Stands in for a loaded checkpoint: the mode switch's surface, none of the weights."""

    cfg = {"max_len": 73728}
    model_dir = "/nowhere"
    backbone = "dummy"

    def decide(self, state, questions, alpha="0.1", views=1, **kw):
        """A stand-in that answers in the engine's real shape, including per-type keys."""
        out = {}
        for qid, q in questions.items():
            t = q["type"]
            if t == "choice":
                labels = list(q["criteria"])
            elif t == "score":
                labels = list(range(len(q["criteria"])))
            else:
                labels = [False, True]
            probs = {str(labels[i]): 1.0 / len(labels) for i in range(len(labels))}
            a = {"probs": probs, "confidence": max(probs.values())}
            if t == "choice":
                a["choice"] = labels[0]
            elif t == "score":
                a["score"] = 0
                a["expected_score"] = 0.0
            else:
                a["boolean"] = 0.5
            out[qid] = a
        return {"answers": out}


def _separable(n=40, d=48, classes=("a", "b", "c"), seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(len(classes), d)) * 3.0
    X, y = [], []
    for i, c in enumerate(classes):
        X.append(centres[i] + rng.normal(scale=0.35, size=(n, d)))
        y += [c] * n
    return np.vstack(X), y


def _vega_with_head(mode="ttt", labels=("a", "b"), cv=0.95):
    v = Vega(_Engine(), mode=mode)
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(size=(20, 16)) + 4, rng.normal(size=(20, 16)) - 4])
    v.heads["q"] = (TTTHead(list(labels)).fit(X, [labels[0]] * 20 + [labels[1]] * 20), cv)
    v.features = lambda state, q: X[0]
    return v


BOOLEAN = {"type": "boolean", "instructions": "does it hold?"}
SCORE3 = {"type": "score", "instructions": "how bad?", "criteria": ["low", "mid", "high"]}


# ------------------------------------------------------------------ 1. crashes that named nothing
def test_a_head_asked_for_labels_it_never_saw_names_both_sides():
    """It raised `max() arg is an empty sequence`, naming neither the head nor the question."""
    v = _vega_with_head("ttt", labels=("a", "b"))
    with pytest.raises(RuntimeError) as excinfo:
        v.decide("s", {"q": BOOLEAN})
    message = str(excinfo.value)
    assert "q" in message, "the message must name the question"
    assert "a" in message and "b" in message, "and what the head was fitted on"
    assert "true" in message, "and what the question actually offers"


def test_fit_names_the_offending_label_and_the_head_label_set():
    X, _ = _separable()
    with pytest.raises(ValueError) as excinfo:
        TTTHead(["a", "b"]).fit(X, ["a"] * 20 + ["b"] * 19 + ["zzz"])
    message = str(excinfo.value)
    assert "zzz" in message, "the offending label"
    assert "a" in message and "b" in message, "and the label set it belongs to"


def test_predicting_with_an_unfitted_head_says_so():
    with pytest.raises(RuntimeError, match="not been fitted"):
        TTTHead(["a", "b"]).predict(np.zeros(4))


def test_predicting_with_an_allowed_set_that_shares_nothing_says_so():
    X, y = _separable(classes=("a", "b"))
    head = TTTHead(["a", "b"]).fit(X, y)
    with pytest.raises(ValueError) as excinfo:
        head.predict(X[0], allowed=["x", "y"])
    message = str(excinfo.value)
    assert "no overlap" in message and "x" in message


def test_fit_reports_a_label_outside_the_question_options_instead_of_raising():
    """`fit` has a report per question, so this belongs in the report, not in a traceback."""
    v = Vega(_Engine())
    v.features = lambda state, q: np.random.default_rng(0).normal(size=16)
    qs = {"q": {"type": "choice", "instructions": "which?", "criteria": {"a": "", "b": ""}}}
    report = v.fit([("s%d" % i, {"q": "zzz"}) for i in range(12)], qs)
    assert report["q"]["fitted"] is False
    assert "zzz" in report["q"]["reason"] and "a" in report["q"]["reason"]
    assert "q" not in v.heads


def test_fit_reports_a_single_class_question_rather_than_fitting_nothing():
    v = Vega(_Engine())
    v.features = lambda state, q: np.random.default_rng(0).normal(size=16)
    qs = {"q": {"type": "choice", "instructions": "which?", "criteria": {"a": "", "b": ""}}}
    report = v.fit([("s%d" % i, {"q": "a"}) for i in range(12)], qs)
    assert report["q"]["fitted"] is False
    assert "same label" in report["q"]["reason"]
    assert "q" not in v.heads


# ------------------------------------------------------------------ 2. cross-validation
def test_every_example_is_still_scored_exactly_once_on_imbalanced_data():
    """Folds used to come from one permutation, so a rare class could miss a training fold."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 16))
    y = ["a"] * 55 + ["b"] * 5
    head = TTTHead(["a", "b"])
    acc = head.cross_val_accuracy(X, y, folds=5)
    assert acc is not None
    assert head.cross_val_n == 60, "every example must still be scored exactly once"


def test_a_class_too_rare_to_stratify_is_excluded_from_the_estimate():
    """5 folds needs 5 members per class to place one in each. Fewer and it cannot be stratified."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(62, 8))
    y = ["a"] * 40 + ["b"] * 20 + ["c"] * 2
    head = TTTHead(["a", "b", "c"])
    acc = head.cross_val_accuracy(X, y, folds=5)
    assert acc is not None
    assert head.cross_val_n == 60, "the two rare examples are excluded; the other 60 are scored once"


def test_cross_val_returns_none_when_no_class_can_fill_the_folds():
    """Returning a number here would be a head scoring itself on data it was fitted on."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 8))
    y = ["a"] * 29 + ["b"]
    head = TTTHead(["a", "b"])
    assert head.cross_val_accuracy(X, y, folds=5) is None


def test_cross_val_n_is_readable_before_any_estimate_has_been_run():
    head = TTTHead(["a", "b"])
    assert head.cross_val_n == 0, "reading it must never be an AttributeError"


def test_cross_val_is_deterministic():
    """A fixed seed, so the same examples always produce the same reported accuracy."""
    X, y = _separable()
    first = TTTHead(["a", "b", "c"]).cross_val_accuracy(X, y)
    second = TTTHead(["a", "b", "c"]).cross_val_accuracy(X, y)
    assert first == second


def test_cross_val_still_high_on_separable_data_and_low_on_noise():
    X, y = _separable(classes=("a", "b", "c"))
    assert TTTHead(["a", "b", "c"]).cross_val_accuracy(X, y) > 0.9
    Xn = np.random.default_rng(1).normal(size=(80, 48))
    yn = [["a", "b"][i % 2] for i in range(80)]
    assert TTTHead(["a", "b"]).cross_val_accuracy(Xn, yn) < 0.75, \
        "labels independent of features: the number that tells a caller the head learned nothing"


@pytest.mark.parametrize("n_adapters", [1, 2, 3])
def test_cross_val_survives_a_class_with_one_example(n_adapters):
    """The degenerate case that used to be reachable: a class that cannot fill any fold."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 8))
    y = ["a"] * (19 - n_adapters) + ["b"] * n_adapters + ["rare"]
    head = TTTHead(["a", "b", "rare"])
    acc = head.cross_val_accuracy(X, y, folds=3)
    assert acc is None or 0.0 <= acc <= 1.0, "never a crash and never a nonsense number"


# ------------------------------------------------------------------ 3. the answer shape
def test_a_boolean_question_answered_by_a_head_carries_its_boolean_field():
    """The engine keys a boolean answer "boolean"; auto is documented as reproducing that shape."""
    v = _vega_with_head("ttt", labels=("false", "true"))
    ans = v.decide("s", {"q": BOOLEAN})["answers"]["q"]
    assert "boolean" in ans
    assert set(ans["probs"]) == {"false", "true"}
    assert ans["boolean"] == pytest.approx(ans["probs"]["true"])


def test_a_score_question_answered_by_a_head_carries_score_and_expected_score():
    v = _vega_with_head("ttt", labels=("0", "1", "2"))
    ans = v.decide("s", {"q": SCORE3})["answers"]["q"]
    assert ans["score"] in (0, 1, 2)
    assert "expected_score" in ans
    assert 0.0 <= ans["expected_score"] <= 2.0


def test_a_choice_question_still_carries_choice():
    v = _vega_with_head("ttt", labels=("a", "b"))
    qs = {"q": {"type": "choice", "instructions": "which?", "criteria": {"a": "", "b": ""}}}
    ans = v.decide("s", qs)["answers"]["q"]
    assert ans["choice"] in ("a", "b")


def test_auto_answers_carry_the_same_keys_regardless_of_which_readout_produced_them():
    """The whole point of "auto is the engine, flags and all": the shape must not change."""
    v = _vega_with_head("auto", labels=("a", "b"))
    fitted = v.decide("s", {"q": {"type": "choice", "instructions": "which?",
                                  "criteria": {"a": "", "b": ""}}})["answers"]["q"]
    assert fitted["readout"] == "ttt"
    assert "choice" in fitted and "confidence" in fitted

    unfitted = v.decide("s", {"other": {"type": "boolean", "instructions": "?"}})["answers"]["other"]
    assert unfitted["readout"] == "engine"
    # The engine answer carries the calibration fields the head answer deliberately does not, which
    # is the documented difference between the two readouts -- but the answer-type key matches.
    assert "boolean" in unfitted and "choice" in fitted


def test_auto_marks_every_answer_with_the_readout_that_produced_it():
    v = _vega_with_head("auto", labels=("a", "b"))
    questions = {"q": {"type": "choice", "instructions": "which?", "criteria": {"a": "", "b": ""}},
                 "other": {"type": "choice", "instructions": "which?", "criteria": {"a": "", "c": ""}}}
    out = v.decide("s", questions)
    assert out["answers"]["q"]["readout"] == "ttt"
    assert out["answers"]["other"]["readout"] == "engine"
    assert [k for k in out["answers"]] == ["q", "other"], "asked order is kept"


def test_a_head_answer_is_marked_ttt_even_in_ttt_mode():
    v = _vega_with_head("ttt", labels=("a", "b"))
    qs = {"q": {"type": "choice", "instructions": "which?", "criteria": {"a": "", "b": ""}}}
    assert v.decide("s", qs)["answers"]["q"]["readout"] == "ttt"
