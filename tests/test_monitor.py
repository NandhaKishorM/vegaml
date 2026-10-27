"""The drift watch: what it calls stable, what it pages on, and what it refuses to judge."""
from __future__ import annotations

import random

import pytest

from vegaml.monitor import Monitor, psi

REF = {"team": {"n": 400, "predicted_mix": {"billing": 0.5, "technical": 0.3, "sales": 0.2},
                "mean_confidence": 0.74, "review_rate": 0.2, "accuracy": 0.81}}


def _traffic(mix, confidence=0.74, n=400, line=0.6, seed=3):
    rng = random.Random(seed)
    options, weights = zip(*mix.items(), strict=True)
    for _ in range(n):
        c = min(0.999, max(0.34, rng.gauss(confidence, 0.12)))
        yield {"answers": {"team": {"choice": rng.choices(options, weights)[0], "confidence": c,
                                    "route": "review" if c < line else "auto"}}}


def _watch(mix, **kw):
    mon = Monitor(REF)
    for result in _traffic(mix, **kw):
        mon.observe(result)
    return mon.report()


# ---------------------------------------------------------------------- the statistic

def test_the_index_is_zero_only_when_the_mixes_agree():
    mix = {"a": 0.5, "b": 0.5}
    assert psi(mix, mix) == 0.0
    assert psi(mix, {"a": 0.6, "b": 0.4}) > 0.0


def test_the_index_is_symmetric_and_survives_an_option_that_stopped_appearing():
    a, b = {"a": 0.5, "b": 0.5}, {"a": 0.9, "b": 0.1}
    assert psi(a, b) == pytest.approx(psi(b, a))
    assert psi({"a": 0.5, "b": 0.5}, {"a": 1.0}) > 0.25      # finite, not infinite


# ---------------------------------------------------------------------- the verdicts

def test_steady_traffic_is_left_alone():
    report = _watch({"billing": 0.5, "technical": 0.3, "sales": 0.2})
    assert report["status"] == "ok"
    assert report["questions"]["team"]["psi"] < 0.1


def test_a_prior_shift_is_what_this_pages_on():
    report = _watch({"billing": 0.15, "technical": 0.25, "sales": 0.6})
    q = report["questions"]["team"]
    assert report["status"] == "recalibrate"
    assert q["psi"] >= 0.25 and "answer mix has moved" in q["reason"]


def test_a_mild_drift_is_a_watch_and_not_a_page():
    mon = Monitor(REF, psi_watch=0.01, psi_alert=0.25)
    for result in _traffic({"billing": 0.4, "technical": 0.35, "sales": 0.25}):
        mon.observe(result)
    q = mon.report()["questions"]["team"]
    assert q["status"] == "watch" and "drifting" in q["reason"]


def test_a_collapse_in_confidence_is_caught_even_when_the_mix_holds():
    report = _watch({"billing": 0.5, "technical": 0.3, "sales": 0.2}, confidence=0.58)
    q = report["questions"]["team"]
    assert report["status"] == "recalibrate"
    assert q["psi"] < 0.1                                # the mix never moved
    assert "mean confidence has fallen" in q["reason"]
    assert "review load has risen" in q["reason"]        # and the band is costing money


def test_an_option_the_calibration_never_produced_is_a_different_question():
    report = _watch({"billing": 0.4, "technical": 0.3, "sales": 0.2, "fraud": 0.1})
    q = report["questions"]["team"]
    assert q["status"] == "recalibrate"
    assert "'fraud'" in q["reason"] and "never produced" in q["reason"]


def test_too_few_answers_is_measured_but_not_acted_on():
    mon = Monitor(REF)
    for result in _traffic({"billing": 0.1, "technical": 0.1, "sales": 0.8}, n=20):
        mon.observe(result)
    q = mon.report()["questions"]["team"]
    assert q["status"] == "insufficient" and q["psi"] > 0.25
    assert "under the 50 this monitor will call a verdict on" in q["reason"]
    assert mon.report()["status"] == "insufficient"


def test_a_question_with_no_reference_is_named_rather_than_judged():
    mon = Monitor(REF)
    mon.observe({"answers": {"sentiment": {"choice": "positive", "confidence": 0.9}}})
    q = mon.report()["questions"]["sentiment"]
    assert q["status"] == "insufficient" and "not part of the calibration" in q["reason"]


def test_the_worst_question_sets_the_top_level_status():
    ref = dict(REF)
    ref["churn"] = {"n": 100, "predicted_mix": {"false": 0.5, "true": 0.5}, "mean_confidence": 0.7}
    mon = Monitor(ref)
    for result in _traffic({"billing": 0.5, "technical": 0.3, "sales": 0.2}):
        mon.observe(result)
    for _ in range(80):
        mon.observe({"answers": {"churn": {"boolean": 0.95, "confidence": 0.95}}})
    report = mon.report()
    assert report["questions"]["team"]["status"] == "ok"
    assert report["questions"]["churn"]["status"] == "recalibrate"      # every answer is "true"
    assert report["status"] == "recalibrate"


# ---------------------------------------------------------------------- reading answers

def test_every_answer_type_and_readout_is_counted():
    mon = Monitor({"q": {"predicted_mix": {}}})
    mon.observe({"answers": {"q": {"choice": "a", "confidence": 0.9}}})
    mon.observe({"answers": {"q": {"score": 2, "confidence": 0.9}}})
    mon.observe({"answers": {"q": {"boolean": 0.8, "confidence": 0.9}}})
    mon.observe({"answers": {"q": {"boolean": 0.2, "confidence": 0.9}}})
    mon.observe({"q": {"probs": {"B": 0.7, "a": 0.3}}})       # a bare answers mapping
    assert mon.counts["q"]["labels"] == {"a": 1, "2": 1, "true": 1, "false": 1, "b": 1}


def test_something_that_is_not_a_result_is_refused():
    with pytest.raises(ValueError, match="decide\\(\\) result"):
        Monitor(REF).observe({"answers": "not a mapping"})


def test_a_calibration_with_no_reference_cannot_be_watched():
    class _Old:
        reference = {}

    with pytest.raises(ValueError, match="no reference mix"):
        Monitor.from_calibration(_Old())


# ---------------------------------------------------------------------- persistence

def test_counters_survive_a_restart(tmp_path):
    mon = Monitor(REF)
    for result in _traffic({"billing": 0.5, "technical": 0.3, "sales": 0.2}, n=100):
        mon.observe(result)
    path = tmp_path / "state.json"
    mon.save_state(path)
    back = Monitor.load_state(path)
    assert back.counts == mon.counts
    assert back.report()["questions"]["team"]["psi"] == mon.report()["questions"]["team"]["psi"]
    for result in _traffic({"billing": 0.5, "technical": 0.3, "sales": 0.2}, n=100, seed=9):
        back.observe(result)
    assert back.report()["questions"]["team"]["n"] == 200       # it accumulated, not restarted
