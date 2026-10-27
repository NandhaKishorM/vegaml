"""The review band: the bound, the fitting, the routing, and the gap it exists to close."""
from __future__ import annotations

import random

import pytest

from vegaml.band import Band, crossfit, fit_threshold, sweep, wilson_lower


def _rows(n=300, seed=1, floor=0.34, ceiling=0.99, lift=0.1):
    """Confidence spread over the range, correctness rising with it: a usable band exists."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        c = rng.uniform(floor, ceiling)
        out.append((c, rng.random() < min(0.995, c + lift)))
    return out


# ---------------------------------------------------------------------- the bound

def test_the_bound_makes_a_narrow_band_pay_for_being_narrow():
    assert wilson_lower(12, 12) < wilson_lower(120, 120) < 1.0
    assert wilson_lower(0, 0) == 0.0
    assert wilson_lower(50, 100) < 0.5


def test_a_tighter_confidence_is_a_lower_bound():
    assert wilson_lower(90, 100, 0.99) < wilson_lower(90, 100, 0.90)


def test_an_unsupported_confidence_is_refused_rather_than_rounded():
    with pytest.raises(ValueError, match="confidence must be one of"):
        wilson_lower(9, 10, 0.937)


# ---------------------------------------------------------------------- the sweep

def test_the_sweep_starts_at_everything_automatic_and_narrows():
    points = sweep(_rows(n=60))
    assert points[0]["threshold"] == 0.0 and points[0]["review_rate"] == 0.0
    assert points[0]["n_auto"] == 60
    rates = [p["review_rate"] for p in points]
    assert rates == sorted(rates)                      # a higher line reviews at least as much


def test_the_sweep_refuses_an_empty_corpus():
    with pytest.raises(ValueError, match="no rows"):
        sweep([])


# ---------------------------------------------------------------------- fitting

def test_a_precision_target_is_held_on_the_bound_not_on_the_point_estimate():
    cut = fit_threshold(_rows(), target_precision=0.95)
    assert cut["reachable"]
    assert cut["precision_lb"] >= 0.95
    assert cut["precision"] >= cut["precision_lb"]
    assert 0.0 < cut["review_rate"] < 1.0


def test_a_higher_target_never_costs_less_review():
    rows = _rows()
    low = fit_threshold(rows, target_precision=0.80)
    high = fit_threshold(rows, target_precision=0.95)
    assert high["review_rate"] >= low["review_rate"]
    assert high["threshold"] >= low["threshold"]


def test_an_unreachable_target_escalates_everything_and_says_what_was_possible():
    cut = fit_threshold(_rows(), target_precision=0.9999)
    assert not cut["reachable"]
    assert cut["review_rate"] == 1.0 and cut["n_auto"] == 0
    assert cut["best_achievable"]["precision_lb"] < 0.9999
    assert "every answer goes to review" in cut["reason"]


def test_a_review_budget_is_a_ceiling_on_the_reviewers_needed():
    cut = fit_threshold(_rows(), review_rate=0.2)
    assert cut["reachable"] and cut["review_rate"] >= 0.2
    assert cut["review_rate"] < 0.3                 # and it does not overshoot by much
    assert cut["target_precision"] is None


def test_a_review_budget_caps_what_a_precision_target_may_spend():
    rows = _rows()
    free = fit_threshold(rows, target_precision=0.95)
    capped = fit_threshold(rows, target_precision=0.95, max_review_rate=0.1)
    assert free["reachable"] and free["review_rate"] > 0.1
    assert not capped["reachable"]                  # the target cannot be met inside the budget
    assert capped["best_achievable"]["review_rate"] <= 0.1


def test_exactly_one_target_is_required():
    rows = _rows(n=40)
    with pytest.raises(ValueError, match="exactly one"):
        fit_threshold(rows)
    with pytest.raises(ValueError, match="exactly one"):
        fit_threshold(rows, target_precision=0.9, review_rate=0.2)


def test_the_held_out_estimate_is_measured_on_rows_the_threshold_did_not_see():
    rows = _rows()
    held = crossfit(rows, target_precision=0.95)
    assert held["folds"] == 5 and held["n"] == len(rows)
    assert 0.0 < held["precision"] <= 1.0
    assert crossfit(_rows(n=8), target_precision=0.95) is None      # too little to split


# ---------------------------------------------------------------------- routing

def test_a_band_from_configured_thresholds_needs_no_labels():
    band = Band.from_thresholds({"a": 0.7}, default=0.5)
    assert band.threshold_for("a") == 0.7
    assert band.threshold_for("b") == 0.5           # the default covers the rest
    assert Band.from_thresholds({"a": 0.7}).threshold_for("b") is None


def test_a_band_with_no_threshold_is_refused_at_construction():
    with pytest.raises(ValueError, match="names no threshold"):
        Band({"a": {"target_precision": 0.9}})


def test_routing_names_the_line_it_compared_against():
    band = Band.from_thresholds({"q": 0.72})
    low = band.route({"confidence": 0.61}, "q")
    high = band.route({"confidence": 0.9}, "q")
    assert low["route"] == "review" and low["review"] is True
    assert "0.610 is below the 0.720 review line" in low["route_reason"]
    assert high["route"] == "auto" and high["review"] is False


def test_a_question_nobody_configured_is_marked_rather_than_auto_approved():
    band = Band.from_thresholds({"q": 0.72})
    out = band.route({"confidence": 0.1}, "other")
    assert out["route"] == "unbanded" and out["review"] is False
    assert "no review line is configured" in out["route_reason"]


def test_an_answer_with_no_confidence_cannot_be_routed():
    assert Band(default=0.5).route({"choice": "a"}, "q")["route"] == "unbanded"


def test_the_band_fires_on_an_answer_whose_conformal_set_can_never_be_empty():
    """The whole reason this module exists.

    This answer abstains on nothing: the conformal set holds every option, so the published
    abstain flag is False and always would be. The band still escalates it, because it reads
    confidence and not the set.
    """
    answer = {"confidence": 0.41, "abstain": False, "prediction_set": ["a", "b", "c"],
              "probs": {"a": 0.41, "b": 0.33, "c": 0.26}, "choice": "a"}
    routed = Band(default=0.6).apply({"answers": {"q": answer}})
    assert routed["answers"]["q"]["route"] == "review"
    assert routed["answers"]["q"]["abstain"] is False       # untouched, and still useless
    assert routed["band"] == {"auto": 0, "review": 1, "unbanded": [], "default": 0.6}


def test_the_band_covers_a_head_answer_which_carries_no_abstain_flag_at_all():
    out = Band(default=0.6).apply({"mode": "both",
                                   "answers": {"q": {"confidence": 0.95}},
                                   "ttt": {"q": {"confidence": 0.5, "readout": "ttt"}}})
    assert out["answers"]["q"]["route"] == "auto"
    assert out["ttt"]["q"]["route"] == "review"
    assert out["band"]["auto"] == 1 and out["band"]["review"] == 1


def test_apply_leaves_the_input_untouched():
    answers = {"q": {"confidence": 0.3}}
    result = {"answers": answers}
    Band(default=0.6).apply(result)
    assert "route" not in answers["q"] and "band" not in result


def test_the_summary_names_every_unbanded_question_once():
    out = Band.from_thresholds({"a": 0.5}).apply(
        {"answers": {"a": {"confidence": 0.9}, "b": {"confidence": 0.9}},
         "ttt": {"b": {"confidence": 0.9}}})
    assert out["band"]["unbanded"] == ["b"]
    assert out["band"]["auto"] == 1 and out["band"]["unbanded"] is not None


# ---------------------------------------------------------------------- persistence

def test_a_band_survives_a_round_trip(tmp_path):
    band = Band.from_thresholds({"a": 0.7}, default=0.5)
    path = tmp_path / "band.json"
    band.save(path)
    back = Band.load(path)
    assert back.questions == band.questions and back.default == band.default
    assert back.route({"confidence": 0.6}, "a")["route"] == "review"
