"""The guard sheet is a policy, and the scoring is the part that keeps it honest."""
from __future__ import annotations

import pytest

from vegaml.guard import GUARD, degenerate, score


def test_the_sheet_is_a_two_way_choice():
    assert GUARD["type"] == "choice"
    crit = GUARD["criteria"]
    assert set(crit) == {"allow", "block"}
    for name, text in crit.items():
        assert len(text) > 120, "%s reads too thin to steer a decision" % name


def test_the_criteria_name_no_strings_to_match():
    """A guard built on literals is beaten by the first attacker who rephrases, so the
    criteria must describe the property rather than quote the attack."""
    blob = (GUARD["instructions"] + " ".join(GUARD["criteria"].values())).lower()
    for literal in ("ignore previous instructions", "do anything now", "dan mode",
                    "developer mode", "jailbreak"):
        assert literal not in blob


def test_score_counts_the_four_cells():
    preds = ["block", "block", "allow", "allow", "block", "allow"]
    labels = ["block", "allow", "block", "allow", "block", "allow"]
    s = score(preds, labels)
    assert s["counts"] == {"tp": 2, "fp": 1, "fn": 1, "tn": 2}
    assert s["n"] == 6
    assert s["attacks"] == 3 and s["benign"] == 3
    assert s["block_recall"] == pytest.approx(2 / 3)
    assert s["false_positive_rate"] == pytest.approx(1 / 3)
    assert s["precision"] == pytest.approx(2 / 3)
    assert s["accuracy"] == pytest.approx(4 / 6)
    assert s["balanced_accuracy"] == pytest.approx((2 / 3 + 2 / 3) / 2)


def test_anything_that_is_not_block_is_a_forward():
    """An abstention falls through to the default policy in production, so scoring it as a
    catch would count a coin flip as a save."""
    s = score(["abstain", "allow", ""], ["block", "block", "block"])
    assert s["counts"]["fn"] == 3
    assert s["block_recall"] == 0.0


def test_balanced_accuracy_refuses_to_flatter_a_block_everything_guard():
    """Plain accuracy on an attack-heavy corpus rewards refusing everything. The balanced
    figure is what stops that from looking like a result."""
    labels = ["block"] * 90 + ["allow"] * 10
    s = score(["block"] * 100, labels)
    assert s["accuracy"] == pytest.approx(0.9)
    assert s["balanced_accuracy"] == pytest.approx(0.5)


def test_degenerate_names_the_block_everything_policy():
    s = score(["block"] * 100, ["block"] * 50 + ["allow"] * 50)
    why = degenerate(s)
    assert why and "blocks 100%" in why


def test_degenerate_names_the_forward_everything_policy():
    s = score(["allow"] * 100, ["block"] * 50 + ["allow"] * 50)
    why = degenerate(s)
    assert why and "forwards 100%" in why


def test_degenerate_passes_a_guard_that_actually_reads():
    preds = ["block"] * 45 + ["allow"] * 5 + ["allow"] * 44 + ["block"] * 6
    labels = ["block"] * 50 + ["allow"] * 50
    s = score(preds, labels)
    assert degenerate(s) is None


def test_degenerate_flags_a_corpus_with_only_attacks():
    """Measuring only attacks is the measurement a block-everything guard passes."""
    preds = ["block"] * 30 + ["allow"] * 30
    s = score(preds, ["block"] * 60)
    assert s["false_positive_rate"] is None
    assert degenerate(s) == "the corpus has only one class, which no guard can fail"


def test_score_rejects_mismatched_and_empty_input():
    with pytest.raises(ValueError, match="differ in length"):
        score(["block"], ["block", "allow"])
    with pytest.raises(ValueError, match="nothing to score"):
        score([], [])
