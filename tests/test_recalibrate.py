"""The vegaml-calibrate command: one command to refit, one to ask whether you need to."""
from __future__ import annotations

import json
import random

import pytest

from vegaml import recalibrate

QUESTIONS = {"churn": {"type": "boolean", "instructions": "Does the customer threaten to leave?"}}


class _Graded:
    """Claims a confidence the state carries, and is right that often: a fittable band exists."""

    def decide(self, state, questions, mode=None, alpha="0.1", views=1):
        p = float(state["p"])
        probs = {"true": p, "false": 1.0 - p} if state["says"] == "true" else {"true": 1.0 - p,
                                                                              "false": p}
        return {"answers": {q: {"probs": dict(probs)} for q in questions}}


@pytest.fixture
def fake(monkeypatch):
    import vegaml

    loads = []

    def load(repo, **kw):
        loads.append({"repo": repo, **kw})
        return _Graded()

    monkeypatch.setattr(vegaml, "load", load)
    return loads


def _labels_file(tmp_path, n=200, seed=4):
    rng = random.Random(seed)
    path = tmp_path / "labels.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(n):
            says = rng.choice(["true", "false"])
            p = rng.uniform(0.5, 0.99)
            truth = says if rng.random() < p else ("false" if says == "true" else "true")
            f.write(json.dumps({"state": {"says": says, "p": round(p, 4)},
                                "labels": {"churn": truth}}) + "\n")
    return path


def _questions_file(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(QUESTIONS))
    return path


# ---------------------------------------------------------------------- fit

def test_fit_writes_one_file_with_temperature_conformal_band_and_reference(tmp_path, fake, capsys):
    out = tmp_path / "calibration.json"
    code = recalibrate.main(["fit", "--labels", str(_labels_file(tmp_path)),
                             "--questions", "@%s" % _questions_file(tmp_path),
                             "--target-precision", "0.9", "--out", str(out)])
    assert code == 0
    payload = json.loads(out.read_text())
    assert payload["questions"]["churn"]["temperature"] > 0
    assert "qhat" in payload["questions"]["churn"]
    assert payload["band"]["questions"]["churn"]["threshold"] > 0
    assert payload["reference"]["churn"]["predicted_mix"]
    printed = capsys.readouterr().out
    assert "review band" in printed and "written to" in printed


def test_fit_reports_what_the_precision_target_costs_in_review_load(tmp_path, fake, capsys):
    recalibrate.main(["fit", "--labels", str(_labels_file(tmp_path)),
                      "--questions", "@%s" % _questions_file(tmp_path),
                      "--target-precision", "0.9", "--out", str(tmp_path / "c.json")])
    printed = capsys.readouterr().out
    assert "% of traffic" in printed
    assert "held out, the same rule scored" in printed


def test_fit_takes_a_review_budget_instead_of_a_target(tmp_path, fake):
    out = tmp_path / "c.json"
    recalibrate.main(["fit", "--labels", str(_labels_file(tmp_path)),
                      "--questions", "@%s" % _questions_file(tmp_path),
                      "--review-rate", "0.15", "--out", str(out)])
    band = json.loads(out.read_text())["report"]["churn"]["band"]
    assert band["requested_review_rate"] == 0.15 and band["review_rate"] >= 0.15


def test_fit_refuses_both_targets_at_once(tmp_path, fake):
    with pytest.raises(SystemExit, match="not both"):
        recalibrate.main(["fit", "--labels", str(_labels_file(tmp_path)),
                          "--questions", "@%s" % _questions_file(tmp_path),
                          "--target-precision", "0.9", "--review-rate", "0.2"])


def test_fit_accepts_the_typed_question_flags(tmp_path, fake):
    out = tmp_path / "c.json"
    assert recalibrate.main(["fit", "--labels", str(_labels_file(tmp_path)),
                             "--boolean", "churn=Does the customer threaten to leave?",
                             "--out", str(out)]) == 0
    assert "churn" in json.loads(out.read_text())["questions"]


def test_fit_takes_the_sheet_off_the_rows_when_no_flag_carries_one(tmp_path, fake):
    path = tmp_path / "labels.jsonl"
    rng = random.Random(7)
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(60):
            says = rng.choice(["true", "false"])
            f.write(json.dumps({"state": {"says": says, "p": 0.8},
                                "labels": {"churn": says},
                                "questions": QUESTIONS}) + "\n")
    out = tmp_path / "c.json"
    assert recalibrate.main(["fit", "--labels", str(path), "--out", str(out)]) == 0
    assert "churn" in json.loads(out.read_text())["questions"]


def test_fit_with_no_questions_anywhere_says_where_to_put_them(tmp_path, fake):
    with pytest.raises(SystemExit, match="no questions"):
        recalibrate.main(["fit", "--labels", str(_labels_file(tmp_path)),
                          "--out", str(tmp_path / "c.json")])


def test_fit_exits_non_zero_when_a_question_could_not_be_calibrated(tmp_path, fake):
    path = tmp_path / "labels.jsonl"
    path.write_text("\n".join(json.dumps({"state": {"says": "true", "p": 0.8},
                                          "labels": {"churn": "true"}}) for _ in range(4)))
    assert recalibrate.main(["fit", "--labels", str(path),
                             "--questions", "@%s" % _questions_file(tmp_path),
                             "--out", str(tmp_path / "c.json")]) == 1


def test_a_labels_line_without_labels_names_the_line(tmp_path, fake):
    path = tmp_path / "labels.jsonl"
    path.write_text(json.dumps({"state": "x"}) + "\n")
    with pytest.raises(SystemExit, match="line 1 needs"):
        recalibrate.main(["fit", "--labels", str(path),
                          "--questions", "@%s" % _questions_file(tmp_path)])


def test_a_mistyped_labels_path_is_an_error_not_a_traceback(tmp_path, fake):
    with pytest.raises(SystemExit, match="cannot read"):
        recalibrate.main(["fit", "--labels", str(tmp_path / "nope.jsonl"),
                          "--questions", "@%s" % _questions_file(tmp_path)])


# ---------------------------------------------------------------------- check

def _calibration(tmp_path, fake, capsys=None):
    out = tmp_path / "calibration.json"
    recalibrate.main(["fit", "--labels", str(_labels_file(tmp_path)),
                      "--questions", "@%s" % _questions_file(tmp_path),
                      "--target-precision", "0.9", "--out", str(out)])
    if capsys is not None:
        capsys.readouterr()                      # the fit's own output is not under test here
    return out


def _answers_file(tmp_path, name, true_share, confidence=0.8, n=200, seed=11):
    rng = random.Random(seed)
    path = tmp_path / name
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(n):
            true = rng.random() < true_share
            f.write(json.dumps({"answers": {"churn": {
                "boolean": 0.9 if true else 0.1,
                "confidence": min(0.999, max(0.34, rng.gauss(confidence, 0.1)))}}}) + "\n")
    return path


def test_check_is_quiet_and_exits_zero_when_the_traffic_still_fits(tmp_path, fake, capsys):
    cal = _calibration(tmp_path, fake, capsys)
    ref = json.loads(cal.read_text())["reference"]["churn"]
    answers = _answers_file(tmp_path, "a.jsonl", ref["predicted_mix"].get("true", 0.5),
                            confidence=ref["mean_confidence"])
    code = recalibrate.main(["check", "--answers", str(answers), "--calibration", str(cal)])
    assert code == 0
    assert "status: ok" in capsys.readouterr().out


def test_check_exits_non_zero_on_a_shift_and_prints_the_command_to_run(tmp_path, fake, capsys):
    cal = _calibration(tmp_path, fake, capsys)
    answers = _answers_file(tmp_path, "a.jsonl", 0.97, confidence=0.45)
    code = recalibrate.main(["check", "--answers", str(answers), "--calibration", str(cal)])
    printed = capsys.readouterr().out
    assert code == 1
    assert "status: recalibrate" in printed
    assert "vegaml-calibrate fit --labels" in printed


def test_check_can_be_told_to_fail_on_a_watch(tmp_path, fake):
    cal = _calibration(tmp_path, fake)
    ref = json.loads(cal.read_text())["reference"]["churn"]
    answers = _answers_file(tmp_path, "a.jsonl", 0.70, confidence=ref["mean_confidence"])
    assert recalibrate.main(["check", "--answers", str(answers), "--calibration", str(cal),
                             "--fail-on", "watch"]) == 1


def test_check_carries_counters_across_runs(tmp_path, fake):
    cal = _calibration(tmp_path, fake)
    state = tmp_path / "state.json"
    answers = _answers_file(tmp_path, "a.jsonl", 0.5, n=30)
    recalibrate.main(["check", "--answers", str(answers), "--calibration", str(cal),
                      "--state", str(state), "--json"])
    first = json.loads(state.read_text())["counts"]["churn"]["n"]
    recalibrate.main(["check", "--answers", str(answers), "--calibration", str(cal),
                      "--state", str(state), "--json"])
    assert json.loads(state.read_text())["counts"]["churn"]["n"] == 2 * first


def test_check_prints_json_when_asked(tmp_path, fake, capsys):
    cal = _calibration(tmp_path, fake, capsys)
    answers = _answers_file(tmp_path, "a.jsonl", 0.5)
    recalibrate.main(["check", "--answers", str(answers), "--calibration", str(cal), "--json"])
    report = json.loads(capsys.readouterr().out)
    assert report["status"] in ("ok", "watch", "recalibrate", "insufficient")
    assert report["questions"]["churn"]["n"] == 200


def test_check_against_a_calibration_from_before_bands_existed(tmp_path, fake):
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"alpha": 0.1, "report": {},
                               "questions": {"churn": {"type": "boolean",
                                                       "labels": ["false", "true"],
                                                       "temperature": 1.0, "qhat": 0.5,
                                                       "examples": 40}}}))
    answers = _answers_file(tmp_path, "a.jsonl", 0.5)
    with pytest.raises(SystemExit, match="no reference mix"):
        recalibrate.main(["check", "--answers", str(answers), "--calibration", str(old)])
