"""The eval harness, against a fake engine with known behaviour, and its command."""
from __future__ import annotations

import json

import pytest

from vegaml.evals import evaluate, main

Q = {"churn": {"type": "boolean", "instructions": "Does the customer threaten to leave?"},
     "team": {"type": "choice", "instructions": "Which team?",
              "criteria": {"billing": "payments", "technical": "bugs"}}}


class _Scripted:
    """Predicts whatever the state names, at a fixed confidence, with a one-option set."""

    def __init__(self, confidence=0.9):
        self.confidence = confidence

    def decide(self, state, questions, mode=None, alpha="0.1", views=1):
        answers = {}
        for qid, q in questions.items():
            claim = state["claim"][qid]
            if q["type"] == "boolean":
                probs = {"true": self.confidence, "false": 1 - self.confidence}
                if claim == "false":
                    probs = {"true": 1 - self.confidence, "false": self.confidence}
                answers[qid] = {"probs": probs, "boolean": probs["true"],
                                "confidence": self.confidence, "prediction_set": [claim],
                                "abstain": False}
            else:
                probs = {k: (self.confidence if k == claim else 1 - self.confidence)
                         for k in q["criteria"]}
                answers[qid] = {"probs": probs, "choice": claim,
                                "confidence": self.confidence, "prediction_set": [claim],
                                "abstain": False}
        return {"answers": answers, "latency_ms": 7.0, "n_tokens": 42}


def _rows(n=10, wrong_every=5):
    """Every `wrong_every`-th row's labels disagree with what the model will claim."""
    rows = []
    for i in range(n):
        claim = {"churn": "true", "team": "billing"}
        labels = dict(claim)
        if i % wrong_every == 0:
            labels = {"churn": "false", "team": "technical"}
        rows.append({"state": {"claim": claim}, "labels": labels})
    return rows


# ---------------------------------------------------------------------- the numbers

def test_accuracy_coverage_and_latency_are_what_the_script_dictates():
    report = evaluate(_Scripted(), _rows(10, wrong_every=5), Q)
    for qid in ("churn", "team"):
        s = report["questions"][qid]
        assert s["n"] == 10 and s["accuracy"] == 0.8
        assert s["coverage"] == 0.8 and s["mean_set_size"] == 1.0
        assert s["abstain_rate"] == 0.0
    o = report["overall"]
    assert o["accuracy"] == 0.8 and o["latency_ms_p50"] == 7.0
    assert o["n_tokens_per_decision"] == 42.0 and o["mode"] == "engine"


def test_ece_sees_through_a_flat_confidence():
    # Right 80% of the time at a claimed 0.9: ECE must be about the 0.1 gap.
    report = evaluate(_Scripted(confidence=0.9), _rows(10, wrong_every=5), Q)
    assert abs(report["overall"]["ece"] - 0.1) < 1e-6


def test_per_row_questions_override_the_sheet():
    rows = [{"state": {"claim": {"churn": "true"}}, "labels": {"churn": "true"},
             "questions": {"churn": {"type": "boolean", "instructions": "Rephrased?"}}}] * 3
    seen = []

    class Spy(_Scripted):
        def decide(self, state, questions, **kw):
            seen.append(questions["churn"]["instructions"])
            return super().decide(state, questions, **kw)

    evaluate(Spy(), rows, Q)
    assert seen == ["Rephrased?"] * 3


def test_rows_labelling_no_known_question_are_an_error():
    with pytest.raises(ValueError, match="no row labels"):
        evaluate(_Scripted(), [{"state": "s", "labels": {"other": "x"}}], Q)
    with pytest.raises(ValueError, match="no rows"):
        evaluate(_Scripted(), [], Q)
    with pytest.raises(ValueError, match="row 0"):
        evaluate(_Scripted(), [{"state": "s"}], Q)
    with pytest.raises(ValueError, match="not one of its options"):
        evaluate(_Scripted(), [{"state": {"claim": {"churn": "true"}},
                                "labels": {"churn": "maybe"}}], Q)


# ---------------------------------------------------------------------- the command

def test_the_command_reads_rows_and_prints_the_report(tmp_path, monkeypatch, capsys):
    data = tmp_path / "rows.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in _rows(10)), encoding="utf-8")
    import vegaml
    monkeypatch.setattr(vegaml, "load", lambda name=None, **kw: _Scripted())
    assert main([str(data), "--questions", json.dumps(Q)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["overall"]["accuracy"] == 0.8


def test_the_command_refuses_bad_input_by_name(tmp_path, monkeypatch):
    bad = tmp_path / "rows.jsonl"
    bad.write_text("{not json}", encoding="utf-8")
    with pytest.raises(SystemExit, match="line 1"):
        main([str(bad), "--questions", json.dumps(Q)])
    ok = tmp_path / "ok.jsonl"
    ok.write_text(json.dumps(_rows(1)[0]), encoding="utf-8")
    with pytest.raises(SystemExit, match="not JSON"):
        main([str(ok), "--questions", "{bad"])
    with pytest.raises(SystemExit, match="non-empty object"):
        main([str(ok), "--questions", "{}"])


def test_the_command_names_an_unreadable_file_instead_of_tracebacking(tmp_path, monkeypatch):
    """Every other input error here is a prefixed SystemExit; a bad path must not be a raw
    FileNotFoundError, and must not cost a checkpoint load to report it."""
    loaded = []
    import vegaml
    monkeypatch.setattr(vegaml, "load",
                        lambda name=None, **kw: loaded.append(name) or _Scripted())
    good_rows = tmp_path / "rows.jsonl"
    good_rows.write_text("\n".join(json.dumps(r) for r in _rows(3)), encoding="utf-8")
    good_sheet = tmp_path / "questions.json"
    good_sheet.write_text(json.dumps(Q), encoding="utf-8")

    with pytest.raises(SystemExit, match=r"--questions cannot read 'nope\.json'"):
        main([str(good_rows), "--questions", "@nope.json"])
    with pytest.raises(SystemExit, match=r"cannot read 'nope\.jsonl'"):
        main(["nope.jsonl", "--questions", "@" + str(good_sheet)])
    assert not loaded, "a mistyped path is refused before the model is paid for"


def test_the_command_reads_a_questions_file_it_is_given(tmp_path, monkeypatch, capsys):
    sheet = tmp_path / "questions.json"
    sheet.write_text(json.dumps(Q), encoding="utf-8")
    data = tmp_path / "rows.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in _rows(10)), encoding="utf-8")
    import vegaml
    monkeypatch.setattr(vegaml, "load", lambda name=None, **kw: _Scripted())
    assert main([str(data), "--questions", "@" + str(sheet)]) == 0
    assert json.loads(capsys.readouterr().out)["overall"]["accuracy"] == 0.8


def test_the_engines_boolean_spelling_counts_as_covered():
    class EngineSpelled(_Scripted):
        def decide(self, state, questions, mode=None, alpha="0.1", views=1):
            out = super().decide(state, questions, mode=mode, alpha=alpha, views=views)
            for qid, q in questions.items():
                if q["type"] == "boolean":
                    a = out["answers"][qid]
                    a["probs"] = {"False": a["probs"]["false"], "True": a["probs"]["true"]}
                    a["prediction_set"] = [p == "true" for p in a["prediction_set"]]
            return out

    report = evaluate(EngineSpelled(), _rows(10, wrong_every=5), Q)
    assert report["questions"]["churn"]["coverage"] == 0.8     # was 0.0 before the fix


def test_unbound_is_counted_apart_from_abstain():
    class OffMap(_Scripted):
        def decide(self, state, questions, mode=None, alpha="0.1", views=1):
            out = super().decide(state, questions, mode=mode, alpha=alpha, views=views)
            for qid in questions:
                out["answers"][qid]["unbound"] = True      # off the calibrated map
                out["answers"][qid]["abstain"] = False
            return out

    report = evaluate(OffMap(), _rows(10), Q)
    assert report["questions"]["churn"]["unbound_rate"] == 1.0
    assert report["questions"]["churn"]["abstain_rate"] == 0.0
    assert report["overall"]["unbound_rate"] == 1.0
