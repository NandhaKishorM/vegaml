"""Pydantic models as question sheets: field mapping, parsing back, and the refusals."""
from __future__ import annotations

import enum
from typing import Literal

import pytest

pytest.importorskip("pydantic", reason="pip install 'vegaml[structured]'")

from pydantic import BaseModel, Field, ValidationError  # noqa: E402

from vegaml.structured import decide_model, parse_result, questions_from_model  # noqa: E402


class Ticket(BaseModel):
    team: Literal["billing", "technical", "sales"] = Field(description="Which team handles this?")
    churn: bool = Field(description="Does the customer threaten to leave?")
    severity: int = Field(ge=1, le=3, description="How severe?",
                          json_schema_extra={"criteria": ["cosmetic", "degraded", "outage"]})


def test_fields_become_their_question_types():
    qs = questions_from_model(Ticket)
    assert qs["team"] == {"type": "choice", "instructions": "Which team handles this?",
                          "criteria": ["billing", "technical", "sales"]}
    assert qs["churn"] == {"type": "boolean", "instructions": "Does the customer threaten to leave?"}
    assert qs["severity"] == {"type": "score", "instructions": "How severe?",
                              "criteria": ["cosmetic", "degraded", "outage"]}


def test_bounded_int_without_criteria_numbers_its_levels():
    class M(BaseModel):
        stars: int = Field(ge=0, le=4, description="How positive?")
    assert questions_from_model(M)["stars"]["criteria"] == ["0", "1", "2", "3", "4"]


def test_str_enum_fields_are_choices_and_parse_back_to_the_enum():
    class Team(enum.Enum):
        BILLING = "billing"
        SALES = "sales"

    class M(BaseModel):
        team: Team = Field(description="Which team?")
    assert questions_from_model(M)["team"]["criteria"] == ["billing", "sales"]
    inst = parse_result(M, {"answers": {"team": {"choice": "sales"}}})
    assert inst.team is Team.SALES


def test_parse_result_builds_a_validated_instance():
    result = {"answers": {"team": {"choice": "billing", "confidence": 0.93},
                          "churn": {"boolean": 0.81},
                          "severity": {"score": 2, "expected_score": 1.7}}}
    t = parse_result(Ticket, result)
    assert t.team == "billing"
    assert t.churn is True
    assert t.severity == 3                     # level 2 counted from ge=1


def test_parse_result_refuses_an_out_of_range_answer():
    bad = {"answers": {"team": {"choice": "legal"}, "churn": {"boolean": 0.2},
                       "severity": {"score": 0}}}
    with pytest.raises(ValidationError):
        parse_result(Ticket, bad)


def test_decide_model_round_trip_keeps_the_calibrated_detail():
    class FakeVega:
        def decide(self, state, questions, mode=None, alpha="0.1", views=1):
            self.got = (state, questions, mode, alpha, views)
            return {"answers": {"team": {"choice": "technical", "confidence": 0.8,
                                         "prediction_set": ["technical", "billing"]},
                                "churn": {"boolean": 0.2},
                                "severity": {"score": 0}},
                    "latency_ms": 1.0, "n_tokens": 5, "views": views}

    v = FakeVega()
    inst, result = decide_model(v, "the state", Ticket, mode="engine", alpha="0.05", views=2)
    assert v.got == ("the state", questions_from_model(Ticket), "engine", "0.05", 2)
    assert (inst.team, inst.churn, inst.severity) == ("technical", False, 1)
    assert result["answers"]["team"]["prediction_set"] == ["technical", "billing"]


def test_missing_description_is_refused_naming_the_field():
    class M(BaseModel):
        churn: bool
    with pytest.raises(ValueError, match="churn"):
        questions_from_model(M)


def test_unbounded_int_is_refused():
    class M(BaseModel):
        stars: int = Field(description="How positive?")
    with pytest.raises(TypeError, match="ge and le"):
        questions_from_model(M)


def test_wrong_criteria_count_is_refused():
    class M(BaseModel):
        stars: int = Field(ge=0, le=2, description="x", json_schema_extra={"criteria": ["a", "b"]})
    with pytest.raises(ValueError, match="2 criteria for 3 levels"):
        questions_from_model(M)


def test_undecidable_types_are_refused():
    class M(BaseModel):
        notes: float = Field(description="free text?")
    with pytest.raises(TypeError, match="not a decidable type"):
        questions_from_model(M)


def test_non_string_literals_are_refused():
    class M(BaseModel):
        n: Literal[1, 2] = Field(description="which?")
    with pytest.raises(TypeError, match="strings"):
        questions_from_model(M)


def test_the_extra_is_declared():
    import pathlib
    import sys
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib
    root = pathlib.Path(__file__).resolve().parent.parent
    py = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert any(d.startswith("pydantic") for d in py["project"]["optional-dependencies"]["structured"])
