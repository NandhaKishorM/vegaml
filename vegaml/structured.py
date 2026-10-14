"""Typed decisions into typed Python: a pydantic model is the question sheet and the answer.

    pip install "vegaml[structured]"

    from typing import Literal
    from pydantic import BaseModel, Field
    from vegaml.structured import decide_model

    class Ticket(BaseModel):
        team: Literal["billing", "technical", "sales"] = Field(
            description="Which team handles this?")
        churn: bool = Field(description="Does the customer threaten to leave?")
        severity: int = Field(ge=0, le=3, description="How severe is the problem?")

    ticket, result = decide_model(v, "I was billed twice, refund me or I cancel.", Ticket)
    ticket.team                       # "billing", validated by the model
    result["answers"]["team"]         # the calibrated detail is still all there

Each field is one question. A bool is a boolean question, a Literal (or str Enum) a choice
whose options are the allowed values, and a bounded int (ge/le) a score whose levels span the
range; the field description is the instruction the model reads, so it has to say something.
Level criteria for a score come from Field(json_schema_extra={"criteria": [...]}) when the
range alone does not describe them.

The result is the decide() dictionary unchanged, next to a validated instance, so nothing
calibrated is lost by going typed: probabilities, prediction sets and abstain flags sit one
key away.
"""
from __future__ import annotations

import enum
from typing import Dict, Literal, Tuple, Type, get_args, get_origin

from pydantic import BaseModel

try:                                           # annotated_types ships with pydantic v2
    from annotated_types import Ge, Le
except ImportError:                            # pragma: no cover
    Ge = Le = None


def _bounds(info):
    lo = hi = None
    for m in info.metadata:
        if Ge is not None and isinstance(m, Ge):
            lo = int(m.ge)
        if Le is not None and isinstance(m, Le):
            hi = int(m.le)
    return lo, hi


def questions_from_model(model: Type[BaseModel]) -> Dict[str, Dict]:
    """One question per field. The field's description is the instruction, and is required."""
    questions: Dict[str, Dict] = {}
    for name, info in model.model_fields.items():
        if not info.description:
            raise ValueError("field %r has no description; the description is the instruction "
                             "the model reads, so it cannot be empty" % name)
        ann = info.annotation
        q: Dict = {"instructions": info.description}
        extra = info.json_schema_extra if isinstance(info.json_schema_extra, dict) else {}
        if ann is bool:
            q["type"] = "boolean"
        elif get_origin(ann) is Literal:
            values = get_args(ann)
            if not all(isinstance(v, str) for v in values):
                raise TypeError("field %r: Literal options must all be strings, got %r" % (name, values))
            q["type"] = "choice"
            q["criteria"] = list(values)
        elif isinstance(ann, type) and issubclass(ann, enum.Enum):
            values = [m.value for m in ann]
            if not all(isinstance(v, str) for v in values):
                raise TypeError("field %r: Enum values must all be strings, got %r" % (name, values))
            q["type"] = "choice"
            q["criteria"] = values
        elif ann is int:
            lo, hi = _bounds(info)
            if lo is None or hi is None or hi <= lo:
                raise TypeError("field %r: an int question needs ge and le bounds (at least two "
                                "levels), e.g. Field(ge=0, le=3, description=...)" % name)
            q["type"] = "score"
            crit = extra.get("criteria")
            if crit is not None:
                if len(crit) != hi - lo + 1:
                    raise ValueError("field %r: %d criteria for %d levels (%d..%d)"
                                     % (name, len(crit), hi - lo + 1, lo, hi))
                q["criteria"] = [str(c) for c in crit]
            else:
                q["criteria"] = [str(v) for v in range(lo, hi + 1)]
        else:
            raise TypeError("field %r: %r is not a decidable type. Use bool, Literal[str, ...], "
                            "a str Enum, or a bounded int" % (name, ann))
        questions[name] = q
    return questions


def parse_result(model: Type[BaseModel], result: Dict) -> BaseModel:
    """The decide() result into a validated instance of the model that asked the questions."""
    values = {}
    for name, info in model.model_fields.items():
        a = result["answers"][name]
        ann = info.annotation
        if ann is bool:
            values[name] = bool(a["boolean"] >= 0.5)
        elif get_origin(ann) is Literal:
            values[name] = a["choice"]
        elif isinstance(ann, type) and issubclass(ann, enum.Enum):
            values[name] = ann(a["choice"])
        else:                                  # bounded int: score levels count from ge
            lo, _ = _bounds(info)
            values[name] = lo + int(a["score"])
    return model(**values)


def decide_model(v, state, model: Type[BaseModel], mode=None, alpha: str = "0.1",
                 views: int = 1) -> Tuple[BaseModel, Dict]:
    """decide() with the model's fields as the questions. Returns (instance, full result)."""
    result = v.decide(state, questions_from_model(model), mode=mode, alpha=alpha, views=views)
    return parse_result(model, result), result
