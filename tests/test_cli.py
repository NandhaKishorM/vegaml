"""The vegaml command, run against a fake checkpoint: flag parsing, stdin, JSONL, and the errors."""
from __future__ import annotations

import io
import json

import pytest

from vegaml import cli


class _FakeVega:
    def __init__(self):
        self.calls = []

    def decide(self, state, questions, mode=None, alpha="0.1", views=1):
        self.calls.append({"state": state, "questions": questions, "mode": mode,
                           "alpha": alpha, "views": views})
        return {"answers": {q: {"confidence": 0.9} for q in questions},
                "latency_ms": 1.0, "n_tokens": 7, "views": views}


@pytest.fixture
def fake(monkeypatch):
    v = _FakeVega()
    loads = []

    def load(repo, **kw):
        loads.append({"repo": repo, **kw})
        return v

    import vegaml
    monkeypatch.setattr(vegaml, "load", load)
    v.loads = loads
    return v


def _run(argv, fake_stdin=""):
    import sys
    old = sys.stdin
    sys.stdin = io.StringIO(fake_stdin)
    try:
        return cli.main(argv)
    finally:
        sys.stdin = old


def test_typed_flags_become_questions(fake, capsys):
    assert _run(["the state",
                 "--boolean", "churn=Does the customer threaten to leave?",
                 "--choice", "team=Which team?=billing, technical,sales",
                 "--score", "stars=How positive?=bad,ok,good"]) == 0
    call = fake.calls[0]
    assert call["state"] == "the state"
    assert call["questions"]["churn"] == {"type": "boolean",
                                          "instructions": "Does the customer threaten to leave?"}
    assert call["questions"]["team"] == {"type": "choice", "instructions": "Which team?",
                                         "criteria": ["billing", "technical", "sales"]}
    assert call["questions"]["stars"] == {"type": "score", "instructions": "How positive?",
                                          "criteria": ["bad", "ok", "good"]}
    out = json.loads(capsys.readouterr().out)
    assert set(out["answers"]) == {"churn", "team", "stars"}


def test_score_without_levels_is_refused(fake):
    with pytest.raises(SystemExit, match="ordered levels"):
        _run(["s", "--score", "stars=How positive?"])


def test_questions_json_inline_and_from_file(fake, tmp_path):
    q = {"q1": {"type": "boolean", "instructions": "x?"}}
    _run(["s", "--questions", json.dumps(q)])
    assert fake.calls[0]["questions"] == q
    p = tmp_path / "q.json"
    p.write_text(json.dumps(q))
    _run(["s", "--questions", "@%s" % p])
    assert fake.calls[1]["questions"] == q


def test_duplicate_question_id_is_an_error_not_an_override(fake):
    with pytest.raises(SystemExit, match="more than once"):
        _run(["s", "--boolean", "q=a?", "--questions", '{"q": {"type": "boolean", "instructions": "b?"}}'])


def test_state_comes_from_stdin_when_dashed_or_absent(fake):
    _run(["-", "--boolean", "q=x?"], fake_stdin="piped state")
    assert fake.calls[0]["state"] == "piped state"
    _run(["--boolean", "q=x?"], fake_stdin="also piped")
    assert fake.calls[1]["state"] == "also piped"


def test_mode_alpha_views_and_model_pass_through(fake):
    _run(["s", "--boolean", "q=x?", "--mode", "engine", "--alpha", "0.05", "--views", "2",
          "--model", "4b", "--device", "cpu", "--max-len", "4096"])
    assert fake.calls[0]["mode"] == "engine"
    assert fake.calls[0]["alpha"] == "0.05"
    assert fake.calls[0]["views"] == 2
    assert fake.loads[0] == {"repo": "4b", "device": "cpu", "max_len": 4096}


def test_jsonl_batch_loads_once_and_answers_per_line(fake, tmp_path, capsys):
    p = tmp_path / "b.jsonl"
    p.write_text('{"state": "one"}\n\n'
                 '{"state": "two", "questions": {"own": {"type": "boolean", "instructions": "o?"}}}\n')
    assert _run(["--jsonl", str(p), "--boolean", "q=x?"]) == 0
    assert len(fake.loads) == 1
    assert [c["state"] for c in fake.calls] == ["one", "two"]
    assert list(fake.calls[0]["questions"]) == ["q"]
    assert list(fake.calls[1]["questions"]) == ["own"]
    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert len(lines) == 2


def test_jsonl_line_without_questions_anywhere_is_an_error(fake, tmp_path):
    p = tmp_path / "b.jsonl"
    p.write_text('{"state": "one"}\n')
    with pytest.raises(SystemExit, match="no questions"):
        _run(["--jsonl", str(p)])


def test_no_questions_is_a_usage_error(fake):
    with pytest.raises(SystemExit, match="no questions"):
        _run(["s"])


def test_bad_inline_json_names_the_problem(fake):
    with pytest.raises(SystemExit, match="not valid JSON"):
        _run(["s", "--questions", "{nope"])


def test_choice_without_options_is_refused(fake):
    with pytest.raises(SystemExit, match="names no options"):
        _run(["s", "--choice", "team=Which team?"])


def test_score_with_one_level_is_refused(fake):
    with pytest.raises(SystemExit, match="ordered levels"):
        _run(["s", "--score", "stars=How positive?=only"])


# ------------------------------------------------ an `=` inside the instruction, not a separator

def test_boolean_instruction_keeps_its_equals_sign(fake):
    """A boolean question has no option list, so every `=` after the id belongs to the prose.
    The instruction is part of the prompt the model reads, so losing a tail changes the question."""
    _run(["s", "--boolean", "sev=Score this from =0 to 4"])
    assert fake.calls[0]["questions"]["sev"]["instructions"] == "Score this from =0 to 4"


def test_choice_instruction_keeps_its_equals_sign(fake):
    """Options are the text after the LAST `=`, so the instruction may contain one."""
    _run(["s", "--choice", "team=Which team handles =payments? =billing,technical"])
    q = fake.calls[0]["questions"]["team"]
    assert q["instructions"] == "Which team handles =payments? "
    assert q["criteria"] == ["billing", "technical"]


def test_score_instruction_keeps_its_equals_sign(fake):
    _run(["s", "--score", "sev=How severe? (=1 is lowest)=low,mid,high"])
    q = fake.calls[0]["questions"]["sev"]
    assert q["instructions"] == "How severe? (=1 is lowest)"
    assert q["criteria"] == ["low", "mid", "high"]


def test_choice_of_only_empty_options_is_refused(fake):
    """`==` passes the raw-text check but splits to zero options; a choice needs at least one."""
    with pytest.raises(SystemExit, match="names no options"):
        _run(["s", "--choice", "team=Which team?=="])


def test_pretty_prints_indented_json(fake, capsys):
    _run(["s", "--boolean", "q=x?", "--pretty"])
    out = capsys.readouterr().out
    assert out.startswith("{\n")
    assert json.loads(out)["n_tokens"] == 7


def test_entry_point_is_declared():
    import pathlib
    import sys
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib
    root = pathlib.Path(__file__).resolve().parent.parent
    py = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["scripts"]["vegaml"] == "vegaml.cli:main"
