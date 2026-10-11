"""The Decision Index adapter: the type mapping, the distribution, and what it refuses.

The kit is not a dependency of this package, so `decision_index.engines` is stubbed here with the
three names the adapter imports. The budgets are checked against a tiny whitespace tokenizer rather
than the real one: what matters is that the adapter refuses instead of letting a prompt be cut.
"""
from __future__ import annotations

import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _stub_kit():
    """The three names the adapter imports, used only when the real kit is not installed."""
    engines = types.ModuleType("decision_index.engines")

    class Unsupported(ValueError):
        pass

    class Engine:
        provenance: dict = {}

        def __init__(self, **options):
            self.options = options

    engines.Unsupported = Unsupported
    engines.Engine = Engine
    package = types.ModuleType("decision_index")
    package.engines = engines
    sys.modules.setdefault("decision_index", package)
    sys.modules.setdefault("decision_index.engines", engines)


@pytest.fixture(scope="module")
def adapter():
    _stub_kit()
    sys.path.insert(0, str(ROOT))
    try:
        from bench import decision_index_engine as module
    finally:
        sys.path.remove(str(ROOT))
    return module


class _Tok:
    """One token per whitespace-separated word, which is all the budget checks need."""

    bos_token_id = None

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(str(text).split())))}


def _engine(adapter, max_len=1000, opt_max=32, query_first=0):
    eng = adapter.VegaEngine.__new__(adapter.VegaEngine)
    eng.tok = _Tok()
    eng.max_len = max_len
    eng.opt_max = opt_max
    eng.query_first = query_first
    return eng


CHOICE = {"type": "choice", "instructions": "Which team?", "criteria": {"a": "first", "b": "second"}}
NOUL = {"type": "noul", "instructions": "Is the summary supported?"}


def test_the_kits_yes_no_type_becomes_the_one_this_package_uses(adapter):
    assert adapter.to_vega("q", NOUL)["type"] == "boolean"
    assert adapter.to_vega("q", CHOICE)["type"] == "choice"


def test_a_type_vega_does_not_have_is_refused_not_guessed(adapter):
    with pytest.raises(adapter.Unsupported):
        adapter.to_vega("q", {"type": "score", "instructions": "how bad?", "criteria": ["a", "b"]})


def test_every_option_gets_a_probability_and_they_sum_to_one(adapter):
    answers = {"q": {"probs": {"a": 0.3, "b": 0.6}, "choice": "b"}}
    out = adapter.to_kit_answers({"q": CHOICE}, answers)["q"]
    assert set(out["probabilities"]) == {"a", "b"}
    assert abs(sum(out["probabilities"].values()) - 1) < 1e-9
    assert out["choice"] == "b"


def test_renormalising_never_reorders_the_options(adapter):
    answers = {"q": {"probs": {"a": 0.6, "b": 0.3}, "choice": "a"}}
    out = adapter.to_kit_answers({"q": CHOICE}, answers)["q"]
    assert out["probabilities"]["a"] > out["probabilities"]["b"]
    assert out["choice"] == "a"


def test_a_yes_no_answer_carries_the_probability_of_yes(adapter):
    answers = {"q": {"probs": {"False": 0.25, "True": 0.75}, "boolean": 0.75}}
    out = adapter.to_kit_answers({"q": NOUL}, answers)["q"]
    assert out == {"type": "noul", "noul": 0.75}


def test_instructions_past_the_budget_are_refused_rather_than_cut(adapter):
    eng = _engine(adapter)
    long_q = dict(CHOICE, instructions="word " * (adapter.INS_MAX_TOKENS + 50))
    with pytest.raises(adapter.Unsupported, match="instructions"):
        eng._fixed_tokens("q", adapter.internal(adapter.to_vega("q", long_q)))


def test_an_option_past_the_per_option_budget_is_refused_rather_than_cut(adapter):
    eng = _engine(adapter, opt_max=8)
    wide = dict(CHOICE, criteria={"a": "short", "b": "word " * 40})
    with pytest.raises(adapter.Unsupported, match="per-option budget"):
        eng._fixed_tokens("q", adapter.internal(adapter.to_vega("q", wide)))


def test_too_many_options_for_the_context_are_refused_rather_than_cut(adapter):
    eng = _engine(adapter, max_len=60)
    many = dict(CHOICE, criteria={"opt%03d" % i: "option %d" % i for i in range(200)})
    with pytest.raises(adapter.Unsupported, match="context window"):
        eng._fixed_tokens("q", adapter.internal(adapter.to_vega("q", many)))


def test_a_state_that_does_not_fit_is_refused_rather_than_cut(adapter):
    eng = _engine(adapter, max_len=100)
    shape = adapter.internal(adapter.to_vega("q", CHOICE))
    fixed = eng._fixed_tokens("q", shape)
    eng._check_state("q", shape, fixed, 10)             # room to spare
    with pytest.raises(adapter.Unsupported, match="context window"):
        eng._check_state("q", shape, fixed, 10_000)


def test_a_state_that_fits_is_accepted(adapter):
    eng = _engine(adapter, max_len=1000)
    shape = adapter.internal(adapter.to_vega("q", CHOICE))
    assert eng._fixed_tokens("q", shape) < 1000


# ----------------------------------------------------------------- the ttt readout
class _Head:
    """A head that always names one label, so routing is what the tests observe."""

    def __init__(self, winner):
        self.winner = winner

    def predict(self, x, allowed=None):
        keys = list(allowed or [self.winner])
        rest = (1.0 - 0.9) / max(1, len(keys) - 1) if len(keys) > 1 else 0.0
        return {k: (0.9 if k == self.winner else rest) for k in keys}


def _ttt_engine(adapter, heads, decide=None):
    eng = _engine(adapter)
    eng.repo, eng.views, eng.alpha, eng.mode = "0.8b", 1, "0.1", "ttt"
    eng.heads = heads
    eng.vega = types.SimpleNamespace(
        features=lambda state, question: [0.0],
        decide=decide or (lambda *a, **k: {"answers": {}, "n_tokens": 0}))
    return eng


def test_the_answer_space_is_the_key_not_the_question_name(adapter):
    """Two rows of one benchmark share a space; a different option set does not."""
    a = adapter.internal(adapter.to_vega("q1", CHOICE))
    b = adapter.internal(adapter.to_vega("q2", dict(CHOICE, instructions="Asked differently?")))
    c = adapter.internal(adapter.to_vega("q1", dict(CHOICE, criteria={"a": "first", "c": "third"})))
    assert adapter.space_key(a) == adapter.space_key(b)
    assert adapter.space_key(a) != adapter.space_key(c)


def test_a_fitted_space_is_answered_by_the_head(adapter):
    space = adapter.space_key(adapter.internal(adapter.to_vega("q", CHOICE)))
    eng = _ttt_engine(adapter, {space: _Head("b")})
    response, raw = adapter.VegaEngine.__call__(eng, "some state", {"q": CHOICE})
    assert raw["readout"] == {"q": "ttt"}
    assert response["answers"]["q"]["choice"] == "b"
    assert abs(sum(response["answers"]["q"]["probabilities"].values()) - 1) < 1e-9


def test_a_space_with_no_head_falls_back_to_the_engine(adapter):
    def decide(state, asked, **kw):
        return {"answers": {k: {"probs": {"a": 0.8, "b": 0.2}, "choice": "a"} for k in asked},
                "n_tokens": 7}

    eng = _ttt_engine(adapter, {}, decide=decide)
    response, raw = adapter.VegaEngine.__call__(eng, "some state", {"q": CHOICE})
    assert raw["readout"] == {"q": "engine"}
    assert response["answers"]["q"]["choice"] == "a"


def test_one_request_can_split_between_the_two_readouts(adapter):
    wide = dict(CHOICE, criteria={"a": "first", "c": "third"})

    def decide(state, asked, **kw):
        assert set(asked) == {"other"}, "only the unfitted question reaches the engine"
        return {"answers": {"other": {"probs": {"a": 0.1, "c": 0.9}, "choice": "c"}}, "n_tokens": 3}

    space = adapter.space_key(adapter.internal(adapter.to_vega("q", CHOICE)))
    eng = _ttt_engine(adapter, {space: _Head("b")}, decide=decide)
    response, raw = adapter.VegaEngine.__call__(eng, "s", {"q": CHOICE, "other": wide})
    assert raw["readout"] == {"q": "ttt", "other": "engine"}
    assert response["answers"]["q"]["choice"] == "b"
    assert response["answers"]["other"]["choice"] == "c"


def test_a_yes_no_head_reports_the_probability_of_yes(adapter):
    space = adapter.space_key(adapter.internal(adapter.to_vega("q", NOUL)))
    eng = _ttt_engine(adapter, {space: _Head("true")})
    response, _ = adapter.VegaEngine.__call__(eng, "s", {"q": NOUL})
    assert response["answers"]["q"]["noul"] == pytest.approx(0.9)


def test_ttt_mode_without_examples_is_refused_rather_than_run_empty(adapter):
    with pytest.raises(ValueError, match="examples"):
        adapter.VegaEngine(mode="ttt")


def test_an_unknown_mode_is_refused(adapter):
    with pytest.raises(ValueError, match="mode must be one of"):
        adapter.VegaEngine(mode="both")


def test_gold_labels_become_the_strings_the_head_is_fitted_on(adapter):
    assert adapter.label_text({"type": "boolean"}, True) == "true"
    assert adapter.label_text({"type": "boolean"}, 0) == "false"
    assert adapter.label_text({"type": "choice"}, "billing") == "billing"


def test_a_string_false_gold_is_not_fitted_as_a_positive(adapter):
    """`bool("false")` is True, so the truthiness test fitted every string-false example as a
    positive one -- inverting the class the head is being asked to predict."""
    assert adapter.label_text({"type": "boolean"}, "false") == "false"
    assert adapter.label_text({"type": "boolean"}, "true") == "true"
    assert adapter.label_text({"type": "boolean"}, "False") == "false"
    assert adapter.label_text({"type": "boolean"}, 1.0) == "true"
    assert adapter.label_text({"type": "boolean"}, 0.0) == "false"


def test_a_choice_question_may_carry_its_options_as_a_list(adapter):
    """The package's own loader normalises a list of bare option names into a mapping. The adapter
    did not, so a valid kit question raised inside the prompt builder instead of working."""
    listed = {"type": "choice", "instructions": "which?", "criteria": ["a", "b"]}
    shape = adapter.internal(adapter.to_vega("q", listed))
    assert shape["crit"] == {"a": None, "b": None}, "normalised the way the package does it"
    from vegaml._common import option_texts
    assert option_texts(shape) == ["a", "b"], "and the normalised shape is usable by the prompt builder"


def test_a_request_answered_by_heads_still_reports_a_latency_and_token_count(adapter):
    """A head still reads the frozen backbone, so it is timed and counted like any other answer.

    Before this, a request with nothing left for the engine reported latency_ms: None, views: None
    and usage.input_tokens == 0 -- not a measurement of anything.
    """
    space = adapter.space_key(adapter.internal(adapter.to_vega("q", CHOICE)))
    eng = _ttt_engine(adapter, {space: _Head("b")})
    response, raw = adapter.VegaEngine.__call__(eng, "some state", {"q": CHOICE})
    assert raw["latency_ms"] is not None and raw["latency_ms"] >= 0
    assert raw["views"] == 1
    assert response["usage"]["input_tokens"] > 0, "the frozen read happened; it must be counted"


def test_a_mixed_request_reports_a_token_count_covering_both_readouts(adapter):
    """Head questions were never attributed, so a mixed request under-reported its own cost."""
    wide = dict(CHOICE, criteria={"a": "first", "c": "third"})

    def decide(state, asked, **kw):
        return {"answers": {"other": {"probs": {"a": 0.1, "c": 0.9}, "choice": "c"}}, "n_tokens": 3}

    space = adapter.space_key(adapter.internal(adapter.to_vega("q", CHOICE)))
    eng = _ttt_engine(adapter, {space: _Head("b")}, decide=decide)
    engine_only = _ttt_engine(adapter, {}, decide=decide)
    mixed, _ = adapter.VegaEngine.__call__(eng, "s", {"q": CHOICE, "other": wide})
    alone, _ = adapter.VegaEngine.__call__(engine_only, "s", {"other": wide})
    assert mixed["usage"]["input_tokens"] > alone["usage"]["input_tokens"], \
        "a head-answered question costs a frozen read and must be counted"


def test_provenance_mode_matches_the_runtime_mode(adapter):
    """Provenance hardcoded mode="engine", so a ttt run's environment.json contradicted the
    runtime() block for the very same run."""
    import inspect

    src = inspect.getsource(adapter.VegaEngine.__init__)
    assert '"mode": self.mode' in src, "provenance must record the mode actually in force"
    assert '"mode": "engine"' not in src


def test_each_example_is_fitted_on_its_own_question_not_the_spaces_first_one(adapter, tmp_path):
    """The bug this guards: BANKING77 carries its utterance in the instructions over an empty
    state, so fitting every example through one shared question gives identical features and a
    head fitted on noise. The features call must see each row's own instructions."""
    rows = [{"state": {}, "expected": {"q": "a" if i % 2 else "b"},
             "questions": {"q": dict(CHOICE, instructions="utterance number %d" % i)}}
            for i in range(adapter.MIN_EXAMPLES + 4)]
    path = tmp_path / "examples.jsonl"
    path.write_text("\n".join(__import__("json").dumps(r) for r in rows), encoding="utf-8")

    seen = []

    class _Head:
        def fit(self, X, y):
            return self

        def cross_val_accuracy(self, X, y, folds=5):
            return 0.9

    eng = adapter.VegaEngine.__new__(adapter.VegaEngine)
    eng.vega = types.SimpleNamespace(
        features=lambda state, question: seen.append(question["instructions"]) or [0.0])
    eng.heads = {}
    monkey = adapter.TTTHead if hasattr(adapter, "TTTHead") else None
    assert monkey is None, "TTTHead is imported inside the method, not at module level"
    import vegaml
    original = vegaml.TTTHead
    vegaml.TTTHead = lambda labels, **kw: _Head()
    try:
        report = adapter.VegaEngine._fit_heads(eng, str(path))
    finally:
        vegaml.TTTHead = original

    assert len(seen) == len(rows)
    assert len(set(seen)) == len(rows), "every example must be read through its own question"
    assert next(iter(report.values()))["fitted"] is True
