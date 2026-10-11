"""The public surface and the mode switch, without loading a checkpoint."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import vegaml
from vegaml import MODES, TTTHead, Vega


class _FakeEngine:
    """Stands in for a loaded checkpoint: enough surface for the switch, none of the weights."""

    cfg = {"max_len": 73728}
    model_dir = "/nowhere"

    def __init__(self):
        self.calls = []

    def decide(self, state, questions, alpha="0.1", views=1):
        self.calls.append((state, tuple(questions), alpha, views))
        return {"answers": {q: {"probs": {"a": 1.0}, "choice": "a"} for q in questions}}


QS = {"q": {"type": "choice", "instructions": "which?", "criteria": {"a": "", "b": ""}}}


def _vega_with_head(mode="engine"):
    v = Vega(_FakeEngine(), mode=mode)
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(size=(20, 16)) + 4, rng.normal(size=(20, 16)) - 4])
    y = ["a"] * 20 + ["b"] * 20
    v.heads["q"] = (TTTHead(["a", "b"]).fit(X, y), 0.95)
    v.features = lambda state, q: X[0]      # the frozen read, stubbed
    return v


def test_auto_is_the_default_mode():
    assert Vega(_FakeEngine()).mode == "auto"


def test_auto_with_nothing_fitted_is_the_engine_flags_and_all():
    """The reason auto is safe as the default: fit nothing and it is the engine's own answer."""
    engine = _FakeEngine()
    out = Vega(engine).decide("s", QS)
    assert out["mode"] == "auto" and "ttt" not in out
    assert out["answers"]["q"]["readout"] == "engine"
    assert engine.calls, "the engine was asked"


def test_auto_prefers_a_head_that_is_known_to_beat_chance():
    v = _vega_with_head("auto")             # cv 0.95 over two labels, chance 0.5
    out = v.decide("s", QS)
    assert out["answers"]["q"]["readout"] == "ttt"
    assert not v.engine.calls, "no question was left for the engine"


def test_auto_leaves_a_head_at_or_below_chance_to_the_engine():
    v = _vega_with_head("auto")
    head, _ = v.heads["q"]
    v.heads["q"] = (head, 0.5)              # exactly chance over two labels
    out = v.decide("s", QS)
    assert out["answers"]["q"]["readout"] == "engine"


def test_auto_leaves_an_unmeasured_head_to_the_engine():
    """cv None is "unknown", and auto does not trust an unmeasured head without being asked."""
    v = _vega_with_head("auto")
    head, _ = v.heads["q"]
    v.heads["q"] = (head, None)
    out = v.decide("s", QS)
    assert out["answers"]["q"]["readout"] == "engine"


def test_auto_can_split_one_request_between_the_readouts():
    v = _vega_with_head("auto")
    questions = dict(QS, other={"type": "choice", "instructions": "which?",
                                "criteria": {"a": "", "c": ""}})
    out = v.decide("s", questions)
    assert [k for k in out["answers"]] == ["q", "other"], "asked order is kept"
    assert out["answers"]["q"]["readout"] == "ttt"
    assert out["answers"]["other"]["readout"] == "engine"
    assert v.engine.calls[-1][1] == ("other",), "only the unfitted question reached the engine"


def test_unknown_mode_is_rejected_at_construction_and_at_call():
    with pytest.raises(ValueError):
        Vega(_FakeEngine(), mode="magic")
    with pytest.raises(ValueError):
        Vega(_FakeEngine()).decide("s", QS, mode="magic")


def test_ttt_mode_without_a_fitted_head_says_so_instead_of_guessing():
    with pytest.raises(RuntimeError, match="fit"):
        Vega(_FakeEngine(), mode="ttt").decide("s", QS)


def test_ttt_mode_answers_from_the_head():
    out = _vega_with_head("ttt").decide("s", QS)
    assert out["mode"] == "ttt"
    ans = out["answers"]["q"]
    assert ans["choice"] == "a" and 0.0 <= ans["confidence"] <= 1.0
    assert ans["cv_accuracy"] == 0.95


def test_both_mode_returns_the_two_readouts_side_by_side():
    out = _vega_with_head("both").decide("s", QS)
    assert out["mode"] == "both"
    assert "answers" in out and "ttt" in out
    assert out["ttt"]["q"]["choice"] == "a"


def test_both_mode_skips_questions_with_no_head_rather_than_failing():
    v = _vega_with_head("both")
    qs = dict(QS, other={"type": "boolean", "instructions": "?", "criteria": {"true": "", "false": ""}})
    out = v.decide("s", qs)
    assert set(out["ttt"]) == {"q"}


def test_fit_reports_examples_cv_and_the_chance_line(monkeypatch):
    v = Vega(_FakeEngine())
    rng = np.random.default_rng(2)
    feats = {}

    def fake_features(state, q):
        feats.setdefault(state, rng.normal(size=32) + (8.0 if state.startswith("a") else -8.0))
        return feats[state]

    v.features = fake_features
    examples = [("a%d" % i, {"q": "a"}) for i in range(10)] + \
               [("b%d" % i, {"q": "b"}) for i in range(10)]
    report = v.fit(examples, QS)
    assert report["q"]["fitted"] is True
    assert report["q"]["examples"] == 20
    assert report["q"]["chance"] == 0.5
    assert report["q"]["at_or_below_chance"] is False
    assert "q" in v.heads


def test_unmeasurable_cv_is_reported_as_unknown_not_as_a_pass():
    v = Vega(_FakeEngine())
    rng = np.random.default_rng(3)
    v.features = lambda state, q: rng.normal(size=16)
    examples = [("s%d" % i, {"q": ["a", "b"][i % 2]}) for i in range(8)]   # 8 < 2 x 5 folds
    report = v.fit(examples, QS)
    assert report["q"]["fitted"] is True
    assert report["q"]["cv_accuracy"] is None
    assert report["q"]["at_or_below_chance"] is None, "unknown must not be reported as False"
    assert "cv_note" in report["q"]


def test_fit_refuses_to_fit_on_too_few_examples():
    v = Vega(_FakeEngine())
    v.features = lambda state, q: np.zeros(8)
    report = v.fit([("s", {"q": "a"})], QS)
    assert report["q"]["fitted"] is False and "at least" in report["q"]["reason"]


def test_load_defaults_to_the_small_model(monkeypatch):
    """No argument means 0.8B. That is the published default and must not drift."""
    from vegaml import hub

    seen = {}

    def fake_snapshot_download(repo_id, **kw):
        seen.update(kw, repo_id=repo_id)
        return "/tmp/snapshot"

    stub = types.ModuleType("huggingface_hub")
    stub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", stub)

    assert hub.fetch() == "/tmp/snapshot"
    assert seen["repo_id"] == hub.DEFAULT_REPO
    assert seen["allow_patterns"] == hub.ALLOW_PATTERNS, "the default must read the repository root"


def test_the_4b_reads_from_its_own_folder(monkeypatch):
    import os

    from vegaml import hub

    seen = {}

    def fake_snapshot_download(repo_id, **kw):
        seen.update(kw, repo_id=repo_id)
        return "/tmp/snapshot"

    stub = types.ModuleType("huggingface_hub")
    stub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", stub)

    out = hub.fetch("4b")
    assert seen["repo_id"] == hub.DEFAULT_REPO, "both sizes live in one repository"
    # os.path.join, not "/": the separator is the platform's, so a test that hardcodes "/" is a test
    # that can only pass on POSIX. This one failed on every Windows checkout.
    assert out == os.path.join("/tmp/snapshot", "4b")
    assert all(p.startswith("4b/") for p in seen["allow_patterns"]
               if p != hub.COUNTER_FILE), "only the counter file sits outside the folder"


def test_a_repo_id_or_path_still_works(monkeypatch, tmp_path):
    from vegaml import hub

    seen = {}
    stub = types.ModuleType("huggingface_hub")
    stub.snapshot_download = lambda repo_id, **kw: (seen.update(kw, repo_id=repo_id), "/tmp/s")[1]
    monkeypatch.setitem(sys.modules, "huggingface_hub", stub)

    hub.fetch("someone/their-model")
    assert seen["repo_id"] == "someone/their-model"
    assert seen["allow_patterns"] == hub.ALLOW_PATTERNS
    assert hub.fetch(str(tmp_path)) == str(tmp_path)


def test_modes_are_the_documented_four():
    assert MODES == ("auto", "engine", "ttt", "both")


def test_public_names_are_exported():
    for name in ("load", "Vega", "TTTHead", "fetch", "__version__"):
        assert hasattr(vegaml, name)
    # label_ids and MODES are imported by name from outside the package, so `from vegaml import *`
    # has to carry them too.
    for name in ("label_ids", "MODES"):
        assert name in vegaml.__all__, "%s is part of the surface but is not exported" % name


def test_every_name_in_all_actually_exists():
    """__all__ listing a name that is not defined turns `from vegaml import *` into an ImportError
    in the user's code, not in ours."""
    for name in vegaml.__all__:
        assert hasattr(vegaml, name), "__all__ lists %r, which the package does not define" % name


def test_a_star_import_supplies_the_names_other_modules_rely_on():
    """The actual failure mode: `from vegaml import *` followed by using label_ids."""
    namespace = {}
    exec("from vegaml import *", namespace)      # noqa: S102 - exercising the public API is the point
    for name in ("load", "Vega", "TTTHead", "fetch", "label_ids", "MODES"):
        assert name in namespace, "%r is missing from a star import" % name
    assert namespace["label_ids"]({"type": "boolean", "criteria": {}}) == ["false", "true"]
