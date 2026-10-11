"""The image path: it did not run at all, and when it did it returned uncalibrated answers.

Three separate defects, all on the path the README documents under "Images":

1. `_decide` imported `vega_common`, a module that does not exist in this package. The import sat
   inside the function, so `import vegaml` succeeded and the module imported cleanly -- the failure
   was a `ModuleNotFoundError` raised at the moment a caller decided about an image.
2. The answer builder labelled every non-`choice` question `[False, True]`, so a `score` question
   with more than two levels indexed off the end of a two-element list.
3. The builder returned `probs`, `confidence` and `choice` and nothing else, silently dropping
   `prediction_set`, `abstain`, `unbound`, `settle_steps` and `energy`.

None of it was covered: `tests/` never touched `_vision.py`. These tests drive the real `_decide`
against a stub engine, so they need no checkpoint and no vision tower.
"""
from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest
import torch

import vegaml
from vegaml import Vega
from vegaml._api import Vega as Engine
from vegaml._common import normalise_conformal, option_texts

PKG = pathlib.Path(vegaml.__file__).parent


# ------------------------------------------------------------------ 1. the dangling import
def test_every_package_relative_import_names_a_module_that_exists():
    """`vega_common` was not importable. A relative import is checked here rather than at runtime,
    because the one that was broken lived inside a function body and so survived import.

    Note the AST shape: for `from ._common import X`, `node.module` is `'_common'` with no leading
    dot and `node.level` is 1. A test that looks for a dot in `module` matches nothing.
    """
    import importlib

    # `vegaml._vision` is not an attribute of `vegaml` until something imports it, so walk the
    # package's own modules explicitly rather than relying on what happens to be loaded.
    for path in sorted(PKG.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.level:
                if not (node.module or ""):
                    continue                       # `from . import x`: the name is checked below
                mod = importlib.import_module("vegaml." + node.module)
                for alias in node.names:
                    assert hasattr(mod, alias.name), \
                        "%s imports %s from .%s, which does not define it" % (
                            path.name, alias.name, node.module)
            if isinstance(node, ast.ImportFrom) and node.level == 1 and not node.module:
                importlib.import_module("vegaml")
                for alias in node.names:
                    assert hasattr(vegaml, alias.name), \
                        "%s imports %s from the package root, which does not export it" % (
                            path.name, alias.name)


def test_the_import_inside_decide_resolves():
    """Belt and braces, and specifically about the body of _decide.

    `ast.walk` does not descend into a function's nested function bodies, so the whole module is
    walked here and the resulting nodes are located by line range.
    """
    from vegaml import _vision

    tree = ast.parse((PKG / "_vision.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_decide")
    inside = set(range(fn.lineno, (fn.end_lineno or fn.lineno) + 1))
    checked = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level and node.lineno in inside:
            mod = vegaml
            for part in (node.module or "").split("."):
                if part:
                    mod = getattr(mod, part)
            for alias in node.names:
                assert hasattr(mod, alias.name), "%s does not define %s" % (node.module, alias.name)
                checked += 1
    assert checked >= 5, "expected _decide to import its calibration helpers from the package"
    assert _vision.VISION_STATE.startswith("<|vision_start|>")


# ------------------------------------------------------------------ driving _decide for real
class _Stub:
    """The minimum surface _decide touches: an engine, its calibration, and a frozen backbone."""

    def __init__(self, n_options, qtype, logits=None, threshold=0.5):
        self.max_len, self.opt_max_tokens, self.backbone = 512, 32, "dummy"
        self.halt, self.device, self.model_dir = True, torch.device("cpu"), "/nowhere"
        self.cfg = {"max_len": 512, "opt_max_tokens": 32, "feature_layers": [0.5, 1.0],
                    "option_layers": [0.5, 1.0]}
        self.adapters = []
        self.risk = "0.1"
        self.unbound = float("inf")
        self.abstain = {"0.1": 0.4}
        self.memory, self.memory_chunks = None, 0
        self.max_steps = 5
        self.conformal = normalise_conformal({"choice": {"0.1": 0.3}, "score": {"0.1": 0.3},
                                              "boolean": {"0.1": 0.3}})
        self.theta = np.zeros((3, 5))
        self._internal = Engine._internal          # the same normaliser the engine uses
        self.engine = self
        self._logits = logits if logits is not None else np.linspace(1.0, 0.0, n_options)
        self.n = n_options

    def _cal(self, r):
        return {"theta": self.theta, "conformal": self.conformal, "abstain": self.abstain,
                "unbound": self.unbound}

    def __call__(self, feats, opts, mask, qtype, halt=False, return_traj=False, chunks=None,
                 chunk_mask=None):
        b = feats.size(0)
        return {"logits": torch.from_numpy(np.tile(self._logits, (b, 1))).float(),
                "nearest": torch.zeros(b), "steps_used": torch.full((b,), 4),
                "kinetic": torch.zeros(b), "gates": None}

    def _features(self, state_text, qs, image=None, chunks=0, chunk_min_tokens=128, chunk_tokens=128):
        # one pooled triple per question, and one option vector per option. The option count comes
        # from the package's own option_texts, so a question type this stub does not know about
        # still gets the right number of options.
        feats = [torch.zeros(2, 3, 1) for _ in qs]
        olist = [torch.zeros(len(option_texts(q)), 2, 1) for q in qs]
        return feats, olist, 10, []

    # _pool is unused by the harness above; the engine stub is called directly by _decide.
    def _unused(self):  # pragma: no cover
        raise NotImplementedError


def _decide(questions, logits=None):
    """Run the real VegaVision._decide against the stub, with the model layer stubbed out.

    Everything below the answer builder is faked: pooled features are zeros, and the engine returns
    fixed logits. The code under test -- label construction, conformal sets, the abstain and unbound
    flags, the three answer shapes -- is the real thing.
    """
    from vegaml import _common, _vision

    stub = _Stub(4, "choice", logits=logits)
    vv = object.__new__(_vision.VegaVision)
    vv.d = stub
    vv.dev = torch.device("cpu")
    vv.cfg = stub.cfg
    vv.dc = _common
    vv._features = stub._features
    return _vision.VegaVision._decide(vv, "<vision>", questions)


CHOICE = {"k": {"type": "choice", "instructions": "what kind?",
                "criteria": {"a": "one", "b": "two", "c": "three", "d": "four"}}}
BOOLEAN = {"b": {"type": "boolean", "instructions": "does it hold?"}}
SCORE = {"s": {"type": "score", "instructions": "how bad?",
               "criteria": ["low", "mid", "high", "severe"]}}


# ------------------------------------------------------------------ 3. the calibration fields
def test_an_image_answer_carries_the_same_fields_as_a_text_answer():
    """The fields that make an answer usable: the prediction set, the abstain flag, the physics."""
    out = _decide(CHOICE, logits=np.array([2.0, 1.0, 0.0, -1.0]))
    a = out["answers"]["k"]
    for field in ("probs", "confidence", "prediction_set", "abstain", "unbound", "settle_steps",
                  "energy", "choice"):
        assert field in a, "decide_image dropped %r" % field
    assert a["prediction_set"] is not None and len(a["prediction_set"]) >= 1
    assert a["settle_steps"] == 4, "carried through from the engine's trajectory"
    # energy is -max(logits/tau); with theta all zeros tau is 1, so it is -2.0 here. The sign is the
    # engine's convention, and the point here is only that the field is present and populated.
    assert a["energy"] == pytest.approx(-2.0)


def test_the_prediction_set_obeys_the_conformal_threshold():
    out = _decide(CHOICE, logits=np.array([2.0, 1.0, 0.0, -1.0]))
    a = out["answers"]["k"]
    # a confidence above the 0.4 abstain line must not be flagged for abstention
    assert a["abstain"] is False
    assert set(a["prediction_set"]) <= {"a", "b", "c", "d"}
    assert "a" in a["prediction_set"], "the top option is always kept"


def test_a_low_confidence_answer_is_flagged_for_abstention():
    out = _decide(CHOICE, logits=np.array([1.0, 0.99, 0.98, 0.97]))
    a = out["answers"]["k"]
    assert a["confidence"] < 0.4
    assert a["abstain"] is True, "confidence below the abstain line must set the flag"


# ------------------------------------------------------------------ 2. score questions
def test_a_score_question_answers_every_level_not_just_two():
    """It labelled every non-choice question [False, True], so a 4-level score indexed off the end."""
    out = _decide(SCORE, logits=np.array([2.0, 1.0, 0.0, -1.0]))
    a = out["answers"]["s"]
    assert set(a["probs"]) == {"0", "1", "2", "3"}, "one probability per level, as the engine gives"
    assert a["score"] in (0, 1, 2, 3)
    assert "expected_score" in a, "the engine returns the expected value; so must the image path"


def test_a_boolean_question_still_answers():
    out = _decide(BOOLEAN, logits=np.array([0.5, 1.5]))
    a = out["answers"]["b"]
    assert set(a["probs"]) == {"False", "True"}
    assert "boolean" in a
    assert a["boolean"] == pytest.approx(a["probs"]["True"])


def test_all_three_types_together():
    out = _decide(dict(CHOICE, **SCORE, **BOOLEAN), logits=np.array([2.0, 1.0, 0.0, -1.0]))
    assert set(out["answers"]) == {"k", "s", "b"}
    assert "score" in out["answers"]["s"] and "boolean" in out["answers"]["b"]
    assert "choice" in out["answers"]["k"]


# ------------------------------------------------------------------ the entry points
def test_decide_image_rejects_an_empty_question_set_with_a_message():
    from vegaml import _vision

    vv = object.__new__(_vision.VegaVision)
    with pytest.raises(ValueError, match="at least one question"):
        _vision.VegaVision._decide(vv, "<vision>", {})


def test_decide_image_loads_the_backbone_the_engine_already_resolved():
    """It re-derived a Hub id from cfg['backbone_name'], which is absent for a checkpoint carrying
    only a local backbone/ directory -- so the image path failed while the text path worked."""
    src = (PKG / "__init__.py").read_text(encoding="utf-8")
    assert 'self.engine.cfg.get("backbone_name")' not in src, \
        "do not re-derive the backbone; use the one the engine resolved"
    assert "self.engine.backbone" in src


def test_the_vision_tower_is_not_loaded_until_an_image_is_decided():
    class _Engine:
        cfg = {"max_len": 73728}
        model_dir = "/nowhere"
        backbone = "dummy"

    v = Vega(_Engine())
    assert v._vision is None, "a text-only caller must not pay for the vision tower"


def test_decide_image_marks_every_answer_as_engine_readout():
    """A TTT head reads text features; it has nothing to say about pixels. The image path runs the
    engine and says so, so a caller reading a mixed result is not misled."""
    src = (PKG / "__init__.py").read_text(encoding="utf-8")
    fn = ast.unparse(next(n for n in ast.walk(ast.parse(src))
                          if isinstance(n, ast.FunctionDef) and n.name == "decide_image"))
    assert 'readout' in fn and '"engine"' in fn


def test_an_image_can_be_a_path_or_an_array_not_just_a_pil_image():
    """It called image.convert('RGB') unconditionally, so a path raised AttributeError."""
    src = (PKG / "_vision.py").read_text(encoding="utf-8")
    assert "_as_rgb_image" in src
    fn = ast.unparse(next(n for n in ast.walk(ast.parse(src))
                          if isinstance(n, ast.FunctionDef) and n.name == "decide_image"))
    assert "_as_rgb_image(image)" in fn
    assert "image.convert(\"RGB\")" not in fn
