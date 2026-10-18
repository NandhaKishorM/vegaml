"""attach(): a user fits a port for their own reader, with no training data and no retraining.

The heavy ends (two frozen readers) are exercised by the real notebooks; these tests pin the
math and the contracts: the ridge recovers a linear map, fit_port lands features on their
shared-space targets well enough that a port round-trips them, the probe set is made of valid
questions, bare reader names resolve to Hub ids, and calibrate() writes a block the loader
reads back.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

from vegaml import label_ids
from vegaml._api import Vega as ApiVega
from vegaml._attach import PROBES, _abstain_threshold, _probe_questions, calibrate, fit_port, ridge_fit
from vegaml.hub import backbone_hub_id


def test_probes_are_valid_questions():
    assert len(PROBES) >= 5
    types = set()
    for q in PROBES.values():
        ids = label_ids(q)
        assert len(ids) >= 2
        types.add(q["type"])
    assert types == {"choice", "boolean", "score"}   # the port must see every answer type
    # and they survive the same normalisation decide() applies
    assert len(_probe_questions()) == len(PROBES)


def test_ridge_recovers_a_linear_map():
    torch.manual_seed(0)
    W = torch.randn(12, 7)
    b = torch.randn(7)
    X = torch.randn(4096, 12)
    Y = X @ W + b + 0.01 * torch.randn(4096, 7)
    Wf, bf, cos, _ = ridge_fit(X, Y)
    assert cos > 0.99
    assert torch.allclose(Wf, W, atol=0.05)
    assert torch.allclose(bf, b, atol=0.05)


def test_fit_port_lands_on_the_shared_space():
    """Features that ARE a linear image of shared-space targets must port back onto them."""
    torch.manual_seed(1)
    N, M, Lf, d_in, d_sh = 1500, 4200, 2, 20, 12
    y_items = torch.randn(N, Lf, 3, d_sh)
    y_opts = torch.randn(M, d_sh)
    A = [torch.randn(d_sh, d_in) for _ in range(Lf)]        # reader = targets through A
    x_items = torch.stack([y_items[:, li] @ A[li] for li in range(Lf)], 1)
    x_items += 0.01 * torch.randn_like(x_items)
    x_opts = y_opts @ A[-1] + 0.01 * torch.randn(M, d_in)
    port, cosines = fit_port(x_items, x_opts, y_items, y_opts, d_sh)
    assert all(c > 0.98 for c in cosines), cosines
    with torch.no_grad():
        back = port.items(x_items)
        assert float(torch.nn.functional.cosine_similarity(
            back.reshape(-1, d_sh), y_items.reshape(-1, d_sh), dim=-1).mean()) > 0.98
        back_o = port.opts(x_opts[:, None, :])[:, -1]
        assert float(torch.nn.functional.cosine_similarity(back_o, y_opts, dim=-1).mean()) > 0.98


def test_bare_reader_names_resolve_to_hub_ids(tmp_path):
    assert backbone_hub_id("Qwen3.5-0.8B") == "Qwen/Qwen3.5-0.8B"
    assert backbone_hub_id("gemma-3-1b-it") == "google/gemma-3-1b-it"
    assert backbone_hub_id("someone/their-model") == "someone/their-model"   # already an id
    assert backbone_hub_id(str(tmp_path)) == str(tmp_path)                   # a local path
    assert backbone_hub_id("not-a-known-reader") == "not-a-known-reader"


def test_abstain_threshold_semantics():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    assert _abstain_threshold(conf, np.array([1.0, 1, 1, 1]), 0.1) == pytest.approx(0.6)
    assert _abstain_threshold(conf, np.array([0.0, 0, 0, 0]), 0.1) == 1.0
    # one mistake at the bottom: answering above it keeps the error inside the budget
    t = _abstain_threshold(conf, np.array([1.0, 1, 1, 0]), 0.2)
    assert t == pytest.approx(0.7)


class _StubEngine:
    max_steps = 8


class _StubVega:
    """Just enough of _api.Vega for calibrate(): a _view whose logits carry real signal."""

    _internal = staticmethod(ApiVega._internal)

    def __init__(self, model_dir):
        self.model_dir = str(model_dir)
        self.engine = _StubEngine()
        self.rng = np.random.default_rng(0)
        self.theta = np.zeros((3, 5))
        self.conformal, self.abstain = {}, {}
        self.risk, self.unbound = "0.1", float("inf")

    def _view(self, text, qs, template, orders):
        R, K = len(qs), max(len(o) for o in orders)
        logits = self.rng.normal(0.0, 1.0, (R, K))
        logits[:, int(text) % K] += 4.0            # the state encodes the right answer
        out = {"kinetic": torch.rand(R), "nearest": torch.rand(R),
               "steps_used": torch.randint(1, 8, (R,))}
        return {"logits": logits, "mask": torch.ones(R, K, dtype=torch.bool), "out": out}


def test_calibrate_writes_a_block_the_loader_reads(tmp_path):
    with open(os.path.join(tmp_path, "vega_config.json"), "w") as f:
        json.dump({"d_model": 8}, f)
    v = _StubVega(tmp_path)
    questions = {"team": {"type": "choice", "instructions": "which team",
                          "criteria": {"a": None, "b": None, "c": None}}}
    examples = [(str(i), {"team": ["a", "b", "c"][i % 3]}) for i in range(150)]
    report = calibrate(v, examples, questions)

    cfg = json.load(open(os.path.join(tmp_path, "vega_config.json")))
    cal = cfg["calibration"]
    assert cal["fitted_by"] == "vegaml.calibrate"
    assert np.asarray(cal["theta"]).shape == (3, 5)
    assert "choice" in cal["conformal"] and "0.1" in cal["conformal"]["choice"]
    # a tighter risk budget can only RAISE the confidence needed to answer
    ab = cal["abstain_threshold"]
    assert ab["0.05"] >= ab["0.1"] >= ab["0.2"]
    assert report["accuracy"] > 0.9                        # the signal was there to find
    assert report["per_type"] == {"choice": 150}
    # and the live object was updated in place, not just the file
    assert v.theta.shape == (3, 5)
    assert "choice" in v.conformal


def test_calibrate_never_writes_through_an_hf_cache_symlink(tmp_path):
    # A Hub-loaded model dir is an HF cache snapshot: vega_config.json is a symlink into the
    # content-addressed blob store. The refit must land in a real file at the snapshot path
    # and leave the blob byte-identical, or the cache's hashes stop being true.
    blob = tmp_path / "blobs" / "c92eb46450a8206b25d9765933b810b8acb05ed9"
    os.makedirs(blob.parent)
    original = json.dumps({"d_model": 8})
    blob.write_text(original)
    snap = tmp_path / "snapshot"
    os.makedirs(snap)
    os.symlink(os.path.join("..", "blobs", blob.name), snap / "vega_config.json")

    v = _StubVega(snap)
    questions = {"team": {"type": "choice", "instructions": "which team",
                          "criteria": {"a": None, "b": None, "c": None}}}
    calibrate(v, [(str(i), {"team": ["a", "b", "c"][i % 3]}) for i in range(150)], questions)

    assert blob.read_text() == original                    # the blob kept its promised bytes
    assert not os.path.islink(snap / "vega_config.json")   # the snapshot got a real file
    assert json.load(open(snap / "vega_config.json"))["calibration"]["fitted_by"] == "vegaml.calibrate"


def test_calibrate_refuses_labels_outside_the_options(tmp_path):
    with open(os.path.join(tmp_path, "vega_config.json"), "w") as f:
        json.dump({}, f)
    v = _StubVega(tmp_path)
    questions = {"q": {"type": "boolean", "instructions": "is it"}}
    with pytest.raises(ValueError, match="not one of its options"):
        calibrate(v, [(str(i), {"q": "maybe"}) for i in range(40)], questions)


def test_attach_refuses_a_corpus_too_small():
    from vegaml._attach import attach

    with pytest.raises(ValueError, match="at least 100"):
        attach("some/reader", ["one text"], "out")


def test_attach_engine_copy_survives_a_readonly_previous_run(tmp_path):
    # The anchor's engine.safetensors comes from the HF cache, where blobs are read-only, and
    # the first attach carried that mode into out_dir. A second attach into the same dir must
    # still land, and leave a writable file behind for the run after it.
    from vegaml._attach import _copy_fresh

    src = tmp_path / "cache" / "engine.safetensors"
    src.parent.mkdir()
    src.write_bytes(b"engine-bytes")
    src.chmod(0o444)                                  # the HF cache's mode
    dst = tmp_path / "out" / "engine.safetensors"
    dst.parent.mkdir()

    _copy_fresh(str(src), str(dst))
    assert dst.read_bytes() == b"engine-bytes"
    assert dst.stat().st_mode & 0o200                 # owner-writable, unlike the cache

    _copy_fresh(str(src), str(dst))                   # the second attach: used to PermissionError
    assert dst.read_bytes() == b"engine-bytes"
