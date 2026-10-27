"""Device policy without weights, plus synthetic parity checks on an actual MPS host."""
from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest
import torch

from vegaml._common import Perception, VegaEngine, backbone_dtype, resolve_device


@pytest.mark.parametrize("cuda,mps,expected", [
    (True, True, "cuda"), (True, False, "cuda"),
    (False, True, "mps"), (False, False, "cpu"),
])
def test_automatic_device_priority(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert resolve_device().type == expected


@pytest.mark.parametrize("requested", ["cpu", "cuda:1", "mps", torch.device("cpu")])
def test_explicit_device_is_preserved(monkeypatch, requested):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device(requested) == torch.device(requested)


def test_explicit_unavailable_mps_is_not_silently_changed(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS is not available"):
        resolve_device("mps")


@pytest.mark.parametrize("name", ["cpu", "mps"])
def test_cpu_and_mps_backbones_use_float32_without_querying_cuda(monkeypatch, name):
    def unexpected(*args):
        pytest.fail("non-CUDA dtype policy must not query CUDA hardware")
    monkeypatch.setattr(torch.cuda, "get_device_capability", unexpected)
    assert backbone_dtype(torch.device(name)) == torch.float32


@pytest.mark.parametrize("major,expected", [(7, torch.float16), (8, torch.bfloat16), (9, torch.bfloat16)])
def test_cuda_precision_policy_is_unchanged(monkeypatch, major, expected):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (major, 0))
    assert backbone_dtype(torch.device("cuda")) == expected


def test_api_places_perception_and_engine_on_selected_device(monkeypatch, tmp_path):
    from safetensors import torch as st

    from vegaml import _api

    cfg = {"backbone_name": "test/backbone", "feature_layers": [0], "option_layers": [0], "d_model": 4}
    (tmp_path / "vega_config.json").write_text(json.dumps(cfg))
    seen = {}

    class FakePerception:
        def __init__(self, path, device, feature_layers, option_layers, quantize=None):
            seen["perception"] = device
            self.feature_layers = feature_layers

    class FakeEngine:
        def __init__(self, *args):
            pass

        def load_state_dict(self, state):
            pass

        def to(self, device):
            seen["engine"] = device
            return self

        def eval(self):
            return self

        def set_adapters(self, mode):
            pass

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(_api, "Perception", FakePerception)
    monkeypatch.setattr(_api, "VegaEngine", FakeEngine)
    monkeypatch.setattr(st, "load_file", lambda path: {})
    v = _api.Vega(str(tmp_path), adapters="off")
    assert v.device == seen["perception"] == seen["engine"] == torch.device("mps")


def test_perception_loads_float32_then_moves_to_mps(monkeypatch):
    seen = {}

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Identity()])
            self.config = types.SimpleNamespace(hidden_size=4)

        def to(self, device):
            seen["device"] = device
            return self

    def load_model(path, **kw):
        seen.update(kw)
        return FakeModel()

    stub = types.ModuleType("transformers")
    stub.__version__ = "5.17.0"
    stub.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda path: object())
    stub.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=load_model)
    monkeypatch.setitem(sys.modules, "transformers", stub)
    Perception("test/backbone", torch.device("mps"), [0], [0])
    assert seen["dtype"] == torch.float32
    assert seen["device"] == torch.device("mps")
    assert seen["attn_implementation"] == "sdpa"
    assert "device_map" not in seen, "single-device loading uses .to(), not Accelerate dispatch"


DEVICES = ["cpu", pytest.param("mps", marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires actual Apple MPS hardware"))]


@pytest.mark.parametrize("device", DEVICES)
def test_device_pooling_matches_cpu_with_padding_and_chunks(device):
    """Exercise real bincount/index_add kernels without downloading a backbone."""
    hidden = torch.arange(2 * 8 * 4, dtype=torch.float32).reshape(2, 8, 4) / 10
    items = [{"ids": np.arange(n, dtype=np.int32),
              "spans": {"state": (0, 4), "question": (4, 5), "options": [(5, 6), (6, n)]}}
             for n in (8, 7)]

    def run(device):
        p = Perception.__new__(Perception)
        p.device = torch.device(device)
        p.tok = types.SimpleNamespace(pad_token_id=0)
        p.feature_layers = p.option_layers = p.capture = [0]
        p._run = lambda ids, mask: {0: hidden.to(device)}
        return p._pooled_batch(items, chunks=2, chunk_min_tokens=2, chunk_tokens=2)

    cpu, result = run("cpu"), run(device)
    for reference, actual in zip((cpu[0], cpu[1], *cpu[2]), (result[0], result[1], *result[2]), strict=True):
        assert actual.device.type == device
        torch.testing.assert_close(actual.cpu(), reference, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
@torch.no_grad()
def test_device_engine_matches_cpu_for_all_question_types(device):
    torch.manual_seed(0)
    cfg = {"latent_dim": 8, "world_dim": 8, "proj_hidden": 16, "n_steps": 6, "dt_max": 0.5,
           "settle_eps": 1e-3, "dropout": 0.0, "feature_dropout": 0.0, "ssm_state": 4}
    engine = VegaEngine(d_in=12, n_feat_layers=2, n_opt_layers=1, cfg=cfg).eval()
    mask = torch.ones(3, 5, dtype=torch.bool)
    mask[0, -1] = False
    batch = (torch.randn(3, 2, 3, 12), torch.randn(3, 5, 1, 12), mask, torch.tensor([0, 1, 2]))
    cpu = engine(*batch, halt=True, return_traj=False)
    result = engine.to(device)(*(t.to(device) for t in batch), halt=True, return_traj=False)
    for key in ("logits", "zT", "kinetic", "grad_norm", "nearest"):
        assert result[key].device.type == device
        torch.testing.assert_close(result[key].cpu(), cpu[key], rtol=1e-3, atol=1e-4)
    assert torch.equal(result["steps_used"].cpu(), cpu["steps_used"])
