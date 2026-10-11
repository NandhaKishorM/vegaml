"""The fine-tuning gate was never trained. This is the test for that.

`stage_train` set `engine.adapter_mode = "on"`, and in that mode `VegaEngine._set_gates` returns a
constant 1 without ever calling `gate_nets`. So the gate loss had no autograd graph, the gate network
stayed at its random initialisation for the whole run, and the script still wrote
`gate_trained: True` into the adapter metadata.

Inference defaults to `adapters="auto"`, which takes the *other* branch -- `sigmoid(gate_net(x))`
on that untrained network -- so every shipped adapter routed on random weights while reporting that
its gate had been trained.

Nothing caught it because no test ran the training path. These tests run the engine's gate in both
modes with a tiny synthetic engine, so the bug is now pinned on the behaviour rather than on the
text of the script.
"""
from __future__ import annotations

import ast
import pathlib

import pytest
import torch
import torch.nn.functional as F

from vegaml._common import ADAPTER_TARGETS, QTYPES, VegaEngine, attach_adapter

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "notebooks" / "vega_finetune.py"

# A small engine is enough: the bug is in how the gate is produced, not in anything sized.
CFG = dict(latent_dim=8, world_dim=8, proj_hidden=16, n_steps=3, dt_max=0.1, settle_eps=1e-3)


def _engine_with_adapter(mode, seed=0):
    """The parameter set-up stage_train performs: an adapter, LoRA and gate trainable."""
    torch.manual_seed(seed)
    engine = VegaEngine(16, 2, 2, CFG)
    j = attach_adapter(engine, rank=4, alpha=4.0)
    engine.adapter_mode = mode
    for p in engine.parameters():
        p.requires_grad_(False)
    for path in ADAPTER_TARGETS:
        mod = engine
        for part in path.split(".")[:-1]:
            mod = getattr(mod, part) if not part.isdigit() else mod[int(part)]
        lin = mod[int(path.split(".")[-1])]
        lin.lora_A[j].requires_grad_(True)
        lin.lora_B[j].requires_grad_(True)
    for p in engine.gate_nets[j].parameters():
        p.requires_grad_(True)
    return engine, j


def _forward(engine, n=4):
    torch.manual_seed(1)
    return engine(torch.randn(n, 2, 3, 16), torch.randn(n, 3, 2, 16),
                  torch.ones(n, 3, dtype=torch.bool),
                  torch.full((n,), QTYPES["choice"]), return_traj=False)


def _gate_loss(gates, target):
    """The BCE term stage_train adds to the task loss."""
    return F.binary_cross_entropy(gates.clamp(1e-6, 1 - 1e-6), torch.full_like(gates, target))


# ------------------------------------------------------------------ the bug, from both sides
def test_mode_on_returns_a_constant_gate_and_never_calls_the_gate_network():
    engine, j = _engine_with_adapter("on")
    out = _forward(engine)
    assert torch.all(out["gates"][:, j] == 1.0), \
        "mode=on hardcodes the gate open, whatever the gate network says"
    out["logits"].sum().backward()
    assert all(p.grad is None for p in engine.gate_nets[j].parameters()), \
        "in mode=on the gate network is never called, so nothing can train it"


def test_the_gate_loss_has_no_graph_in_mode_on():
    """The consequence the training step actually hit.

    In "on" the whole gate term is built from a constant, so it carries no gradient at all and
    `backward()` refuses it. The training step did not fail loudly; it carried on with a gate term
    that contributed nothing.
    """
    engine, j = _engine_with_adapter("on")
    gates = _forward(engine)["gates"][:, j]
    loss = _gate_loss(gates, 1.0) + _gate_loss(gates, 0.0)
    assert not loss.requires_grad, "the gate loss is inert in mode=on"
    with pytest.raises(RuntimeError):
        loss.backward()


def test_the_same_loss_trains_the_gate_in_auto_mode():
    engine, j = _engine_with_adapter("auto")
    gates = _forward(engine)["gates"][:, j]
    assert gates.min() < 0.5 < gates.max(), \
        "a real gate is sigmoid of something; it must not be pinned open or shut"
    (_gate_loss(gates, 1.0) + _gate_loss(gates, 0.0)).backward()
    grads = [p.grad for p in engine.gate_nets[j].parameters()]
    assert any(g is not None and float(g.abs().sum()) > 0 for g in grads), \
        "the gate loss must reach the gate network"


def test_a_training_step_actually_moves_the_gate():
    """The end-to-end claim: train for a few steps and the gate output changes.

    A gradient that exists but does nothing would pass the test above.
    """
    engine, j = _engine_with_adapter("auto")
    opt = torch.optim.AdamW(list(engine.gate_nets[j].parameters()), lr=0.5)
    before = _forward(engine)["gates"][:, j].detach().clone()
    for _ in range(3):
        out = _forward(engine)
        gates = out["gates"][:, j]
        (_gate_loss(gates, 1.0) + _gate_loss(gates, 0.0)).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    after = _forward(engine)["gates"][:, j].detach()
    assert not torch.allclose(before, after), "training the gate must change what it reports"


# ------------------------------------------------------------------ the script agrees
def _stage(name):
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    return ast.unparse(next(n for n in ast.walk(tree)
                            if isinstance(n, ast.FunctionDef) and n.name == name))


def test_training_runs_the_mode_that_actually_trains_the_gate():
    """It ran "on", so the gate it trained and reported on was the constant one."""
    body = _stage("stage_train")
    assert "adapter_mode = 'auto'" in body, \
        'training must run "auto"; "on" short-circuits the gate network and trains nothing'
    assert "adapter_mode = 'on'" not in body


def test_eval_measures_the_gate_the_loader_will_actually_use():
    """stage_eval also forced "on", so its reported mean gate was a constant 1.000."""
    body = _stage("stage_eval")
    assert "adapter_mode = 'auto'" in body, "eval must measure the gate the loader uses by default"
    assert "adapter_mode = 'on'" not in body


def test_the_notebook_carries_the_fixed_script():
    """The notebook embeds this script; if they drift the notebook still trains the old gate.

    The notebook is read as JSON rather than grepped, because a cell's source escapes its quotes:
    the text in the file is `adapter_mode = \\"auto\\"`, not what the line looks like unescaped.
    """
    import json

    nb = json.loads((ROOT / "notebooks" / "vega_finetune_2xT4_kaggle.ipynb").read_text(encoding="utf-8"))
    written = "".join("".join(c["source"]) for c in nb["cells"]
                      if c["cell_type"] == "code" and "".join(c["source"]).startswith("%%writefile"))
    assert 'engine.adapter_mode = "auto"' in written, \
        "rerun notebooks/build_finetune_nb.py -- the notebook still says adapter_mode = 'on'"
    assert 'engine.adapter_mode = "on"' not in written
