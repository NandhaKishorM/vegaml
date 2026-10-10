"""The engine's well evaluation and step loop, on synthetic features: no checkpoint, no network.

The fast path fused two well evaluations into one (`well_potential` + `well_energy_at`) and dropped the
per-step `settled.all()` break, which was a GPU->host sync. Both changes must leave every number alone,
so this file pins the two things they could break: the fused math against the two calls it replaced, and
the freeze of a settled particle against a longer horizon.
"""
from __future__ import annotations

import torch

from vegaml._common import VegaEngine

CFG = {"latent_dim": 8, "world_dim": 8, "proj_hidden": 16, "n_steps": 6, "dt_max": 0.5,
       "settle_eps": 1e-3, "dropout": 0.0, "feature_dropout": 0.0, "ssm_state": 4}
D_IN = 12


def _engine(seed=0, settle_eps=1e-3):
    """`settle_eps` is a knob the tests turn: at 1.0 every particle settles on the first step, which is
    what the freeze test needs, while the default never settles and exercises the full horizon."""
    cfg = dict(CFG, settle_eps=settle_eps)
    torch.manual_seed(seed)
    return VegaEngine(d_in=D_IN, n_feat_layers=2, n_opt_layers=1, cfg=cfg).eval()


def _batch(eng, B=3, K=5, seed=0):
    """A batch whose last row carries a padded option, which every masking path has to survive."""
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(B, 2, 3, D_IN, generator=g)
    opts = torch.randn(B, K, 1, D_IN, generator=g)
    mask = torch.ones(B, K, dtype=torch.bool)
    mask[0, -1] = False
    qtype = torch.zeros(B, dtype=torch.long)
    return feats, opts, mask, qtype


def _wells(eng, B=3, K=5, seed=1):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(B, eng.D, generator=g)
    c = torch.randn(B, K, eng.D, generator=g)
    log_a = torch.randn(B, K, generator=g)
    sigma = torch.rand(B, K, generator=g) + 0.5
    mask = torch.ones(B, K, dtype=torch.bool)
    mask[1, -2:] = False
    return z, c, log_a, sigma, mask


def test_fused_well_evaluation_matches_the_two_calls_it_replaced():
    eng = _engine()
    z, c, log_a, sigma, mask = _wells(eng)
    u_ref, grad_ref = eng.potential_and_grad(z, c, log_a, sigma, mask)
    e_ref, _ = eng.well_energy(z, c, log_a, sigma, mask)
    u, grad, q = eng.well_potential(z, c, log_a, sigma, mask)
    e = eng.well_energy_at(q, log_a, mask)
    assert torch.allclose(u, u_ref, rtol=1e-6, atol=1e-6)
    assert torch.allclose(grad, grad_ref, rtol=1e-6, atol=1e-6)
    assert torch.allclose(e, e_ref, rtol=1e-6, atol=1e-6)


def test_a_settled_particle_does_not_move_when_the_horizon_grows():
    """What the removed `break` was allowed to assume: `halt=True` freezes a settled item."""
    eng = _engine(settle_eps=1.0)
    batch = _batch(eng)
    short = eng(*batch, n_steps=6, halt=True)
    long = eng(*batch, n_steps=12, halt=True)
    settled = short["steps_used"] < 6
    assert settled.any(), "the fixture must settle at least one particle inside the short horizon"
    assert torch.equal(short["logits"][settled], long["logits"][settled])
    assert torch.equal(short["steps_used"][settled], long["steps_used"][settled])


def test_the_loop_reports_the_step_a_particle_settled_in():
    eng = _engine()
    batch = _batch(eng)
    out = eng(*batch, n_steps=6, halt=True)
    assert out["steps_used"].shape == (3,)
    assert bool((out["steps_used"] >= 1).all()) and bool((out["steps_used"] <= 6).all())


def test_forward_is_deterministic_between_calls():
    """No host sync and no hidden state: two calls on the same input give the same tensors."""
    eng = _engine()
    batch = _batch(eng)
    a = eng(*batch, n_steps=4, halt=True, return_traj=False)
    b = eng(*batch, n_steps=4, halt=True, return_traj=False)
    assert torch.equal(a["logits"], b["logits"])
    assert torch.equal(a["zT"], b["zT"])
