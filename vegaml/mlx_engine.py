"""The decision engine on Apple silicon through MLX, without torch in the process.

    pip install "vegaml[mlx]"

    from vegaml.mlx_engine import MLXEngine
    eng = MLXEngine.from_dir("path/to/checkpoint")      # vega_config.json + engine.safetensors
    out = eng(feats, opts, opt_mask, qtype)             # numpy in, numpy out

This is the settling computation itself: pooled perception features in, well energies and the
physics readouts out. It reproduces the torch engine's inference numbers (the parity test holds
them to 1e-4) from the same engine.safetensors, read through safetensors' numpy loader, so a
process that only runs the engine needs neither torch nor a GPU runtime.

Scope: the base engine, eval semantics (dropout off), with the long-state memory reader when the
checkpoint has one. Gated adapters stay with the torch path; the calibrated probability layer
(thermal temperatures, conformal sets) lives above the engine and is framework-free already.

device="gpu" (the default) runs on Metal, whose matmul accumulates below full float32: readouts
land within about 1e-2 of the torch numbers and the parity test shows decisions and settling
steps unchanged. device="cpu" is bit-serious float32 and matches torch to 1e-4; pick it when
comparing numbers rather than serving them.
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, Optional

import mlx.core as mx
import numpy as np

QTYPES = {"choice": 0, "score": 1, "boolean": 2}   # ids fixed by the checkpoint format
_SCORE = QTYPES["score"]


def _gelu(x):
    return 0.5 * x * (1 + mx.erf(x / math.sqrt(2.0)))


def _softplus(x):
    return mx.logaddexp(x, 0.0)


def _layer_norm(x, w, b, eps=1e-5):
    mu = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    return (x - mu) / mx.sqrt(var + eps) * w + b


def _norm(x, axis=-1):
    return mx.sqrt(mx.sum(x * x, axis=axis))


class _MLP:
    """LayerNorm -> Linear -> GELU -> Linear, the engine's projector block at eval."""

    def __init__(self, sd, prefix):
        self.lnw, self.lnb = sd[prefix + ".net.0.weight"], sd[prefix + ".net.0.bias"]
        self.w1, self.b1 = sd[prefix + ".net.1.weight"], sd[prefix + ".net.1.bias"]
        self.w2, self.b2 = sd[prefix + ".net.4.weight"], sd[prefix + ".net.4.bias"]

    def __call__(self, x):
        x = _layer_norm(x, self.lnw, self.lnb)
        x = _gelu(x @ self.w1.T + self.b1)
        return x @ self.w2.T + self.b2


class _Foresight:
    """One gated residual dynamics step: s -> s + out(gelu(x) * sigmoid(gate))."""

    def __init__(self, sd, prefix):
        self.lnw, self.lnb = sd[prefix + ".norm.weight"], sd[prefix + ".norm.bias"]
        self.wi, self.bi = sd[prefix + ".inp.weight"], sd[prefix + ".inp.bias"]
        self.wo, self.bo = sd[prefix + ".out.weight"], sd[prefix + ".out.bias"]

    def __call__(self, s):
        u = _layer_norm(s, self.lnw, self.lnb) @ self.wi.T + self.bi
        x, gate = mx.split(u, 2, axis=-1)
        return s + (_gelu(x) * mx.sigmoid(gate)) @ self.wo.T + self.bo


class _Memory:
    """Question-steered attention over the state's chunk means (checkpoints with memory_chunks)."""

    def __init__(self, sd, prefix, heads=4):
        self.q = _MLP(sd, prefix + ".q")
        self.kv_lnw, self.kv_lnb = sd[prefix + ".kv.0.weight"], sd[prefix + ".kv.0.bias"]
        self.kv_w, self.kv_b = sd[prefix + ".kv.1.weight"], sd[prefix + ".kv.1.bias"]
        self.pos = sd[prefix + ".pos"]
        self.wo, self.bo = sd[prefix + ".out.weight"], sd[prefix + ".out.bias"]
        self.heads = heads

    def __call__(self, qx, ch, cmask):
        B, C, _ = ch.shape
        S = self.wo.shape[1]
        h, dh = self.heads, S // self.heads
        q = self.q(qx).reshape(B, h, dh)
        n = mx.maximum(mx.sum(cmask, axis=-1, keepdims=True), 1).astype(mx.float32)
        P = self.pos.shape[0]
        pidx = mx.minimum(((mx.arange(C)[None, :] + 0.5) / n * P).astype(mx.int32), P - 1)
        kv = _layer_norm(ch, self.kv_lnw, self.kv_lnb) @ self.kv_w.T + self.kv_b + self.pos[pidx]
        k, v = mx.split(kv, 2, axis=-1)
        k, v = k.reshape(B, C, h, dh), v.reshape(B, C, h, dh)
        att = mx.einsum("bhd,bchd->bhc", q, k) / math.sqrt(dh)
        att = mx.softmax(mx.where(cmask[:, None, :], att, -1e4), axis=-1)
        r = mx.einsum("bhc,bchd->bhd", att, v).reshape(B, S)
        lit = mx.any(cmask, axis=-1, keepdims=True).astype(mx.float32)
        return (r @ self.wo.T + self.bo) * lit


class MLXEngine:
    """The torch engine's forward, reproduced in MLX from the same engine.safetensors."""

    def __init__(self, cfg: Dict, state: Dict[str, np.ndarray], device: str = "gpu"):
        sd = {k: mx.array(np.asarray(v, dtype=np.float32)) for k, v in state.items()
              if not k.startswith("gate_nets")}
        if device not in ("gpu", "cpu"):
            raise ValueError("device must be \"gpu\" or \"cpu\", got %r" % device)
        self.device = mx.gpu if device == "gpu" else mx.cpu
        self.cfg = cfg
        self.D, self.S = cfg["latent_dim"], cfg["world_dim"]
        self.max_steps = cfg["n_steps"]
        self.dt_max = cfg["dt_max"]
        self.settle_eps = cfg["settle_eps"]
        self.force_max = cfg.get("force_max", 20.0)
        self.mix, self.opt_mix = sd["mix"], sd["opt_mix"]
        self.feat_mu, self.feat_sd = sd["feat_mu"], sd["feat_sd"]
        self.opt_mu, self.opt_sd = sd["opt_mu"], sd["opt_sd"]
        self.world = _MLP(sd, "world")
        self.foresight = _Foresight(sd, "foresight")
        self.probe = _MLP(sd, "probe")
        self.impulse = _MLP(sd, "impulse")
        self.qtype_emb = sd["qtype_emb.weight"]
        self.to_z0 = _MLP(sd, "to_z0")
        self.to_p0 = _MLP(sd, "to_p0")
        self.well = _MLP(sd, "well")
        self.rail_dir = _MLP(sd, "rail_dir")
        self.log_kappa = sd["log_kappa"]
        self.ssm_in_w, self.ssm_in_b = sd["ssm_in.weight"], sd["ssm_in.bias"]
        self.A_log = sd["A_log"]
        self.ssm_out_w, self.ssm_out_b = sd["ssm_out.weight"], sd["ssm_out.bias"]
        self.force_scale = sd["force_scale"]
        self.memory = _Memory(sd, "memory") if int(cfg.get("memory_chunks", 0) or 0) else None

    @classmethod
    def from_dir(cls, model_dir: str, device: str = "gpu") -> "MLXEngine":
        from safetensors.numpy import load_file
        with open(os.path.join(model_dir, "vega_config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        return cls(cfg, load_file(os.path.join(model_dir, "engine.safetensors")), device=device)

    # -- pieces, mirroring the torch engine
    def _mixed(self, feats, opts):
        f = (feats - self.feat_mu) / self.feat_sd                        # [B, Lf, 3, d]
        o = (opts - self.opt_mu) / self.opt_sd                           # [B, K, Lo, d]
        w = mx.softmax(self.mix, axis=-1)                                # [3, Lf]
        slots = mx.sum(f * w.T[None, :, :, None], axis=1)                # [B, 3, d]
        o = mx.sum(o * mx.softmax(self.opt_mix, axis=-1)[None, None, :, None], axis=2)   # [B, K, d]
        return slots, o

    def _landscape(self, o, opt_mask, qtype, probe):
        B, K, _ = o.shape
        pb = mx.broadcast_to(probe[:, None, :], (B, K, probe.shape[-1]))
        e = self.well(mx.concatenate([o, pb], axis=-1))
        c = e[..., :self.D] + self.qtype_emb[qtype][:, None, :]
        log_a = mx.clip(e[..., self.D], -4, 4)
        sigma = 0.3 + _softplus(e[..., self.D + 1])
        is_score = qtype == _SCORE
        if bool(mx.any(is_score)):
            m = opt_mask.astype(mx.float32)
            denom = mx.maximum(mx.sum(m, axis=1, keepdims=True), 1)
            r0 = mx.sum(c * m[..., None], axis=1) / denom
            u = self.rail_dir(probe)
            u = u / mx.maximum(_norm(u)[..., None], 1e-12)
            gaps = _softplus(e[..., self.D + 2]) * m
            t = mx.cumsum(gaps, axis=1)
            t = t - mx.sum(t * m, axis=1, keepdims=True) / denom
            rail = r0[:, None, :] + t[..., None] * u[:, None, :]
            c = mx.where(is_score[:, None, None], rail, c)
        return c, log_a, sigma

    def _well_potential(self, z, c, log_a, sigma, opt_mask):
        kappa = mx.exp(self.log_kappa)
        diff = z[:, None, :] - c
        d2 = mx.sum(diff * diff, axis=-1)
        q = d2 / (2 * sigma ** 2)
        g = mx.exp(log_a - q) * opt_mask.astype(mx.float32)
        U = 0.5 * kappa * mx.sum(z * z, axis=-1) - mx.sum(g, axis=-1)
        grad = kappa * z + mx.sum(g[..., None] * diff / (sigma ** 2)[..., None], axis=1)
        return U, grad, q

    def __call__(self, feats, opts, opt_mask, qtype, halt: bool = True,
                 chunks: Optional[np.ndarray] = None, chunk_mask: Optional[np.ndarray] = None) -> Dict:
        with mx.stream(self.device):
            return self._forward(feats, opts, opt_mask, qtype, halt, chunks, chunk_mask)

    def _forward(self, feats, opts, opt_mask, qtype, halt, chunks, chunk_mask) -> Dict:
        feats = mx.array(np.asarray(feats, dtype=np.float32))
        opts = mx.array(np.asarray(opts, dtype=np.float32))
        opt_mask = mx.array(np.asarray(opt_mask, dtype=bool))
        qtype = mx.array(np.asarray(qtype, dtype=np.int32))
        slots, o = self._mixed(feats, opts)
        s = self.world(slots[:, 0])
        if self.memory is not None and chunks is not None and chunk_mask is not None:
            cmask = mx.array(np.asarray(chunk_mask, dtype=bool))
            if bool(mx.any(cmask)):
                ch = mx.array(np.asarray(chunks, dtype=np.float32))
                ch = (ch - self.feat_mu[-1, 0]) / self.feat_sd[-1, 0]
                s = s + self.memory(mx.concatenate([slots[:, 1], slots[:, 2]], axis=-1), ch, cmask)
        s_next = self.foresight(s)
        probe = self.probe(slots[:, 1]) + self.qtype_emb[qtype]
        imp = self.impulse(slots[:, 2])
        ctx = mx.concatenate([s, s_next, probe, imp], axis=-1)
        z, p = self.to_z0(ctx), self.to_p0(ctx)
        c, log_a, sigma = self._landscape(o, opt_mask, qtype, probe)
        B = z.shape[0]
        h = mx.zeros((B, self.A_log.shape[0]))
        T = self.max_steps
        settled = mx.zeros((B,), dtype=mx.bool_)
        steps_used = mx.full((B,), T, dtype=mx.int32)
        U, grad, q = self._well_potential(z, c, log_a, sigma, opt_mask)
        gn = _norm(grad)
        A = mx.exp(self.A_log)
        for t in range(T):
            x = mx.concatenate([z, p, grad, gn[:, None]], axis=-1)
            u = x @ self.ssm_in_w.T + self.ssm_in_b
            bu, dt_raw, fr_raw = u[:, :-2], u[:, -2:-1], u[:, -1:]
            dt = self.dt_max * mx.sigmoid(dt_raw)
            gamma = _softplus(fr_raw)
            h = mx.exp(-A * dt) * h + dt * bu
            force = self.force_scale * mx.tanh(h @ self.ssm_out_w.T + self.ssm_out_b)
            g_eff = grad / (1 + gn[:, None] / self.force_max)
            p_new = p * mx.exp(-gamma * dt) - dt * g_eff + dt * force
            z_new = z + dt * p_new
            if halt:
                keep = settled[:, None]
                p_new = mx.where(keep, p, p_new)
                z_new = mx.where(keep, z, z_new)
            z, p = z_new, p_new
            U, grad, q = self._well_potential(z, c, log_a, sigma, opt_mask)
            gn = _norm(grad)
            kin = 0.5 * mx.sum(p * p, axis=-1)
            now = (kin < self.settle_eps) & (gn < 10 * self.settle_eps) & ~settled
            steps_used = mx.where(now, mx.full(steps_used.shape, t + 1, dtype=mx.int32), steps_used)
            settled = settled | now
        logits = mx.where(opt_mask, -(q - log_a), -1e4)
        diff = z[:, None, :] - c
        d2 = mx.sum(diff * diff, axis=-1)
        nearest = mx.min(mx.where(opt_mask, d2 / (2 * sigma ** 2), 1e4), axis=-1)
        out = {"logits": logits, "zT": z, "kinetic": 0.5 * mx.sum(p * p, axis=-1),
               "grad_norm": gn, "nearest": nearest, "steps_used": steps_used}
        return {k: np.asarray(v) for k, v in out.items()}
