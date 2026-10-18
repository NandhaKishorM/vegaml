"""Attach YOUR OWN frozen reader to the universal (vega-U) checkpoint, locally.

The engine is never retrained and the reader is never fine-tuned: a new reader only needs a
small linear port into the shared space, and fitting that port is self-supervised. The shared
space target for any text is whatever the checkpoint's own root reader (the anchor) computes
for it, so the training pairs come from running the anchor and the new reader over the same
prompts, on text YOU bring. No labels, and none of the original training data.

    import vegaml

    report = vegaml.attach("mistralai/Mistral-7B-Instruct-v0.3",
                           texts=my_texts,             # a few thousand strings of your domain
                           out_dir="vega-u-mistral-7b")
    v = vegaml.load(report["out_dir"])                 # the new reader, same engine

Calibration is the one step that wants labels, and it is deliberately separate: probabilities
from an uncalibrated dir are raw until :func:`calibrate` refits the temperature and thresholds
on your own labelled examples, which match your distribution better than anyone else's data.

The fit quality is reported as per-layer holdout cosines: how faithfully the port lands this
reader's features on the anchor's coordinates. Well below ~0.8 usually means too few or too
narrow texts.
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import time
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ._common import (
    QTYPES,
    Perception,
    build_prompt,
    conformal_thresholds,
    ece_score,
    fit_thermal,
    normalise_conformal,
    option_texts,
    resolve_device,
    seed_all,
    serialize_state,
    softmax_np,
    thermal_features,
    thermal_tau,
)
from ._universal import Port
from .hub import DEFAULT_REPO, VARIANTS, fetch

#: The probe questions every alignment text is read under. They are generic on purpose: the
#: port must map features for any question, so the probes cover the three answer types and a
#: spread of option counts. Both readers see the exact same (text, probe) prompts.
PROBES: Dict[str, Dict] = {
    "topic": {"type": "choice",
              "instructions": "Which of these best describes what the text is mainly about?",
              "criteria": {"money": "prices, billing, payments or finance",
                           "technical": "software, hardware or a technical procedure",
                           "people": "a person, a relationship or a conversation",
                           "events": "something that happened or is planned",
                           "opinion": "a judgement, review or argument",
                           "other": "none of the above"}},
    "problem": {"type": "boolean",
                "instructions": "Does the text describe a problem, fault or complaint?"},
    "request": {"type": "boolean",
                "instructions": "Does the writer ask for something to be done or answered?"},
    "tone": {"type": "choice",
             "instructions": "What is the overall tone of the text?",
             "criteria": {"positive": None, "neutral": None, "negative": None}},
    "urgency": {"type": "score",
                "instructions": "How urgent is the text?",
                "criteria": ["no urgency at all", "slightly time-sensitive",
                             "moderately urgent", "clearly urgent",
                             "critical, immediate action needed"]},
    "specificity": {"type": "score",
                    "instructions": "How specific and detailed is the text?",
                    "criteria": ["entirely vague", "mostly general", "some specifics",
                                 "quite detailed", "dense with specifics"]},
}


def _log(msg):
    print(time.strftime("%H:%M:%S"), "attach:", msg, flush=True)


# --------------------------------------------------------------------------------- ridge
def ridge_fit(X, Y, lams=(1e-3, 1e-2, 1e-1, 1.0, 10.0), chunk=65536):
    """Closed-form ridge X @ W + b -> Y with a 10% holdout choosing lambda; returns
    (W [d_in, d_out], b [d_out], holdout cosine, lambda). Gram matrices accumulate in chunks.
    The same fit the published cold-attached readers' ports started from."""
    n, d = X.shape
    nh = min(max(2048, n // 10), n // 2)
    perm = torch.randperm(n, device=X.device)
    tr, ho = perm[nh:], perm[:nh]
    mx = X[tr].float().mean(0)
    my = Y[tr].float().mean(0)
    G = torch.zeros(d, d, device=X.device)
    C = torch.zeros(d, Y.shape[1], device=X.device)
    for s in range(0, len(tr), chunk):
        xc = X[tr[s:s + chunk]].float() - mx
        yc = Y[tr[s:s + chunk]].float() - my
        G += xc.T @ xc
        C += xc.T @ yc
    Xh, Yh = X[ho].float(), Y[ho].float()
    best = None
    eye = torch.eye(d, device=X.device)
    for lam in lams:
        W = torch.linalg.solve(G + lam * len(tr) * eye, C)
        pred = (Xh - mx) @ W + my
        cos = float(F.cosine_similarity(pred, Yh, dim=-1).mean())
        if best is None or cos > best[0]:
            best = (cos, lam, W)
    cos, lam, W = best
    return W, my - mx @ W, cos, lam


def fit_port(x_items, x_opts, y_items, y_opts, d_shared, x_extra=None, y_extra=None,
             device=None):
    """Fit one linear map per feature layer from standardized features onto shared-space
    targets: the span rows per layer, plus the option rows (and any extra rows, such as memory
    chunk summaries) jointly at the deepest layer, exactly as the training-side attach does.

    x_items [N, Lf, 3, d_in] -> y_items [N, Lf, 3, d_shared]; x_opts [M, d_in] -> y_opts
    [M, d_shared]. Returns (port on CPU, per-layer holdout cosines)."""
    dev = device if device is not None else x_items.device
    Lf, d_in = x_items.shape[1], x_items.shape[-1]
    port = Port(d_in, int(d_shared), Lf)
    cosines = []
    for li in range(Lf):
        X = x_items[:, li].reshape(-1, d_in)
        Y = y_items[:, li].reshape(-1, y_items.shape[-1])
        if li == Lf - 1:
            X = torch.cat([X, x_opts], 0)
            Y = torch.cat([Y, y_opts], 0)
            if x_extra is not None and len(x_extra):
                X = torch.cat([X, x_extra], 0)
                Y = torch.cat([Y, y_extra], 0)
        W, b, cos, lam = ridge_fit(X.to(dev), Y.to(dev))
        with torch.no_grad():
            port.lin[li].weight.copy_(W.T.cpu())
            port.lin[li].bias.copy_(b.cpu())
        cosines.append(round(cos, 4))
        _log("ridge layer %d: holdout cosine %.4f (lambda %g, %d rows)" % (li, cos, lam, len(X)))
    return port, cosines


# --------------------------------------------------------------------------------- extraction
def _probe_questions():
    from ._api import Vega as _ApiVega
    return [(qid, _ApiVega._internal(q)) for qid, q in PROBES.items()]


def _extract_bank(perc, cfg, texts, qs, batch_size=16, chunks=0):
    """Run every (text, probe) prompt through one frozen reader; raw pooled features on CPU.

    All of one text's probe items go through pooled() together, so a long text's state prefix
    is read once and shared, the same way decide() reads it."""
    feats, opts, ch, cm = [], [], [], []
    nopt, qtype = [], []
    pm = cfg.get("prefix_min_tokens", 4096)
    buf: List[Dict] = []

    def flush():
        if not buf:
            return
        if chunks:
            f, o, c, m = perc.pooled(buf, chunks=chunks, prefix_min=pm,
                                     chunk_min_tokens=cfg.get("memory_min_tokens", 2048),
                                     chunk_tokens=cfg.get("memory_chunk_tokens", 128))
            ch.append(c.float().cpu())
            cm.append(m.cpu())
        else:
            f, o = perc.pooled(buf, prefix_min=pm)
        feats.append(f.float().cpu())
        opts.append(o.float().cpu())
        buf.clear()

    done = 0
    for text in texts:
        for _, q in qs:
            k = len(option_texts(q))
            ids, spans = build_prompt(perc.tok, text, q, cfg["max_len"],
                                      option_order=list(range(k)),
                                      opt_max_tokens=cfg["opt_max_tokens"],
                                      query_first_max_state=cfg.get("query_first_max_state", 0))
            buf.append({"ids": np.asarray(ids, dtype=np.int32), "spans": spans})
            nopt.append(k)
            qtype.append(QTYPES[q["t"]])
        if len(buf) >= batch_size:
            flush()
        done += 1
        if done % 200 == 0:
            _log("read %d/%d texts" % (done, len(texts)))
    flush()
    nopt_t = torch.tensor(nopt)
    return {"items": torch.cat(feats), "opts": torch.cat(opts), "nopt": nopt_t,
            "ptr": torch.cat([torch.zeros(1, dtype=torch.long), nopt_t.cumsum(0)]),
            "qtype": torch.tensor(qtype),
            "chunks": torch.cat(ch) if ch else None,
            "cmask": torch.cat(cm) if cm else None}


def _pad_opts(flat, bank, idx):
    """[sum K, ...] flat option rows of items `idx` as padded [B, K, ...] plus the mask."""
    n = bank["nopt"][idx]
    K = int(n.max())
    B = len(idx)
    out = flat.new_zeros((B, K) + flat.shape[1:])
    mask = torch.zeros(B, K, dtype=torch.bool)
    for b in range(B):
        a = int(bank["ptr"][idx[b]])
        k = int(n[b])
        out[b, :k] = flat[a:a + k]
        mask[b, :k] = True
    return out, mask


def _opt_rows(bank, idx):
    """Flat option-row indices of items `idx`, in item order (aligned across readers, since
    the probes give every reader the same option counts)."""
    return torch.cat([torch.arange(int(bank["ptr"][i]), int(bank["ptr"][i + 1]))
                      for i in idx.tolist()])


def _engine_logits(engine, bank, device, cmask=None, bs=256):
    """The engine's raw logits on every item of a RAW bank (the UniversalEngine standardizes
    and ports inside), padded to the bank's widest option count with -1e4."""
    N = len(bank["qtype"])
    Kmax = int(bank["nopt"].max())
    out = torch.full((N, Kmax), -1e4)
    use_chunks = bank["chunks"] is not None and getattr(engine, "memory", None) is not None
    cmask = bank["cmask"] if cmask is None else cmask
    with torch.no_grad():
        for s in range(0, N, bs):
            idx = torch.arange(s, min(s + bs, N))
            o, m = _pad_opts(bank["opts"], bank, idx)
            kw = {}
            if use_chunks:
                kw = {"chunks": bank["chunks"][idx].to(device),
                      "chunk_mask": cmask[idx].to(device)}
            r = engine(bank["items"][idx].to(device), o.to(device), m.to(device),
                       bank["qtype"][idx].to(device), return_traj=False, **kw)
            lg = r["logits"].float().cpu()
            block = torch.full((len(idx), Kmax), -1e4)
            block[:, :lg.shape[1]] = lg.masked_fill(~m, -1e4)
            out[idx] = block
    return out


def _layer_specs(repo_or_path, anchor_cfg, token):
    """The fractional layer taps the universal model was trained with. The checkpoint repo
    ships them in universal/universal_config.json; failing that, they are recovered from the
    anchor's resolved indices."""
    try:
        if repo_or_path and os.path.isdir(repo_or_path):
            with open(os.path.join(repo_or_path, "universal", "universal_config.json")) as f:
                u = json.load(f)
        else:
            from huggingface_hub import snapshot_download
            name = (repo_or_path or "0.8b").strip().lower()
            repo = DEFAULT_REPO if name in VARIANTS else repo_or_path
            local = snapshot_download(repo, token=token,
                                      allow_patterns=["universal/universal_config.json"])
            with open(os.path.join(local, "universal", "universal_config.json")) as f:
                u = json.load(f)
        return u["feature_layers"], u["option_layers"]
    except Exception:
        n = anchor_cfg["n_layers"]
        return ([(i + 1) / n for i in anchor_cfg["feature_layers"]],
                [(i + 1) / n for i in anchor_cfg["option_layers"]])


# --------------------------------------------------------------------------------- attach
def _copy_fresh(src, dst):
    """Copy src to dst as a writable file. src usually sits in the read-only HF cache, and
    shutil.copy would carry that mode over: a second attach into the same out_dir would then
    open a read-only destination and die with PermissionError."""
    if os.path.exists(dst):
        os.chmod(dst, 0o644)
    shutil.copyfile(src, dst)


def attach(backbone: str, texts: Sequence[str], out_dir: str,
           repo_or_path: Optional[str] = None, examples: Optional[Sequence] = None,
           questions: Optional[Mapping] = None, refine_steps: int = 300,
           batch_size: int = 16, max_len: Optional[int] = None, device: Optional[str] = None,
           token=None, seed: int = 13) -> Dict:
    """Fit a port for a new frozen reader onto the universal engine, from your own text.

    `backbone` is the new reader, a Hub id or a local path; it is never fine-tuned.
    `texts` is the alignment corpus, plain strings from your domain: 1,000 or more is a good
    corpus and 100 is the floor; diverse beats big, and long records matter if you will read
    long records. `out_dir` receives a complete servable model dir that ``vegaml.load(out_dir)``
    loads like any shipped reader, same engine byte for byte.

    With `examples` (and `questions`, the shapes :meth:`vegaml.Vega.fit` takes), the new dir is
    calibrated in the same call; without them it is written uncalibrated and :func:`calibrate`
    adds the block later. `max_len` caps the extraction prompt length only, to trade
    long-record fidelity for speed; the written config keeps the full length.

    Returns a report: per-layer ridge cosines, item counts, and the calibration summary."""
    from ._api import Vega as _ApiVega

    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if len(texts) < 100:
        raise ValueError("attach needs at least 100 non-empty texts (got %d); 1,000+ is a "
                         "good alignment corpus" % len(texts))
    dev = resolve_device(device)
    seed_all(seed)
    anchor_dir = fetch(repo_or_path, token=token)
    with open(os.path.join(anchor_dir, "vega_config.json")) as f:
        anchor_cfg = json.load(f)
    if not anchor_cfg.get("port"):
        raise ValueError("%r is not a universal (vega-U) checkpoint: it has no port, so "
                         "there is no shared space to attach a reader to" % anchor_dir)
    feat_spec, opt_spec = _layer_specs(repo_or_path, anchor_cfg, token)
    ex_cfg = dict(anchor_cfg)
    if max_len:
        ex_cfg["max_len"] = int(max_len)
    qs = _probe_questions()

    # 1. the anchor reads every (text, probe) prompt; its UniversalEngine defines the targets
    _log("loading the anchor reader (%s)" % anchor_cfg.get("backbone_name", "repo root"))
    anchor = _ApiVega(anchor_dir, device=str(dev))
    C = int(getattr(anchor.engine, "memory_chunks", 0) or 0)
    bank_a = _extract_bank(anchor.perc, ex_cfg, texts, qs, batch_size=batch_size, chunks=C)
    uni_a = anchor.engine     # port + statistics + the frozen engine; tiny next to the reader
    anchor = None             # free the anchor reader before the new one loads
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # 2. the new reader reads the same prompts
    _log("loading the new reader (%s)" % backbone)
    perc = Perception(backbone, dev, feat_spec, opt_spec)
    bank_b = _extract_bank(perc, ex_cfg, texts, qs, batch_size=batch_size, chunks=C)
    assert torch.equal(bank_a["nopt"], bank_b["nopt"])

    # 3. targets: standardized anchor features through the anchor's own port
    with torch.no_grad():
        mu, sd = uni_a.mu.cpu(), uni_a.sd.cpu()
        omu, osd = uni_a.omu.cpu(), uni_a.osd.cpu()
        port_a = uni_a.port.cpu()
        y_items = port_a.items((bank_a["items"] - mu) / sd)
        y_opts = port_a.opts((bank_a["opts"] - omu) / osd)[:, -1]
        pair = y_ch = None
        if bank_a["chunks"] is not None:
            # chunking is decided per reader by token count, so only rows both readers chunked
            pair = bank_a["cmask"] & bank_b["cmask"]
            y_ch = port_a.chunks((bank_a["chunks"] - mu[-1, 0]) / sd[-1, 0])[pair]
        uni_a.port.to(dev)

    # 4. standardize the new reader's features with its own statistics, then ridge-fit
    stats = {"mu": bank_b["items"].mean(0), "sd": bank_b["items"].std(0).clamp_min(1e-3),
             "omu": bank_b["opts"].mean(0), "osd": bank_b["opts"].std(0).clamp_min(1e-3)}
    x_items = (bank_b["items"] - stats["mu"]) / stats["sd"]
    x_opts_full = (bank_b["opts"] - stats["omu"]) / stats["osd"]       # [M, Lo, d_in]
    x_ch_full = None
    x_ch = None
    if pair is not None:
        x_ch_full = (bank_b["chunks"] - stats["mu"][-1, 0]) / stats["sd"][-1, 0]
        x_ch = x_ch_full[pair]
    d_shared = int(anchor_cfg["d_shared"])
    port, cosines = fit_port(x_items, x_opts_full[:, -1], y_items, y_opts, d_shared,
                             x_extra=x_ch, y_extra=y_ch, device=dev)

    # 5. a short refine against the frozen engine itself: the ported features must not only
    # land near the anchor's coordinates, they must make the engine DECIDE the same way, so
    # the engine's own answers through the anchor are distilled into the port
    if refine_steps:
        eng = uni_a.engine
        for p in eng.parameters():
            p.requires_grad_(False)
        eng.eval()
        _log("teacher pass: the engine's answers through the anchor")
        t_logits = _engine_logits(uni_a, bank_a, dev, cmask=pair)
        t_mask = t_logits > -1e3
        t_probs = torch.softmax(t_logits.masked_fill(~t_mask, -1e4), -1)
        port.to(dev)
        opt = torch.optim.AdamW(port.parameters(), lr=3e-4, weight_decay=0.01)
        N = len(bank_b["qtype"])
        bs = min(256, N)
        use_chunks = pair is not None and getattr(eng, "memory", None) is not None
        for step in range(refine_steps):
            idx = torch.randint(0, N, (bs,))
            f = port.items(x_items[idx].to(dev))
            o_pad, m = _pad_opts(x_opts_full, bank_b, idx)
            o = port.opts(o_pad.to(dev))
            kw = {}
            if use_chunks:
                kw = {"chunks": port.chunks(x_ch_full[idx].to(dev)),
                      "chunk_mask": pair[idx].to(dev)}
            r = eng(f, o, m.to(dev), bank_b["qtype"][idx].to(dev), prestandardized=True,
                    return_traj=False, **kw)
            sl = r["logits"].float()
            K = sl.shape[1]
            tp = t_probs[idx, :K].to(dev)
            tm = t_mask[idx, :K].to(dev)
            kl = -(tp * torch.log_softmax(sl.masked_fill(~tm, -1e4), -1)).sum(-1).mean()
            rows = _opt_rows(bank_b, idx)
            om = port.opts(x_opts_full[rows].to(dev))[:, -1]
            anchor_mse = (F.mse_loss(f, y_items[idx].to(dev))
                          + F.mse_loss(om, y_opts[rows].to(dev)))
            loss = kl + anchor_mse
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if (step + 1) % 50 == 0:
                _log("refine %d/%d | distill %.4f | anchor %.4f"
                     % (step + 1, refine_steps, float(kl), float(anchor_mse)))
        port.cpu()

    # 6. a complete servable dir: the UNCHANGED engine, this reader's port and statistics
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    from safetensors.torch import save_file
    _copy_fresh(os.path.join(anchor_dir, "engine.safetensors"),
                os.path.join(out_dir, "engine.safetensors"))
    save_file({k: v.detach().float().contiguous().cpu() for k, v in port.state_dict().items()},
              os.path.join(out_dir, "port.safetensors"))
    np.savez(os.path.join(out_dir, "stats.npz"),
             **{k: v.numpy().astype(np.float32) for k, v in stats.items()})
    cfg = {k: v for k, v in anchor_cfg.items() if k not in ("calibration", "calibration_teacher")}
    cfg.update({"backbone": backbone, "backbone_name": os.path.basename(backbone.rstrip("/")),
                "n_layers": perc.n_layers, "feature_layers": perc.feature_layers,
                "option_layers": perc.option_layers, "port": "port.safetensors",
                "raw_stats": "stats.npz",
                "attached_by": "vegaml.attach", "attach_texts": len(texts),
                "attach_ridge_cosines": cosines, "attach_refine_steps": int(refine_steps)})
    with open(os.path.join(out_dir, "vega_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    report = {"out_dir": out_dir, "backbone": backbone, "texts": len(texts),
              "items": int(len(bank_b["qtype"])), "d_model": int(perc.d_model),
              "d_shared": d_shared, "ridge_cosines": cosines,
              "refine_steps": int(refine_steps), "calibration": None}
    _log("wrote %s (engine unchanged, byte for byte)" % out_dir)

    # 7. optional calibration in the same call, reusing the reader already in memory
    if examples is not None:
        if not questions:
            raise ValueError("calibrating needs `questions`, the question definitions the "
                             "examples' labels answer (same shapes as Vega.fit)")
        v = _ApiVega(out_dir, backbone=backbone, device=str(dev), perception=perc)
        report["calibration"] = calibrate(v, examples, questions)
    else:
        _log("no examples given, so the dir is uncalibrated: probabilities are raw until "
             "vegaml.calibrate(dir, examples, questions) refits them on your labelled data")
    return report


# --------------------------------------------------------------------------------- calibration
def calibrate(model, examples: Sequence, questions: Mapping, risks=("0.05", "0.1", "0.2"),
              alphas=(0.1, 0.05), seed: int = 13) -> Dict:
    """Refit a model dir's calibration on YOUR labelled examples: the thermal temperature (per
    question type, physics-conditioned where it beats a plain temperature on a holdout), the
    conformal prediction-set thresholds, the abstain thresholds and the unbound threshold.

    `model` is a loaded ``vegaml.Vega``, the inner engine object, or a model dir path.
    `examples` are ``(state, {question_id: label})`` pairs, exactly what :meth:`vegaml.Vega.fit`
    takes, and `questions` maps each id to its definition. Calibration never transfers between
    readers, so this is also THE step after :func:`attach`. Roughly 500 examples per question
    type is usable and 1,000 to 2,000 is where the numbers settle; below 30 for a type, that
    type keeps a neutral temperature and the report says so.

    The block is written into the dir's ``vega_config.json`` and applied to the loaded model in
    place. Returns a summary with per-type counts, the chosen temperature form, and ECE before
    and after."""
    from . import label_ids
    from ._api import Vega as _ApiVega

    if isinstance(model, str):
        v = _ApiVega(model)
    elif hasattr(model, "heads") and hasattr(model, "engine"):   # the vegaml.Vega wrapper
        v = model.engine
    else:
        v = model
    seed_all(seed)
    rows_logits, rows_phys, rows_qt, rows_label, rows_near = [], [], [], [], []
    Kmax = max(len(label_ids(q)) for q in questions.values())
    with torch.no_grad():
        for example in examples:
            state, labels = example[0], example[1]
            qids = [q for q in labels if q in questions]
            if not qids:
                continue
            qs = [v._internal(questions[q]) for q in qids]
            ident = [list(range(len(option_texts(q)))) for q in qs]
            run = v._view(serialize_state(state), qs, 0, ident)
            phys = thermal_features(run["out"], run["mask"], v.engine.max_steps).cpu().numpy()
            for r, qid in enumerate(qids):
                ids = label_ids(questions[qid])
                lab = str(labels[qid]).lower() if questions[qid]["type"] == "boolean" \
                    else str(labels[qid])
                if lab not in ids:
                    raise ValueError("label %r for %r is not one of its options %s"
                                     % (labels[qid], qid, ids))
                k = len(ids)
                row = np.full(Kmax, -1e4)
                row[:k] = run["logits"][r, :k]
                rows_logits.append(row)
                rows_phys.append(phys[r])
                rows_qt.append(QTYPES[questions[qid]["type"]])
                rows_label.append(ids.index(lab))
                rows_near.append(float(run["out"]["nearest"][r]))
    n = len(rows_logits)
    if n < 30:
        raise ValueError("calibration needs at least 30 labelled answers (got %d); 500+ per "
                         "question type is where it starts being trustworthy" % n)
    L = np.asarray(rows_logits)
    P = np.asarray(rows_phys)
    Q = np.asarray(rows_qt)
    Y = np.asarray(rows_label)
    T = np.zeros((n, Kmax))
    T[np.arange(n), Y] = 1.0

    # per question type, a physics-conditioned temperature only where it BEATS a plain one
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    cut = max(1, int(0.2 * n))
    ho, tr = order[:cut], order[cut:]
    theta = np.zeros((3, P.shape[1] + 1))
    chosen = {}
    th_phys = fit_thermal(L[tr], T[tr], Q[tr], P[tr], use_phys=True)
    th_flat = fit_thermal(L[tr], T[tr], Q[tr], P[tr], use_phys=False)
    for qt in range(3):
        if int((Q == qt).sum()) < 30:
            if (Q == qt).any():
                chosen[qt] = "neutral (fewer than 30 examples)"
            continue
        sel = ho[Q[ho] == qt]
        if len(sel) >= 10:
            nlls = []
            for th in (th_phys, th_flat):
                pr = softmax_np(L[sel], thermal_tau(th, Q[sel], P[sel]))
                nlls.append(float(-np.log(np.clip(pr[np.arange(len(sel)), Y[sel]],
                                                  1e-12, 1)).mean()))
            use = th_phys if nlls[0] <= nlls[1] else th_flat
        else:
            use = th_flat
        theta[qt] = use[qt]
        chosen[qt] = "physics-conditioned" if use is th_phys else "plain temperature"

    probs = softmax_np(L, thermal_tau(theta, Q, P))
    conf = probs.max(-1)
    correct = (probs.argmax(-1) == Y).astype(np.float64)
    raw = softmax_np(L, np.ones(n))
    cal = {
        "theta": theta.tolist(),
        "conformal": conformal_thresholds(probs, Y, Q, alphas=alphas),
        "abstain_threshold": {r: _abstain_threshold(conf, correct, float(r)) for r in risks},
        "default_risk": "0.1" if "0.1" in risks else str(risks[0]),
        "unbound_threshold": float(np.quantile(np.asarray(rows_near), 0.995)),
        "fitted_by": "vegaml.calibrate", "n_examples": int(n),
    }
    cfg_path = os.path.join(v.model_dir, "vega_config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["calibration"] = cal
    # A Hub-loaded model's dir is an HF cache snapshot, where every file is a symlink into a
    # content-addressed blob store. Writing through the link would corrupt the shared blob, so
    # the link is replaced with a real file: the snapshot gets the new calibration, the blob
    # keeps the bytes its hash promises, and a fresh download still restores the shipped one.
    if os.path.islink(cfg_path):
        os.unlink(cfg_path)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    v.theta, v.abstain, v.risk = theta, cal["abstain_threshold"], cal["default_risk"]
    v.unbound = cal["unbound_threshold"]
    v.conformal = normalise_conformal(cal["conformal"])
    names = {0: "choice", 1: "score", 2: "boolean"}
    return {"n": int(n),
            "per_type": {names[qt]: int((Q == qt).sum()) for qt in range(3) if (Q == qt).any()},
            "temperature": {names[qt]: how for qt, how in chosen.items()},
            "ece_before": round(ece_score(raw.max(-1), (raw.argmax(-1) == Y)), 4),
            "ece_after": round(ece_score(conf, correct), 4),
            "accuracy": round(float(correct.mean()), 4),
            "written_to": cfg_path}


def _abstain_threshold(conf, correct, risk):
    """The lowest confidence at which answering everything at or above it keeps the error rate
    at or under `risk` on these examples; 1.0 when no threshold achieves it."""
    order = np.argsort(-conf)
    c, ok = conf[order], correct[order]
    err = np.cumsum(1.0 - ok) / np.arange(1, len(ok) + 1)
    feas = np.nonzero(err <= risk)[0]
    if len(feas) == 0:
        return 1.0
    return float(c[feas.max()])
