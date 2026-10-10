"""Train a browser element-choice adapter from WebChain click trajectories.

Run extract / train / eval as in vega_finetune.py; --data selects the local JSONL.
extract --stats-only converts rows without a model, and extract --limit 8 --smoke
builds real tokenizer prompts without loading backbone weights or caching features.
Test websites follow the parent exporter; validation holds out training-side tasks.
WebChain has no unrelated labeled questions, so gate negatives reuse real AG News.
VEGA_WORK defaults to /kaggle/working; browser caches are isolated from emotion caches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# Which size to fine-tune. 0.8b is the default and the one this notebook is timed against; 4b is
# the same architecture with a wider backbone and trains the same way, for longer.
SIZES = {"0.8b": "Qwen/Qwen3.5-0.8B", "4b": "Qwen/Qwen3.5-4B"}
ALIASES = {"0.8b": "0.8b", "08b": "0.8b", "8b": "0.8b", "4b": "4b", "04b": "4b", "0.4b": "4b"}
SIZE = ALIASES.get((os.environ.get("VEGA_SIZE") or "0.8b").strip().lower().rstrip("b") + "b")
if SIZE is None:
    raise SystemExit("VEGA_SIZE must be one of %s" % ", ".join(SIZES))

# A size name is enough: the package resolves it to the right folder of the checkpoint repository.
MODEL_REPO = os.environ.get("VEGA_MODEL_REPO") or SIZE
BACKBONE = os.environ.get("VEGA_BACKBONE") or SIZES[SIZE]

# Optional. The default checkpoint is public and trains with no token at all. Set HF_TOKEN to read
# a private or gated checkpoint, or to push the adapter you train here to your own account.
HF_TOKEN = os.environ.get("HF_TOKEN") or None

WORK = os.environ.get("VEGA_WORK", "/kaggle/working")
# Per size, because the two backbones pool to different widths and the caches must not be mixed.
CACHE = os.path.join(WORK, "features-webchain-" + SIZE)
OUT = os.path.join(WORK, "adapter-webchain-" + SIZE)

DATA = "/home/ckl/projects/S/laya/finetune/out/webchain_cases.jsonl"


def element_text(action):
    # Short labels leave room for all candidates, matching the parent's compact element form.
    text = " ".join(str(action.get("label") or "").split())[:50]
    if action.get("role"):
        text += " (%s)" % action["role"]
    value = action.get("current_value", action.get("value"))
    if value is not None and value != "":
        text += " = %r" % str(value)[:30]
    for key in ("checked", "selected", "expanded"):
        if action.get(key) is not None:
            text += " %s=%s" % (key, action[key])
    return text


def convert_rows(path, limit=None):
    from vegaml._common import serialize_state

    splits = {"train": [], "valid": [], "test": []}
    counts, total = [], 0
    with open(path) as fh:
        for index, line in enumerate(fh):
            if limit is not None and index >= limit:
                break
            row = json.loads(line)
            total += 1
            # Local trajectories use CLICK while the documented schema uses click.
            if row["gold_op"].lower() != "click":
                continue
            page = row["page_obj"]
            criteria = {a["id"]: element_text(a) for a in page["actions"]}
            if len(criteria) != len(page["actions"]) or row["gold_id"] not in criteria:
                raise ValueError("row %d: duplicate candidate ids or missing gold id" % index)
            question = {"type": "choice", "instructions": "Goal: %s  Which element should be clicked next?"
                        % row["goal"], "criteria": criteria}
            state = serialize_state({"page": {k: page[k] for k in ("url", "title", "text")},
                                     "recent_actions": row.get("history") or []})
            held = hashlib.md5(row["website"].encode()).digest()[0] % 5 == 0
            # Hold out whole tasks for calibration so adjacent trajectory steps cannot leak into training.
            valid = hashlib.md5(row["task_id"].encode()).digest()[0] % 10 == 0
            split = "test" if held else "valid" if valid else "train"
            splits[split].append({"state": state, "question": question,
                                  "y": list(criteria).index(row["gold_id"]), "gold_id": row["gold_id"],
                                  "row": index})
            counts.append(len(criteria))
    log("rows %d; clicks %d; non-clicks %d" % (total, len(counts), total - len(counts)))
    log("website split: train %d, test %d" %
        (len(splits["train"]) + len(splits["valid"]), len(splits["test"])))
    log("adapter split: train %d, valid %d, test %d" % tuple(len(splits[k]) for k in splits))
    if counts:
        log("candidates: min %d, median %g, max %d" % (min(counts), np.median(counts), max(counts)))
    return splits


# The gate's negatives: a different public task, so "should this adapter fire" is a real question.
NEG_QUESTION = {
    "type": "choice",
    "instructions": "Which section does this news item belong to?",
    "criteria": {"world": "international affairs", "sports": "sport",
                 "business": "companies and markets", "scitech": "science and technology"},
}


def pick_device(rank=0):
    """cuda on Kaggle, mps on an Apple laptop, cpu otherwise -- so the same script is the one that
    gets smoke-tested locally before it is trusted on two T4s."""
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def log(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------- stage 1: features
def build_items(vega, rows, question=None):
    """Keep identity option order so gold indices always refer to the original candidate ids."""
    from vegaml._common import TEMPLATES, build_prompt, option_texts

    items, k = [], 0
    for row in rows:
        qi = vega.engine_api._internal(question or row["question"])
        texts = option_texts(qi)
        state = row if question else row["state"]
        expected = [vega.perc.tok(TEMPLATES[0]["opt"].format(o=t),
                                  add_special_tokens=False)["input_ids"] for t in texts]
        # Preserve full compact descriptions because the default option token cap can cut element state.
        ids, spans = build_prompt(vega.perc.tok, state, qi, vega.cfg["max_len"],
                                  option_order=list(range(len(texts))),
                                  opt_max_tokens=max(map(len, expected)),
                                  ins_max_tokens=vega.cfg["max_len"],
                                  query_first_max_state=vega.cfg.get("query_first_max_state", 0))
        if [ids[a:b] for a, b in spans["options"]] != expected:
            raise ValueError("candidate text exceeds prompt budget; refusing truncated options")
        items.append({"ids": np.asarray(ids, dtype=np.int32), "spans": spans})
        k = max(k, len(texts))
    return items, k


def smoke_prompts(splits):
    from types import SimpleNamespace

    from transformers import AutoTokenizer

    import vegaml
    from vegaml._api import Vega as Engine

    path = vegaml.fetch(MODEL_REPO, token=HF_TOKEN)
    with open(os.path.join(path, "vega_config.json")) as fh:
        cfg = json.load(fh)
    tok = AutoTokenizer.from_pretrained(BACKBONE, local_files_only=True)
    v = SimpleNamespace(perc=SimpleNamespace(tok=tok), cfg=cfg, engine_api=Engine)
    rows = sorted([r for split in splits.values() for r in split], key=lambda r: r["row"])
    if not rows:
        raise SystemExit("smoke needs at least one click row")
    items, _ = build_items(v, rows)
    for row in rows:
        assert row["gold_id"] in row["question"]["criteria"]
    if items:
        print("row %d decoded prompt:\n%s" % (rows[0]["row"], tok.decode(items[0]["ids"])))
        print("row %d option spans: %d" % (rows[0]["row"], len(items[0]["spans"]["options"])))
    log("smoke: %d prompts; exact candidate token spans and gold membership passed" % len(items))


@torch.no_grad()
def pool_batch(vega, items, k):
    feats, opt_flat = vega.perc.pooled(items, prefix_min=vega.cfg.get("prefix_min_tokens", 4096))
    B = len(items)
    opts = torch.zeros((B, k) + opt_flat.shape[1:], device=feats.device)
    mask = torch.zeros((B, k), dtype=torch.bool, device=feats.device)
    o = 0
    for b, it in enumerate(items):
        kk = len(it["spans"]["options"])
        opts[b, :kk] = opt_flat[o:o + kk]
        mask[b, :kk] = True
        o += kk
    return feats.float().cpu(), opts.float().cpu(), mask.cpu()


def stage_extract(args):
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    device = pick_device(rank)
    log("rank %d/%d on %s" % (rank, world, device))

    splits = convert_rows(args.data, args.limit)
    if args.stats_only:
        return
    if args.smoke:
        smoke_prompts(splits)
        return

    from datasets import load_dataset

    import vegaml
    from vegaml._api import Vega as Engine

    path = vegaml.fetch(MODEL_REPO, token=HF_TOKEN)
    eng = Engine(path, backbone=BACKBONE, device=device, adapters="off")

    class Holder:                      # the two attributes build_items needs, without a wrapper class
        pass

    v = Holder()
    v.perc, v.cfg, v.engine_api = eng.perc, eng.cfg, eng

    os.makedirs(CACHE, exist_ok=True)
    for name, rows in splits.items():
        texts, labels = rows, [r["y"] for r in rows]
        # All ranks and batches share one padded width per split so their caches concatenate safely.
        width = max((len(r["question"]["criteria"]) for r in rows), default=0)
        shard = range(rank, len(texts), world)
        idx = list(shard)
        log("%s: %d of %d rows on this rank" % (name, len(idx), len(texts)))
        F_, O_, M_, Y_ = [], [], [], []
        t0 = time.time()
        for s in range(0, len(idx), args.batch):
            chunk = idx[s:s + args.batch]
            items, k = build_items(v, [texts[i] for i in chunk])
            f, o, m = pool_batch(v, items, width)
            F_.append(f)
            O_.append(o)
            M_.append(m)
            Y_ += [labels[i] for i in chunk]
            if s % (args.batch * 20) == 0:
                done = s + len(chunk)
                log("  %s %d/%d  %.0f items/s" % (name, done, len(idx), done / max(time.time() - t0, 1e-9)))
        if not Y_:
            log("%s rank %d: empty shard, no cache written" % (name, rank))
            continue
        np.savez(os.path.join(CACHE, "%s_rank%d.npz" % (name, rank)),
                 feats=torch.cat(F_).numpy(), opts=torch.cat(O_).numpy(),
                 mask=torch.cat(M_).numpy(), y=np.array(Y_, dtype=np.int64))
        log("%s rank %d: %d rows in %.0fs" % (name, rank, len(Y_), time.time() - t0))

    # gate negatives: a different task entirely, so the gate has something to say no to
    ds = load_dataset("fancyzhx/ag_news", split="train[:%d]" % args.n_neg)
    texts = list(ds["text"])
    idx = list(range(rank, len(texts), world))
    F_, O_, M_ = [], [], []
    for s in range(0, len(idx), args.batch):
        chunk = idx[s:s + args.batch]
        items, k = build_items(v, [texts[i] for i in chunk], NEG_QUESTION)
        f, o, m = pool_batch(v, items, k)
        F_.append(f)
        O_.append(o)
        M_.append(m)
    if not idx:
        log("negatives rank %d: empty shard, no cache written" % rank)
        return
    np.savez(os.path.join(CACHE, "neg_rank%d.npz" % rank),
             feats=torch.cat(F_).numpy(), opts=torch.cat(O_).numpy(), mask=torch.cat(M_).numpy(),
             y=np.zeros(len(idx), dtype=np.int64))
    log("negatives rank %d: %d rows" % (rank, len(idx)))
    log("rank %d done" % rank)


# --------------------------------------------------------------------------- stage 2: the adapter
def load_split(name):
    import glob

    parts = sorted(glob.glob(os.path.join(CACHE, "%s_rank*.npz" % name)))
    if not parts:
        raise SystemExit("no cached features for %r -- run the extract stage first" % name)
    d = [np.load(p) for p in parts]
    return (torch.from_numpy(np.concatenate([x["feats"] for x in d])),
            torch.from_numpy(np.concatenate([x["opts"] for x in d])),
            torch.from_numpy(np.concatenate([x["mask"] for x in d])),
            torch.from_numpy(np.concatenate([x["y"] for x in d])))


def engine_logits(engine, feats, opts, mask, qtype, n_steps=None):
    out = engine(feats, opts, mask, qtype, n_steps=n_steps, halt=False, return_traj=False)
    return out, out["logits"].masked_fill(~mask, -1e4)


def stage_train(args):
    device = pick_device()
    from safetensors.torch import save_file

    import vegaml
    from vegaml._api import Vega as Engine
    from vegaml._common import (
        QTYPES,
        adapter_state,
        attach_adapter,
        conformal_thresholds,
        fit_thermal,
        softmax_np,
        thermal_features,
        thermal_tau,
    )

    path = vegaml.fetch(MODEL_REPO, token=HF_TOKEN)
    eng = Engine(path, backbone=BACKBONE, device=device, adapters="off")
    engine = eng.engine
    base_sha = eng.base_sha256()

    j = attach_adapter(engine, rank=args.rank, alpha=float(args.rank))
    engine.adapter_mode = "on"
    for p in engine.parameters():
        p.requires_grad_(False)
    for path_ in ("world.net.1", "probe.net.1", "impulse.net.1", "to_z0.net.1", "to_p0.net.1",
                  "well.net.1", "well.net.4"):
        mod = engine
        for part in path_.split(".")[:-1]:
            mod = getattr(mod, part) if not part.isdigit() else mod[int(part)]
        lin = mod[int(path_.split(".")[-1])]
        lin.lora_A[j].requires_grad_(True)
        lin.lora_B[j].requires_grad_(True)
    for p in engine.gate_nets[j].parameters():
        p.requires_grad_(True)
    trainable = [p for p in engine.parameters() if p.requires_grad]
    n_par = sum(p.numel() for p in trainable)
    log("training %d parameters across %d tensors" % (n_par, len(trainable)))

    tr = load_split("train")
    va = load_split("valid")
    neg = load_split("neg")
    log("train %d, valid %d, negatives %d" % (len(tr[3]), len(va[3]), len(neg[3])))

    qt_pos = torch.full((args.batch,), QTYPES["choice"], device=device)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    n = len(tr[3])
    if min(n, len(va[3]), len(neg[3])) == 0 or n < args.batch:
        raise SystemExit("training needs nonempty train/valid/negative splits and at least one full batch")
    steps = (n // args.batch) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    engine.train()
    step = 0
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        negperm = torch.randperm(len(neg[3]))
        run, hits, seen = 0.0, 0, 0
        for s in range(0, n - args.batch + 1, args.batch):
            b = perm[s:s + args.batch]
            feats, opts, mask, y = (tr[0][b].to(device), tr[1][b].to(device),
                                    tr[2][b].to(device), tr[3][b].to(device))
            _, logits = engine_logits(engine, feats, opts, mask, qt_pos[:len(b)])
            loss = F.cross_entropy(logits, y)

            # the gate: open on this task, shut on a task it was not trained for
            off = (s // args.batch * args.gate_batch) % max(len(neg[3]) - args.gate_batch, 1)
            nb = negperm[off:off + args.gate_batch]
            if len(nb) > 1:
                nf, no, nm = neg[0][nb].to(device), neg[1][nb].to(device), neg[2][nb].to(device)
                qt_neg = torch.full((len(nb),), QTYPES["choice"], device=device)
                out_pos, _ = engine_logits(engine, feats, opts, mask, qt_pos[:len(b)])
                out_neg, _ = engine_logits(engine, nf, no, nm, qt_neg)
                g_pos = out_pos["gates"][:, j]
                g_neg = out_neg["gates"][:, j]
                gate_loss = (F.binary_cross_entropy(g_pos.clamp(1e-6, 1 - 1e-6),
                                                    torch.ones_like(g_pos)) +
                             F.binary_cross_entropy(g_neg.clamp(1e-6, 1 - 1e-6),
                                                    torch.zeros_like(g_neg)))
                loss = loss + args.gate_weight * gate_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            if step < steps:
                sched.step()
            step += 1
            run += float(loss.detach()) * len(b)
            hits += int((logits.argmax(-1) == y).sum())
            seen += len(b)
        log("epoch %d: loss %.4f  train acc %.4f" % (ep + 1, run / seen, hits / seen))

    # ---- calibration on the validation split, which training never saw
    engine.eval()
    def collect(split):
        feats, opts, mask, y = split
        L, P, G = [], [], []
        with torch.no_grad():
            for s in range(0, len(y), args.batch):
                f = feats[s:s + args.batch].to(device)
                o = opts[s:s + args.batch].to(device)
                m = mask[s:s + args.batch].to(device)
                qt = torch.full((len(f),), QTYPES["choice"], device=device)
                out, logits = engine_logits(engine, f, o, m, qt)
                L.append(logits.cpu().numpy())
                P.append(thermal_features(out, m, engine.max_steps).cpu().numpy())
                G.append(out["gates"][:, j].cpu().numpy())
        return np.concatenate(L), np.concatenate(P), np.concatenate(G), y.numpy()

    Lv, Pv, Gv, yv = collect(va)
    qt_v = np.zeros(len(yv), dtype=np.int64)
    # fit_thermal minimises a cross-entropy against a target DISTRIBUTION, not label indices.
    onehot = np.zeros_like(Lv, dtype=np.float64)
    onehot[np.arange(len(yv)), yv] = 1.0
    theta = fit_thermal(Lv, onehot, qt_v, Pv)
    probs = softmax_np(Lv, thermal_tau(theta, qt_v, Pv))
    conf = conformal_thresholds(probs, yv, qt_v)
    acc = float((probs.argmax(1) == yv).mean())
    log("validation: acc %.4f  mean gate %.3f  open %.2f%%"
        % (acc, float(Gv.mean()), 100 * float((Gv >= 0.5).mean())))

    os.makedirs(OUT, exist_ok=True)
    name = "webchain"
    save_file(adapter_state(engine, j), os.path.join(OUT, name + ".safetensors"))
    meta = {
        "name": name, "rank": args.rank, "alpha": float(args.rank),
        "targets": ["world.net.1", "probe.net.1", "impulse.net.1", "to_z0.net.1", "to_p0.net.1",
                    "well.net.1", "well.net.4"],
        "mode": "auto", "gate_trained": True, "gate_threshold": 0.5,
        "calibration": {"mode": "thermal", "theta": theta.tolist(), "conformal": conf},
        "trained_on": ("WebChain click train tasks; website-md5 test holdout; "
                       "task-md5 validation; negatives: fancyzhx/ag_news"),
        "base_engine_sha256": base_sha,
        "adapter_params": int(n_par),
        "valid_accuracy": acc,
        "gate_stats": {"mean_gate": float(Gv.mean()), "open_frac": float((Gv >= 0.5).mean())},
    }
    with open(os.path.join(OUT, name + ".json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    log("wrote %s/%s.safetensors (+ .json)" % (OUT, name))


# --------------------------------------------------------------------------- stage 3: evaluate
def stage_eval(args):
    device = pick_device()
    from safetensors.torch import load_file

    import vegaml
    from vegaml._api import Vega as Engine
    from vegaml._common import QTYPES, ece_score, load_adapter, softmax_np, thermal_features, thermal_tau

    path = vegaml.fetch(MODEL_REPO, token=HF_TOKEN)
    te = load_split("test")
    y = te[3].numpy()

    def run(engine, theta, j=None):
        L, P, G = [], [], []
        with torch.no_grad():
            for s in range(0, len(y), args.batch):
                f = te[0][s:s + args.batch].to(device)
                o = te[1][s:s + args.batch].to(device)
                m = te[2][s:s + args.batch].to(device)
                qt = torch.full((len(f),), QTYPES["choice"], device=device)
                out = engine(f, o, m, qt, halt=False, return_traj=False)
                L.append(out["logits"].masked_fill(~m, -1e4).cpu().numpy())
                P.append(thermal_features(out, m, engine.max_steps).cpu().numpy())
                if j is not None and out.get("gates") is not None:
                    G.append(out["gates"][:, j].cpu().numpy())
        L, P = np.concatenate(L), np.concatenate(P)
        qt = np.zeros(len(y), dtype=np.int64)
        probs = softmax_np(L, thermal_tau(np.asarray(theta), qt, P))
        pred = probs.argmax(1)
        return (float((pred == y).mean()),
                100 * ece_score(probs.max(1), (pred == y).astype(float)),
                float(np.concatenate(G).mean()) if G else None)

    base = Engine(path, backbone=BACKBONE, device=device, adapters="off")
    acc_b, ece_b, _ = run(base.engine, base.theta)
    log("base engine      acc %.4f   ECE %.2f" % (acc_b, ece_b))

    tuned = Engine(path, backbone=BACKBONE, device=device, adapters="off")
    meta = json.load(open(os.path.join(OUT, "webchain.json")))
    j = load_adapter(tuned.engine, load_file(os.path.join(OUT, "webchain.safetensors")),
                     meta["rank"], meta["alpha"])
    tuned.engine.adapter_mode = "on"
    tuned.engine.eval()
    acc_a, ece_a, gate = run(tuned.engine, meta["calibration"]["theta"], j=j)
    log("+ webchain adapter acc %.4f   ECE %.2f   mean gate %.3f" % (acc_a, ece_a, gate))
    log("delta: accuracy %+.4f, ECE %+.2f" % (acc_a - acc_b, ece_a - ece_b))

    with open(os.path.join(WORK, "webchain_results.json"), "w") as fh:
        json.dump({"n_test": int(len(y)),
                   "base": {"accuracy": acc_b, "ece": ece_b},
                   "adapter": {"accuracy": acc_a, "ece": ece_a, "mean_gate": gate}}, fh, indent=1)
    log("wrote %s/webchain_results.json" % WORK)


def push_adapter(repo, private=True):
    """Upload the trained adapter to a Hub repository of your own.

    Only the adapter and its metadata go up: a few hundred kilobytes, no backbone and no engine.
    Needs HF_TOKEN. Everything else in this script runs without one.
    """
    if not HF_TOKEN:
        log("skipping the push: no HF_TOKEN in the environment")
        return
    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(repo_id=repo, repo_type="model", folder_path=OUT, path_in_repo="adapters",
                      allow_patterns=["*.safetensors", "*.json"])
    results = os.path.join(WORK, "webchain_results.json")
    if os.path.exists(results):
        api.upload_file(path_or_fileobj=results, path_in_repo="webchain_results.json",
                        repo_id=repo, repo_type="model")
    log("pushed the adapter to https://huggingface.co/%s (private=%s)" % (repo, private))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "train", "eval"])
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--limit", type=int, help="read only the first N source rows before filtering clicks")
    ap.add_argument("--stats-only", action="store_true", help="convert and report without loading a model")
    ap.add_argument("--smoke", action="store_true", help="build tokenizer prompts without model inference")
    ap.add_argument("--n-neg", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--gate-batch", type=int, default=16)
    ap.add_argument("--gate-weight", type=float, default=0.5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--push-to", default=os.environ.get("VEGA_PUSH_TO") or None,
                    help="after eval, upload the adapter to this Hub repo id (needs HF_TOKEN)")
    ap.add_argument("--public", action="store_true",
                    help="create the pushed repo public; it is private by default")
    args = ap.parse_args()
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be positive")
    if args.batch < 1 or args.n_neg < 2 or args.gate_batch < 2 or args.epochs < 1:
        ap.error("batch/epochs must be positive and negative/gate batches must have at least two rows")
    if (args.smoke or args.stats_only) and args.stage != "extract":
        ap.error("--smoke and --stats-only apply only to extract")
    {"extract": stage_extract, "train": stage_train, "eval": stage_eval}[args.stage](args)
    if args.stage == "eval" and args.push_to:
        push_adapter(args.push_to, private=not args.public)


if __name__ == "__main__":
    sys.exit(main())
