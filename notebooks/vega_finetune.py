"""Fine-tune a gated adapter for Vega on a public dataset, on two T4s.

Three stages, run in order:

  torchrun --nproc_per_node=2 vega_finetune.py extract   # both GPUs: frozen features, cached to disk
  python vega_finetune.py train                          # one GPU: the adapter, on cached features
  python vega_finetune.py eval                           # base engine vs the new adapter

Why it splits that way. The language model is frozen and is the only expensive part: reading ~24k
short texts is a few GPU-minutes, and it happens exactly once. What is actually trained -- a rank-32
adapter on seven engine projections plus a gate, about 0.8M parameters -- then runs on cached
pooled features, where an epoch is seconds on one GPU. Sharding the read across both T4s is the only
place a second GPU helps, so that is the only place it is used; pretending otherwise would make the
notebook slower, not faster.

Which size: VEGA_SIZE picks the backbone, "0.8b" (the default) or "4b". Both train identically;
the 4B reads more slowly and its cached features are wider, so each size keeps its own cache and
its own output directory.

A Hugging Face token is optional throughout. The checkpoint and both datasets are public, so the
three stages above run with nothing set. Export HF_TOKEN to train against a private checkpoint, and
pass --push-to <your-repo> on the eval stage to upload the adapter you trained to your own account.

The task: dair-ai/emotion, a 6-way choice, written as a typed decision. The gate is trained with
negatives from ag_news so it learns when NOT to fire, which is what makes an adapter routable
rather than always-on.
"""
from __future__ import annotations

import argparse
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
CACHE = os.path.join(WORK, "features-" + SIZE)
OUT = os.path.join(WORK, "adapter-" + SIZE)

EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
QUESTION = {
    "type": "choice",
    "instructions": "Which emotion does this message express?",
    "criteria": {
        "sadness": "low mood, loss, loneliness or regret",
        "joy": "happiness, delight or contentment",
        "love": "affection, warmth or devotion toward someone",
        "anger": "irritation, resentment or rage",
        "fear": "anxiety, worry or dread",
        "surprise": "astonishment at something unexpected",
    },
}
# The dataset's integer labels index QUESTION["criteria"] positionally (see stage_extract), so the
# two orders must agree or every example trains against the wrong label. A mismatch here produces an
# adapter that scores near chance and reports itself as a success, so it is checked once, loudly.
assert list(QUESTION["criteria"]) == EMOTIONS, \
    "the dataset's label order is %s but the question's criteria are %s" % (
        EMOTIONS, list(QUESTION["criteria"]))
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
def build_items(vega, rows, question):
    """One prompt per row, in the engine's own format, with the option order left identity."""
    from vegaml._common import build_prompt, option_texts

    qi = vega.engine_api._internal(question)
    k = len(option_texts(qi))
    items = []
    for text in rows:
        ids, spans = build_prompt(vega.perc.tok, text, qi, vega.cfg["max_len"],
                                  option_order=list(range(k)),
                                  opt_max_tokens=vega.cfg["opt_max_tokens"],
                                  query_first_max_state=vega.cfg.get("query_first_max_state", 0))
        items.append({"ids": np.asarray(ids, dtype=np.int32), "spans": spans})
    return items, k


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

    from datasets import load_dataset

    import vegaml
    from vegaml._api import Vega as Engine

    path = vegaml.fetch(MODEL_REPO, token=HF_TOKEN)
    eng = Engine(path, backbone=BACKBONE, device=device, adapters="off")

    class Holder:                      # the attributes build_items needs, without a wrapper class
        pass

    v = Holder()
    v.perc, v.cfg, v.engine_api = eng.perc, eng.cfg, eng

    os.makedirs(CACHE, exist_ok=True)
    plan = [("train", "train", args.n_train), ("valid", "validation", args.n_valid),
            ("test", "test", args.n_test)]
    for name, split, n in plan:
        ds = load_dataset("dair-ai/emotion", split="%s[:%d]" % (split, n))
        texts, labels = list(ds["text"]), list(ds["label"])
        shard = range(rank, len(texts), world)
        idx = list(shard)
        log("%s: %d of %d rows on this rank" % (name, len(idx), len(texts)))
        F_, O_, M_, Y_ = [], [], [], []
        t0 = time.time()
        for s in range(0, len(idx), args.batch):
            chunk = idx[s:s + args.batch]
            items, k = build_items(v, [texts[i] for i in chunk], QUESTION)
            f, o, m = pool_batch(v, items, k)
            F_.append(f)
            O_.append(o)
            M_.append(m)
            Y_ += [labels[i] for i in chunk]
            if s % (args.batch * 20) == 0:
                done = s + len(chunk)
                log("  %s %d/%d  %.0f items/s" % (name, done, len(idx), done / max(time.time() - t0, 1e-9)))
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
    # np.load on an .npz returns an NpzFile that holds the zip open until it is closed. Reading them
    # without closing leaves every shard locked for the life of the process, which on Windows means
    # the cache directory cannot then be deleted.
    shards = []
    for p in parts:
        with np.load(p) as z:
            shards.append({k: z[k] for k in z.files})
    return (torch.from_numpy(np.concatenate([x["feats"] for x in shards])),
            torch.from_numpy(np.concatenate([x["opts"] for x in shards])),
            torch.from_numpy(np.concatenate([x["mask"] for x in shards])),
            torch.from_numpy(np.concatenate([x["y"] for x in shards])))


def engine_logits(engine, feats, opts, mask, qtype, n_steps=None):
    out = engine(feats, opts, mask, qtype, n_steps=n_steps, halt=False, return_traj=False)
    return out, out["logits"].masked_fill(~mask, -1e4)


def stage_train(args):
    device = pick_device()
    from safetensors.torch import save_file

    import vegaml
    from vegaml._api import Vega as Engine
    from vegaml._common import (
        ADAPTER_TARGETS,
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
    # "auto", NOT "on". In "on" the engine returns a constant gate of 1 without ever calling
    # gate_nets, so the gate loss below has no autograd graph, the gate network stays at its random
    # initialisation, and the gate_trained: True written at the end would be a lie. Inference
    # defaults to "auto", which is sigmoid(gate_net(x)) -- so training has to run "auto" for the
    # gate it trains to be the gate that runs.
    engine.adapter_mode = "auto"
    for p in engine.parameters():
        p.requires_grad_(False)
    # The canonical list, the same one attach_adapter, adapter_state and load_adapter use. Typing
    # the seven paths out a second time meant the script could train one set of projections while
    # the metadata described another, if the package's list ever changed.
    for path_ in ADAPTER_TARGETS:
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
    n = len(tr[3])
    log("train %d, valid %d, negatives %d" % (n, len(va[3]), len(neg[3])))
    if n < args.batch:
        # The loop below takes whole batches, so a split smaller than one batch runs zero iterations
        # and then divides by a zero `seen` on the log line. Reachable from the documented smoke
        # test: a 96-row slice across 2 ranks is 48 rows per rank, and stage 2 is run with --batch 64.
        raise SystemExit("--batch %d is larger than the %d training rows; lower --batch or raise "
                         "--n-train" % (args.batch, n))

    qt_pos = torch.full((args.batch,), QTYPES["choice"], device=device)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    # Iterations actually run per epoch is floor((n - batch) / batch) + 1, which equals n // batch
    # only when batch divides n. Scheduling against n // batch cut the anneal tail off whenever it
    # did not, so the final learning rate was not the scheduled minimum.
    iters = max(1, (n - args.batch) // args.batch + 1)
    steps = iters * args.epochs
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
            # out_pos is reused by the gate step below rather than recomputed with identical
            # arguments, which was doubling the cost of the positive forward pass every step.
            out_pos, logits = engine_logits(engine, feats, opts, mask, qt_pos[:len(b)])
            loss = F.cross_entropy(logits, y)

            # the gate: open on this task, shut on a task it was not trained for
            off = (s // args.batch * args.gate_batch) % max(len(neg[3]) - args.gate_batch, 1)
            nb = negperm[off:off + args.gate_batch]
            if len(nb) > 1:
                nf, no, nm = neg[0][nb].to(device), neg[1][nb].to(device), neg[2][nb].to(device)
                qt_neg = torch.full((len(nb),), QTYPES["choice"], device=device)
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
    # fit_thermal and conformal_thresholds both need 30 examples per question type and skip below
    # that, silently: theta stays all zeros (tau = 1, so the probabilities are raw and
    # uncalibrated) and the conformal table comes back empty. The adapter would then be written with
    # gate_trained: True and prediction_set: None at inference, and the run looks like a success.
    if len(yv) < 30:
        raise SystemExit("validation split has %d rows; thermal calibration and conformal thresholds "
                         "need at least 30. Raise --n-valid." % len(yv))
    if not conf.get("choice"):
        raise SystemExit("no conformal thresholds were fitted for choice questions, so this adapter "
                         "could not produce a prediction set. The validation split is too small.")
    if not np.any(theta):
        raise SystemExit("thermal calibration returned an all-zero theta, which means no question type "
                         "had enough validation rows to fit a temperature")

    os.makedirs(OUT, exist_ok=True)
    name = "emotion"
    save_file(adapter_state(engine, j), os.path.join(OUT, name + ".safetensors"))
    meta = {
        "name": name, "rank": args.rank, "alpha": float(args.rank),
        "targets": list(ADAPTER_TARGETS),
        "mode": "auto", "gate_trained": True, "gate_threshold": 0.5,
        "calibration": {"mode": "thermal", "theta": theta.tolist(), "conformal": conf},
        "trained_on": "dair-ai/emotion train split; gate negatives: fancyzhx/ag_news",
        "base_engine_sha256": base_sha,
        "adapter_params": int(n_par),
        "valid_accuracy": acc,
        "valid_examples": int(len(yv)),
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
    with open(os.path.join(OUT, "emotion.json")) as fh:
        meta = json.load(fh)
    j = load_adapter(tuned.engine, load_file(os.path.join(OUT, "emotion.safetensors")),
                     meta["rank"], meta["alpha"])
    tuned.engine.adapter_mode = "auto"   # not "on": "on" forces every gate to 1 without calling the
    # gate network, so the reported mean gate would be a constant 1.000 and nothing about the
    # trained gate would be measured. This is also the mode the loader uses by default.
    tuned.engine.eval()
    acc_a, ece_a, gate = run(tuned.engine, meta["calibration"]["theta"], j=j)
    log("+ emotion adapter acc %.4f   ECE %.2f   mean gate %.3f" % (acc_a, ece_a, gate))
    log("delta: accuracy %+.4f, ECE %+.2f" % (acc_a - acc_b, ece_a - ece_b))

    with open(os.path.join(WORK, "finetune_results.json"), "w") as fh:
        json.dump({"n_test": int(len(y)),
                   "base": {"accuracy": acc_b, "ece": ece_b},
                   "adapter": {"accuracy": acc_a, "ece": ece_a, "mean_gate": gate}}, fh, indent=1)
    log("wrote %s/finetune_results.json" % WORK)


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
    results = os.path.join(WORK, "finetune_results.json")
    if os.path.exists(results):
        api.upload_file(path_or_fileobj=results, path_in_repo="finetune_results.json",
                        repo_id=repo, repo_type="model")
    log("pushed the adapter to https://huggingface.co/%s (private=%s)" % (repo, private))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "train", "eval"])
    ap.add_argument("--n-train", type=int, default=16000)
    ap.add_argument("--n-valid", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=2000)
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
    {"extract": stage_extract, "train": stage_train, "eval": stage_eval}[args.stage](args)
    if args.stage == "eval" and args.push_to:
        push_adapter(args.push_to, private=not args.public)


if __name__ == "__main__":
    sys.exit(main())
