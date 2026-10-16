"""Vega shared code: config, perception prompts, the latent decision physics engine, thermal calibration, metrics.

Vega follows the WALT world-model recipe (frozen language model = perception, small trainable physics = dynamics):

  text -> [frozen causal LM, one forward pass] -> pooled spans -> [projectors] -> particle + potential wells
       -> [damped Hamiltonian integrator with a selective-SSM thermostat] -> basin the particle settles in -> decision

* The observed state is a point in a world latent s (WALT's physics latent). A one-step world dynamics block
  imagines the next state s' (WALT's z_t -> z_{t+1}), trained on conversation prefixes.
* A question is a measurement: it places a particle at z0 with an initial momentum p0 in a 64-d decision space.
* Every answer option is a Gaussian potential well (centre, depth, width). Score levels sit on a 1-d rail, so
  ordinal neighbours are physical neighbours.
* The particle moves under the wells' forces, friction and a learned memory force for a few steps (early exit when
  it has settled). The Boltzmann occupancy of the wells at the final position is the answer distribution.
* Calibration is thermodynamic: the temperature of that Boltzmann distribution depends on how settled the particle
  is (residual kinetic energy, distance to the nearest well, steps used), fitted on a held-out split.
  Split-conformal prediction sets and an "unbound particle" out-of-distribution flag come from the same energies.

Kept Python 3.9 compatible so the same file runs on Kaggle and on a laptop smoke test.
"""
# Compatibility fix for Pillow 11.x / torchvision in container images
try:
    import PIL._typing, typing
    if not hasattr(PIL._typing, "_Ink"):
        PIL._typing._Ink = typing.Union[tuple[int, ...], str, int]  # type: ignore
except Exception:
    pass

import hashlib
import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

QTYPES = {"choice": 0, "score": 1, "boolean": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}
SLOTS = ("state", "question", "last")   # pooled item vectors per feature layer; options are stored separately


# ----------------------------------------------------------------------------- config
def load_cfg(path: Optional[str] = None) -> Dict:
    path = path or os.environ.get("VEGA_CFG", "vega_config.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------------- text rendering
def serialize_state(state) -> str:
    """The situation as one string. Text passes through; anything else becomes compact JSON."""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def prose_state(state) -> str:
    """Second surface form of a structured state (used for the invariance view): 'Key: value' lines."""
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        lines = []
        for k, v in state.items():
            val = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
            lines.append("%s: %s" % (str(k).replace("_", " ").capitalize(), val))
        return "\n".join(lines)
    return json.dumps(state, ensure_ascii=False)


def _crit_text(v) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def normalise_conformal(table: Dict) -> Dict:
    """Key a checkpoint's conformal table by the question-type names this package uses.

    The table is stored keyed by type name, so a checkpoint written before a type was renamed
    carries the old name and the lookup would silently miss, costing the prediction set without
    raising. When exactly one stored key is unrecognised and exactly one known type is absent,
    the mapping is unambiguous, so it is applied. Anything less clear-cut keeps the recognised
    keys and drops the rest, which is what a lookup would have done anyway, rather than guessing.
    """
    if not isinstance(table, dict):
        return {}
    out = {k: v for k, v in table.items() if k in QTYPES}
    unknown = [k for k in table if k not in QTYPES]
    absent = [k for k in QTYPES if k not in out]
    if len(unknown) == 1 and len(absent) == 1:
        out[absent[0]] = table[unknown[0]]
    return out


def option_texts(q: Dict) -> List[str]:
    """Option texts in label-index order (boolean is always [no, yes], so p[1] is the probability of yes)."""
    t, crit = q["t"], q.get("crit")
    if t == "choice":
        return [str(k) if v in (None, "") else "%s - %s" % (k, _crit_text(v)) for k, v in crit.items()]
    if t == "score":
        return ["level %d - %s" % (i, _crit_text(c)) for i, c in enumerate(crit)]
    crit = crit or {}
    no = crit.get("false")
    yes = crit.get("true")
    return ["no - " + (_crit_text(no) if no not in (None, "") else "the statement does not hold"),
            "yes - " + (_crit_text(yes) if yes not in (None, "") else "the statement holds")]


def answer_space_hash(q: Dict) -> int:
    """Same answer space (type + option texts) -> same hash. Used to pair items for the counterfactual loss."""
    key = q["t"] + "\x1f" + "\x1f".join(option_texts(q))
    return int(hashlib.md5(key.encode("utf-8", "ignore")).hexdigest()[:15], 16)


# Two prompt templates: template 0 is used at inference, template 1 only as the second (invariance) view.
TEMPLATES = [
    {"head": "Observation:\n", "q": "\n\nMeasurement ({t}): {ins}\nPossible outcomes:", "opt": "\n* {o}", "end": "\nOutcome:"},
    {"head": "Here is the situation.\n", "q": "\n\n[{t} decision] {ins}\nChoices:", "opt": "\n- {o}", "end": "\nDecision:"},
]


QUERY_FIRST = "Question to keep in mind while reading ({t}): {ins}\n\n"


def build_prompt(tok, state_text: str, q: Dict, max_len: int, option_order: Optional[List[int]] = None,
                 truncate_left: bool = False, template: int = 0, opt_max_tokens: int = 32, ins_max_tokens: int = 256,
                 query_first_max_state: int = 0):
    """Tokenize one decision prompt for the frozen perception model and record where each part sits.

    Returns (ids, spans) with spans = {"state": (a, b), "question": (a, b), "options": [(a, b), ...]} in token positions
    (options listed in `option_order`). By default the state comes first so its span never sees the question (causal
    attention): the state-span mean is a question-independent world observation, while the question, option and final
    tokens have read the whole state.

    query_first_max_state > 0 (run 8+): a state shorter than that many tokens is preceded by the question as well, so the
    frozen model reads the record knowing what is asked (query-aware reading; helps multi-part questions). Longer
    states keep the question-independent prefix, which all questions about the record share (read once), and are
    read by the question through the engine's memory reader instead.
    """
    T = TEMPLATES[template]
    opts = option_texts(q)
    order = option_order if option_order is not None else list(range(len(opts)))
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    bos = [tok.bos_token_id] if getattr(tok, "bos_token_id", None) is not None else []
    head = enc(T["head"])
    qids = enc(T["q"].format(t=q["t"], ins=str(q["ins"])))[:ins_max_tokens]
    oids = [enc(T["opt"].format(o=opts[i]))[:opt_max_tokens] for i in order]
    end = enc(T["end"])
    fixed = len(bos) + len(head) + len(qids) + sum(len(o) for o in oids) + len(end)
    if fixed > max_len - 16:  # too many / too long options: shrink every option evenly, then the instructions
        per = max(3, (max_len - 16 - len(bos) - len(head) - len(end) - min(len(qids), 64)) // max(1, len(oids)))
        oids = [o[:per] for o in oids]
        qids = qids[:max(8, max_len - 16 - len(bos) - len(head) - len(end) - sum(len(o) for o in oids))]
        fixed = len(bos) + len(head) + len(qids) + sum(len(o) for o in oids) + len(end)
    st = enc(state_text)
    pre = []
    if query_first_max_state and len(st) < query_first_max_state:
        pre = enc(QUERY_FIRST.format(t=q["t"], ins=str(q["ins"])))[:min(128, ins_max_tokens)]
        fixed += len(pre)
    room = max(0, max_len - fixed)
    st = st[max(0, len(st) - room):] if truncate_left else st[:room]
    ids = bos + pre + head
    s0 = len(ids)
    ids = ids + st
    # an empty state still needs a span: fall back to the header tokens
    spans = {"state": (s0, len(ids)) if st else (len(bos), s0)}
    q0 = len(ids)
    ids = ids + qids
    spans["question"] = (q0, len(ids))
    spans["options"] = []
    for o in oids:
        a = len(ids)
        ids = ids + o
        spans["options"].append((a, len(ids)))
    ids = ids + end
    return ids, spans


def episode_prefix_lengths(n_turns: int, max_prefixes: int) -> List[int]:
    if n_turns <= max_prefixes:
        return list(range(1, n_turns + 1))
    return sorted(set(int(round(x)) for x in np.linspace(1, n_turns, max_prefixes)))


def record_items(rec: Dict, rng: Optional[random.Random], cfg: Dict, train: bool, view_frac: float) -> List[Dict]:
    """One stored record -> perception items (one per question or conversation prefix, plus optional second views).

    Each item: state_text, q, order (option permutation), template, target (soft, original option order), label
    (original index), truncate_left, ep_key/ep_step for episodes, view (0 or 1) and pair_key linking the two views.
    """
    items = []
    if rec.get("no_views"):   # long records: a second surface form would double the most expensive reads
        view_frac = 0.0
    if rec.get("kind") == "episode":
        ep, q = rec["ep"], rec["qs"][0]
        lens = episode_prefix_lengths(len(ep["turns"]), cfg["max_prefixes"])
        for step, t in enumerate(lens):
            state = dict(ep["ctx"], conversation=ep["turns"][:t])
            items.append({"state_text": serialize_state(state), "q": q, "order": [0, 1], "template": 0,
                          "target": [1.0 - ep["y"], float(ep["y"])], "label": int(ep["y"]), "truncate_left": True,
                          "ep_step": step, "episode": 1, "view": 0, "pair_key": None})
        return items
    for qi, q in enumerate(rec["qs"]):
        k = len(option_texts(q))
        if k < 2 or q.get("y") is None and not q.get("soft"):
            continue
        target = list(q["soft"]) if q.get("soft") else [1.0 if i == q["y"] else 0.0 for i in range(k)]
        label = int(np.argmax(target)) if q.get("y") is None else int(q["y"])
        order = list(range(k))
        if train and rng is not None:
            kmax = cfg["max_options_train"]
            if q["t"] == "choice" and k > kmax:  # keep the gold option, subsample distractors
                keep = [label] + rng.sample([i for i in range(k) if i != label], kmax - 1)
                order = sorted(keep)
            if q["t"] != "score":
                rng.shuffle(order)
        base = {"q": q, "target": target, "label": label, "truncate_left": False, "ep_step": 0, "episode": 0}
        items.append(dict(base, state_text=serialize_state(rec["state"]), order=order, template=0, view=0,
                          pair_key=(qi if (rng is not None and rng.random() < view_frac) else None)))
        if items[-1]["pair_key"] is not None:
            order2 = list(order)
            if q["t"] != "score":
                for _ in range(3):
                    rng.shuffle(order2)
                    if order2 != order:
                        break
            items.append(dict(base, state_text=prose_state(rec["state"]) if not isinstance(rec["state"], str)
                              else rec["state"], order=order2, template=1, view=1, pair_key=qi))
    return items


# ----------------------------------------------------------------------------- perception helpers
class StopForward(Exception):
    pass


# Long prompts on GPUs without a fused kernel for Qwen3.5's 256-wide attention heads (compute capability < 8, e.g.
# the T4) fall back to PyTorch's reference attention, which builds the full L x L score matrix: ~50 GB at 40k tokens
# (89 of 150 long-context cases ran out of memory on a T4). Computing causal attention one block of queries at a time
# gives the same result in O(block x L) memory. Used only for long single-sequence reads without a padding mask.
CHUNKED_ATTN_MIN_TOKENS = 2048
CHUNKED_ATTN_BLOCK = 512


class chunked_attention:
    """Context manager: route causal, unmasked scaled_dot_product_attention calls with q_len == kv_len > 2 x block
    through query blocks. Everything else goes to the original function unchanged."""

    def __init__(self, enabled=True, block=None):
        self.enabled, self.block = enabled, block or CHUNKED_ATTN_BLOCK   # read at call time (tests change it)

    def __enter__(self):
        if self.enabled:
            self.orig = F.scaled_dot_product_attention
            orig, block = self.orig, self.block

            def sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
                L, S = query.size(-2), key.size(-2)
                if not (is_causal and attn_mask is None and L == S and L > 2 * block):
                    return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal,
                                scale=scale, **kw)
                out = []
                for s in range(0, L, block):
                    e = min(L, s + block)
                    m = torch.ones(e - s, e, dtype=torch.bool, device=query.device).tril(diagonal=s)
                    out.append(orig(query[..., s:e, :], key[..., :e, :], value[..., :e, :], attn_mask=m,
                                    dropout_p=0.0, scale=scale, **kw))
                return torch.cat(out, -2)
            F.scaled_dot_product_attention = sdpa
            torch.nn.functional.scaled_dot_product_attention = sdpa
        return self

    def __exit__(self, *exc):
        if self.enabled:
            F.scaled_dot_product_attention = self.orig
            torch.nn.functional.scaled_dot_product_attention = self.orig
        return False


def find_layers(model):
    """The decoder's block list (works for Gemma 3, Qwen 3, Llama-style and nested multimodal wrappers)."""
    best = None
    for name, mod in model.named_modules():
        if name.endswith("layers") and isinstance(mod, torch.nn.ModuleList):
            if best is None or len(mod) > len(best):
                best = mod
    if best is None:
        raise RuntimeError("could not find the decoder layer list")
    return best


def resolve_layers(spec, n_layers):
    """Layer spec: fractions of depth (0 < f <= 1) or explicit 0-based indices."""
    out = []
    for x in spec:
        i = int(round(x * n_layers)) - 1 if isinstance(x, float) and x <= 1.0 else int(x)
        out.append(max(0, min(n_layers - 1, i)))
    return sorted(set(out))


def resolve_device(device=None):
    """Explicit devices win; otherwise prefer CUDA, then Apple MPS, then CPU."""
    if device:
        dev = torch.device(device)
        if dev.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available in this PyTorch build or on this machine; "
                               "use device=\"cpu\" instead")
        return dev
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def backbone_dtype(device):
    """Keep CPU/MPS in float32; preserve the CUDA capability-based precision policy."""
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    return torch.float32


# ----------------------------------------------------------------------------- frozen perception
class Perception:
    """Frozen causal LM, cut off after the deepest layer we read. Returns pooled span features, never trained."""

    def __init__(self, path, device, feature_layers, option_layers):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        dtype = backbone_dtype(device)
        import transformers
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        # `dtype=` replaced `torch_dtype=` in 4.56; older versions would silently ignore it and load fp32
        kw = {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}
        self.model = AutoModelForCausalLM.from_pretrained(path, attn_implementation="sdpa", **kw)
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.layers = find_layers(self.model)
        self.n_layers = len(self.layers)
        cfgm = self.model.config
        self.d_model = getattr(cfgm, "hidden_size", None) or getattr(getattr(cfgm, "text_config", cfgm), "hidden_size")
        self.feature_layers = resolve_layers(feature_layers, self.n_layers)
        self.option_layers = resolve_layers(option_layers, self.n_layers)
        self.capture = sorted(set(self.feature_layers) | set(self.option_layers))
        self._cap = {}
        last = max(self.capture)
        for i in self.capture:
            def hook(mod, inp, out, i=i):
                self._cap[i] = out[0] if isinstance(out, (tuple, list)) else out
                if i == last:
                    raise StopForward()
            self.layers[i].register_forward_hook(hook)

    def _needs_chunked_attention(self, n_tokens):
        if n_tokens < CHUNKED_ATTN_MIN_TOKENS:
            return False
        if os.environ.get("VEGA_CHUNKED_ATTENTION") == "1":   # force (tests, other GPUs)
            return True
        return self.device.type == "cuda" and torch.cuda.get_device_capability(self.device)[0] < 8

    def _run(self, ids, mask=None, cache=None):
        """One forward pass up to the deepest captured layer; returns {layer: hidden [B, L, d]}."""
        self._cap = {}
        with chunked_attention(mask is None and self._needs_chunked_attention(ids.size(1))):
            try:
                self.model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=cache is not None)
            except StopForward:
                pass
        return dict(self._cap)

    @torch.inference_mode()
    def pooled(self, items, chunks=0, chunk_min_tokens=2048, prefix_min=4096, chunk_tokens=128):
        """items -> (item_feats [B, Lf, 3, d], opt_feats [sum K, Lo, d]) float32 on the device, plus
        (chunk_feats [B, C, d], chunk_mask [B, C]) when chunks = C > 0: the state span of items whose state has at
        least chunk_min_tokens tokens, cut into C equal pieces and mean-pooled at the deepest feature layer.

        Items whose state reaches prefix_min tokens are READ ONCE per distinct state: the state prefix runs once with
        a cache, then each question continues from a copy of it (several questions about one long record cost about
        one read). Shorter items run as one padded batch, as before."""
        B = len(items)
        Lf, Lo = len(self.feature_layers), len(self.option_layers)
        item_out = [None] * B
        opt_out = [None] * B
        ch_out = [None] * B
        groups, short = {}, []
        for b, it in enumerate(items):
            P = it["spans"]["state"][1]
            if P >= prefix_min:
                groups.setdefault((P, it["ids"][:P].tobytes()), []).append(b)
            else:
                short.append(b)
        if short:
            f, o, c = self._pooled_batch([items[b] for b in short], chunks, chunk_min_tokens, chunk_tokens)
            p = 0
            for r, b in enumerate(short):
                k = len(items[b]["spans"]["options"])
                item_out[b], opt_out[b] = f[r], o[p:p + k]
                ch_out[b] = None if c is None else (c[0][r], c[1][r])
                p += k
        for (P, _), idx in groups.items():
            for b, (f, o, c) in zip(idx, self._pooled_prefix([items[b] for b in idx], P, chunks, chunk_min_tokens, chunk_tokens)):
                item_out[b], opt_out[b], ch_out[b] = f, o, c
        item = torch.stack(item_out, 0)
        opt = torch.cat(opt_out, 0)
        if not chunks:
            return item, opt
        d = item.size(-1)
        ch = torch.zeros(B, chunks, d, device=item.device)
        cm = torch.zeros(B, chunks, dtype=torch.bool, device=item.device)
        for b, c in enumerate(ch_out):
            if c is not None:
                ch[b], cm[b] = c
        return item, opt, ch, cm

    def _chunk_pool(self, h, s0, s1, chunks, chunk_min_tokens, chunk_tokens=128):
        """The state span h[s0:s1] ([L, d]) cut into k equal pieces, k = one per chunk_tokens tokens, at least 2 and at
        most `chunks`; returns (means [chunks, d], mask [chunks]) with the first k valid. States shorter than
        chunk_min_tokens get no chunks."""
        n = s1 - s0
        feats = torch.zeros(chunks, h.size(-1), device=h.device)
        mask = torch.zeros(chunks, dtype=torch.bool, device=h.device)
        if n < max(chunk_min_tokens, 2):
            return feats, mask
        k = max(2, min(chunks, n // max(1, chunk_tokens)))
        cid = torch.arange(n, device=h.device) * k // n
        feats[:k].index_add_(0, cid, h[s0:s1].float())
        feats[:k] /= torch.bincount(cid, minlength=k).clamp_min(1).float()[:, None]
        mask[:k] = True
        return feats, mask

    def _pooled_batch(self, items, chunks, chunk_min_tokens, chunk_tokens=128):
        B = len(items)
        L = max(len(it["ids"]) for it in items)
        K = max(len(it["spans"]["options"]) for it in items)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        ids = torch.full((B, L), pad, dtype=torch.long)
        att = torch.zeros((B, L), dtype=torch.long)
        nseg = 3 + K
        seg = torch.full((B, L), B * nseg, dtype=torch.long)   # dummy bucket for unpooled tokens
        for b, it in enumerate(items):
            n = len(it["ids"])
            ids[b, :n] = torch.from_numpy(it["ids"].astype(np.int64))
            att[b, :n] = 1
            sp = it["spans"]
            seg[b, sp["state"][0]:sp["state"][1]] = b * nseg + 0
            seg[b, sp["question"][0]:sp["question"][1]] = b * nseg + 1
            for k, (a, e) in enumerate(sp["options"]):
                seg[b, a:e] = b * nseg + 3 + k
            seg[b, n - 1] = b * nseg + 2
        ids, att, seg = ids.to(self.device), att.to(self.device), seg.to(self.device).flatten()
        # one sequence has no padding: without a mask the attention runs causal kernels instead of materialising an
        # L x L mask (8 GB at 64k tokens)
        cap = self._run(ids, None if B == 1 else att)
        counts = torch.bincount(seg, minlength=B * nseg + 1).clamp_min(1).float()[:, None]
        pooled = {}
        for i in self.capture:
            h = cap[i].float().reshape(B * L, -1)
            sums = torch.zeros(B * nseg + 1, h.size(1), device=self.device).index_add_(0, seg, h)
            pooled[i] = (sums / counts)[:-1].view(B, nseg, -1)
        item = torch.stack([pooled[i][:, :3] for i in self.feature_layers], 1)             # [B, Lf, 3, d]
        valid = torch.zeros(B, K, dtype=torch.bool, device=self.device)
        for b, it in enumerate(items):
            valid[b, :len(it["spans"]["options"])] = True
        opt = torch.stack([pooled[i][:, 3:] for i in self.option_layers], 2)                # [B, K, Lo, d]
        c = None
        if chunks:
            hl = cap[self.feature_layers[-1]]
            cs = [self._chunk_pool(hl[b], it["spans"]["state"][0], it["spans"]["state"][1], chunks, chunk_min_tokens,
                                   chunk_tokens) for b, it in enumerate(items)]
            c = (torch.stack([x[0] for x in cs]), torch.stack([x[1] for x in cs]))
        return item, opt[valid], c

    def _pooled_prefix(self, items, P, chunks, chunk_min_tokens, chunk_tokens=128):
        """Items sharing the same first P tokens (the prompt up to the end of the state): the prefix runs once with a
        cache, each item's remaining tokens continue from a copy of that cache."""
        import copy
        from transformers import DynamicCache
        cfg = self.model.config
        cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
        s0 = items[0]["spans"]["state"][0]
        cache = DynamicCache(config=cfg)
        pre = torch.from_numpy(items[0]["ids"][:P].astype(np.int64))[None].to(self.device)
        cap = self._run(pre, None, cache)
        state = {i: cap[i][0, s0:P].float().mean(0) for i in self.capture}
        c = self._chunk_pool(cap[self.feature_layers[-1]][0], s0, P, chunks, chunk_min_tokens, chunk_tokens) if chunks else None
        del cap
        out = []
        for it in items:
            sp = it["spans"]
            rest = torch.from_numpy(it["ids"][P:].astype(np.int64))[None].to(self.device)
            h = self._run(rest, None, copy.deepcopy(cache))
            mean = lambda i, a, e: h[i][0, a - P:e - P].float().mean(0)
            n = len(it["ids"])
            f = torch.stack([torch.stack([state[i], mean(i, *sp["question"]), h[i][0, n - 1 - P].float()])
                             for i in self.feature_layers])                                   # [Lf, 3, d]
            o = torch.stack([torch.stack([mean(i, a, e) for i in self.option_layers]) for a, e in sp["options"]])  # [K, Lo, d]
            out.append((f, o, c))
        return out


def _backbone_model_type(model) -> str:
    c = model.config
    return getattr(c, "model_type", "") or getattr(getattr(c, "text_config", None), "model_type", "") or ""


def _remote_perception_main(path, device, feature_layers, option_layers, req, res):
    """Child process of RemotePerception: one frozen backbone on one GPU, serving pooled() calls."""
    import traceback
    try:
        dev = torch.device(device)
        if dev.type == "cuda":
            torch.cuda.set_device(dev)
        perc = Perception(path, dev, feature_layers, option_layers)
        res.put(("ready", {"model_type": _backbone_model_type(perc.model), "feature_layers": perc.feature_layers,
                           "option_layers": perc.option_layers,
                           "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"}))
    except Exception:  # noqa: BLE001
        res.put(("error", None, traceback.format_exc()))
        return
    while True:
        job = req.get()
        if job is None:
            return
        jid, items = job
        try:
            f, o = perc.pooled(items)
            res.put(("ok", jid, f.float().cpu().numpy(), o.float().cpu().numpy()))
        except Exception:  # noqa: BLE001
            if dev.type == "cuda":
                torch.cuda.empty_cache()   # after an out-of-memory error the parent retries smaller batches
            res.put(("error", jid, traceback.format_exc()))


class RemotePerception:
    """A Perception copy in its OWN process (e.g. on the second GPU), with the same pooled() interface.

    Why a process: Qwen3.5's flash-linear-attention kernels are autotuned by Triton, which is not thread-safe, so two
    backbone copies driven by threads of one process crash. A separate process has its own Triton state. The child
    starts loading at once; pooled() waits for it. Results come back as float32 CPU tensors."""
    remote = True

    def __init__(self, path, device, feature_layers, option_layers, timeout_s: float = 1800.0):
        import multiprocessing as mp
        import threading
        ctx = mp.get_context("spawn")   # CUDA cannot be re-initialised in a forked child
        self.device = torch.device(device)
        self.timeout_s = timeout_s
        self._req, self._res = ctx.Queue(), ctx.Queue()
        self.proc = ctx.Process(target=_remote_perception_main, daemon=True,
                                args=(path, str(device), feature_layers, option_layers, self._req, self._res))
        self.proc.start()
        self.info = None
        self._lock = threading.Lock()
        self._jid = 0

    def _get(self, timeout_s):
        import queue
        import time
        t0 = time.time()
        while True:
            try:
                return self._res.get(timeout=5)
            except queue.Empty:
                if not self.proc.is_alive():
                    raise RuntimeError("remote perception process on %s died (exit code %s)" % (self.device, self.proc.exitcode))
                if time.time() - t0 > timeout_s:
                    raise TimeoutError("remote perception on %s did not answer in %.0f s" % (self.device, timeout_s))

    def wait_ready(self):
        if self.info is None:
            msg = self._get(self.timeout_s)
            if msg[0] != "ready":
                raise RuntimeError("remote perception on %s failed to load:\n%s" % (self.device, msg[-1]))
            self.info = msg[1]
        return self.info

    @property
    def model_type(self):
        return self.wait_ready()["model_type"]

    def pooled(self, items):
        with self._lock:
            self.wait_ready()
            self._jid += 1
            self._req.put((self._jid, items))
            msg = self._get(self.timeout_s)
            if msg[0] != "ok" or msg[1] != self._jid:
                raise RuntimeError("remote perception on %s failed:\n%s" % (self.device, msg[-1]))
            return torch.from_numpy(msg[2]), torch.from_numpy(msg[3])

    def close(self):
        try:
            self._req.put(None)
            self.proc.join(10)
        finally:
            if self.proc.is_alive():
                self.proc.terminate()


# ----------------------------------------------------------------------------- physics engine
class MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d_hidden, d_out))

    def forward(self, x):
        return self.net(x)


class MemoryReader(nn.Module):
    """Question-steered reading of the state. The state span is also kept as up to C chunk means in order (about one
    per 128 tokens, so short records have a few and long records C); the question reading + final token form a
    query that attends over them, and the result is added to the world latent. States without chunks get exactly
    zero, and the output layer starts at zero, so a new reader changes nothing until it learns to."""

    def __init__(self, d_in: int, S: int, H: int, n_chunks: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.q = MLP(2 * d_in, H // 2, S)
        self.kv = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, 2 * S))
        self.pos = nn.Parameter(torch.zeros(n_chunks, 2 * S))   # where in the record a chunk sits
        self.out = nn.Linear(S, S)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, qx, ch, cmask):
        B, C, _ = ch.shape
        S = self.out.in_features
        h, dh = self.heads, S // self.heads
        q = self.q(qx).view(B, h, dh)
        # relative position: chunk i of a record cut into n pieces sits at (i + 0.5) / n of the way through
        n = cmask.sum(-1, keepdim=True).clamp_min(1).float()
        pidx = ((torch.arange(C, device=ch.device)[None, :] + 0.5) / n * self.pos.size(0)).long().clamp(0, self.pos.size(0) - 1)
        k, v = (self.kv(ch) + self.pos[pidx]).chunk(2, -1)
        k, v = k.view(B, C, h, dh), v.view(B, C, h, dh)
        att = torch.einsum("bhd,bchd->bhc", q, k) / math.sqrt(dh)
        att = torch.softmax(att.masked_fill(~cmask[:, None, :], -1e4), -1)
        r = torch.einsum("bhc,bchd->bhd", att, v).reshape(B, S)
        return self.out(r) * cmask.any(-1, keepdim=True).float()


class WorldDynamics(nn.Module):
    """WALT's Mamba-style dynamics block, one step: s_t -> s_{t+1} (gated residual update)."""

    def __init__(self, d: int, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.inp = nn.Linear(d, 2 * expand * d)
        self.out = nn.Linear(expand * d, d)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, s):
        x, gate = self.inp(self.norm(s)).chunk(2, -1)
        return s + self.out(F.gelu(x) * torch.sigmoid(gate))


class VegaEngine(nn.Module):
    """Latent decision physics on top of frozen perception features.

    Inputs (already standardized):
      feats   [B, Lf, 3, d]  per feature layer: state-span mean, question-span mean, final token
      opts    [B, K, Lo, d]  per option: option-span mean at the option layers
      opt_mask[B, K] bool, qtype [B] long
    """

    def __init__(self, d_in: int, n_feat_layers: int, n_opt_layers: int, cfg: Dict):
        super().__init__()
        D, S, H = cfg["latent_dim"], cfg["world_dim"], cfg["proj_hidden"]
        p = cfg.get("dropout", 0.1)
        self.D, self.S = D, S
        self.max_steps = cfg["n_steps"]
        self.dt_max = cfg["dt_max"]
        self.settle_eps = cfg["settle_eps"]
        self.force_max = cfg.get("force_max", 20.0)
        self.feat_drop = nn.Dropout(cfg.get("feature_dropout", 0.1))
        # ELMo-style scalar mixes over the frozen model's layers, one per slot (state, question, last) + options
        self.mix = nn.Parameter(torch.zeros(3, n_feat_layers))
        self.opt_mix = nn.Parameter(torch.zeros(n_opt_layers))
        # feature standardization (fitted on the first extraction buffer; stored so inference matches training)
        self.register_buffer("feat_mu", torch.zeros(n_feat_layers, 3, d_in))
        self.register_buffer("feat_sd", torch.ones(n_feat_layers, 3, d_in))
        self.register_buffer("opt_mu", torch.zeros(n_opt_layers, d_in))
        self.register_buffer("opt_sd", torch.ones(n_opt_layers, d_in))
        # WALT projector: observation -> world latent, plus one-step world dynamics ("foresight")
        self.world = MLP(d_in, H, S, p)
        self.foresight = WorldDynamics(S)
        # measurement: question reading + final token -> particle position and momentum
        self.probe = MLP(d_in, H, D, p)
        self.impulse = MLP(d_in, H, D, p)
        self.qtype_emb = nn.Embedding(3, D)
        self.to_z0 = MLP(2 * S + 2 * D, H // 2, D, p)
        self.to_p0 = MLP(2 * S + 2 * D, H // 2, D, p)
        # landscape: one Gaussian well per option (centre, log depth, width, rail gap for score levels)
        self.well = MLP(d_in + D, H, D + 3, p)
        self.rail_dir = MLP(D, H // 4, D)
        self.log_kappa = nn.Parameter(torch.tensor(math.log(0.05)))   # weak harmonic confinement
        # selective-SSM thermostat: input-dependent step size and friction, plus a bounded memory force
        n_mem = cfg.get("ssm_state", 32)
        self.ssm_in = nn.Linear(3 * D + 1, n_mem + 2)
        self.A_log = nn.Parameter(torch.log(torch.linspace(0.5, 4.0, n_mem)))
        self.ssm_out = nn.Linear(n_mem, D)
        self.force_scale = nn.Parameter(torch.tensor(0.1))
        with torch.no_grad():
            self.ssm_in.bias[-2] = 0.0    # dt gate ~ sigmoid(0) = 0.5 * dt_max
            self.ssm_in.bias[-1] = -0.5   # friction ~ softplus(-0.5) ~ 0.47
            self.well.net[-1].bias[D] = 1.0    # log depth ~ 1 -> depth ~ 2.7
            self.well.net[-1].bias[D + 1] = 0.5
        # long-state memory (run 8+; absent in older configs so their weights still load)
        self.memory_chunks = int(cfg.get("memory_chunks", 0) or 0)
        self.memory = MemoryReader(d_in, S, H, self.memory_chunks) if self.memory_chunks else None
        # gated adapters (see attach_adapter); empty for a base engine
        self.gate_nets = nn.ModuleList()
        self.adapter_mode = "off"          # "auto": learned gate per question | "on": all adapters on | "off": base engine
        self._gate_state = {"g": None}

    # -- helpers
    def _mix(self, feats, opts, prestandardized):
        f, o = feats.float(), opts.float()
        if not prestandardized:  # raw perception output (inference); the training cache is stored standardized
            f = (f - self.feat_mu) / self.feat_sd                                # [B, Lf, 3, d]
            o = (o - self.opt_mu) / self.opt_sd                                  # [B, K, Lo, d]
        w = torch.softmax(self.mix, -1)                                          # [3, Lf]
        slots = torch.einsum("blsd,sl->bsd", f, w)                               # [B, 3, d]
        o = torch.einsum("bkld,l->bkd", o, torch.softmax(self.opt_mix, -1))      # [B, K, d]
        return self.feat_drop(slots), self.feat_drop(o)

    def landscape(self, o, opt_mask, qtype, probe):
        """Wells: centres [B,K,D], log depth [B,K], width [B,K]. Score questions put their centres on a rail."""
        B, K, _ = o.shape
        e = self.well(torch.cat([o, probe[:, None, :].expand(B, K, -1)], -1))
        c = e[..., :self.D] + self.qtype_emb(qtype)[:, None, :]
        log_a = e[..., self.D].clamp(-4, 4)
        sigma = 0.3 + F.softplus(e[..., self.D + 1])
        is_score = qtype == QTYPES["score"]
        if bool(is_score.any()):
            m = opt_mask.float()
            r0 = (c * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp_min(1)
            u = F.normalize(self.rail_dir(probe), dim=-1)
            gaps = F.softplus(e[..., self.D + 2]) * m
            t = torch.cumsum(gaps, 1)
            t = t - (t * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1)
            rail = r0[:, None, :] + t[..., None] * u[:, None, :]
            c = torch.where(is_score[:, None, None], rail, c)
        return c, log_a, sigma

    @staticmethod
    def well_energy(z, c, log_a, sigma, opt_mask):
        """Readout energy of each well at position z: E_k = |z - c_k|^2 / 2 sigma_k^2 - log a_k (masked: +1e4)."""
        d2 = ((z[:, None, :] - c) ** 2).sum(-1)
        E = d2 / (2 * sigma ** 2) - log_a
        return E.masked_fill(~opt_mask, 1e4), d2

    def potential_and_grad(self, z, c, log_a, sigma, opt_mask):
        """U(z) = kappa/2 |z|^2 - sum_k a_k exp(-|z - c_k|^2 / 2 sigma_k^2), and its analytic gradient."""
        kappa = self.log_kappa.exp()
        diff = z[:, None, :] - c                                   # [B, K, D]
        d2 = (diff ** 2).sum(-1)
        g = torch.exp(log_a - d2 / (2 * sigma ** 2)) * opt_mask.float()   # a_k * gaussian, [B, K]
        U = 0.5 * kappa * (z ** 2).sum(-1) - g.sum(-1)
        grad = kappa * z + (g[..., None] * diff / (sigma ** 2)[..., None]).sum(1)
        return U, grad

    def well_potential(self, z, c, log_a, sigma, opt_mask):
        """U(z), its gradient, and the well exponent q_k = |z - c_k|^2 / (2 sigma_k^2), from one pass.

        The integrator needs U and grad; the answer readout needs E_k = q_k - log a_k. Both start from
        the same [B, K, D] difference, and the step loop used to build it twice -- once for the gradient
        and once in well_energy -- so the wells are evaluated once and q is handed back for free.
        """
        kappa = self.log_kappa.exp()
        diff = z[:, None, :] - c                                   # [B, K, D]
        d2 = (diff ** 2).sum(-1)
        q = d2 / (2 * sigma ** 2)                                   # exponent of the well, [B, K]
        g = torch.exp(log_a - q) * opt_mask.float()                 # a_k * gaussian, [B, K]
        U = 0.5 * kappa * (z ** 2).sum(-1) - g.sum(-1)
        grad = kappa * z + (g[..., None] * diff / (sigma ** 2)[..., None]).sum(1)
        return U, grad, q

    @staticmethod
    def well_energy_at(q, log_a, opt_mask):
        """Readout energy from a well exponent already computed by well_potential."""
        return (q - log_a).masked_fill(~opt_mask, 1e4)

    def forward(self, feats, opts, opt_mask, qtype, n_steps: Optional[int] = None, halt: bool = False,
                return_traj: bool = True, prestandardized: bool = False, chunks=None, chunk_mask=None):
        slots, o = self._mix(feats, opts, prestandardized)
        gates = self._set_gates(slots)
        s = self.world(slots[:, 0])                     # world latent (state only: question-independent)
        if self.memory is not None and chunks is not None and chunk_mask is not None and bool(chunk_mask.any()):
            ch = chunks.float()
            if not prestandardized:   # chunks come from the deepest feature layer's state span
                ch = (ch - self.feat_mu[-1, 0]) / self.feat_sd[-1, 0]
            s = s + self.memory(torch.cat([slots[:, 1], slots[:, 2]], -1), self.feat_drop(ch), chunk_mask)
        s_next = self.foresight(s)                      # imagined next world state
        probe = self.probe(slots[:, 1]) + self.qtype_emb(qtype)
        imp = self.impulse(slots[:, 2])
        ctx = torch.cat([s, s_next, probe, imp], -1)
        z = self.to_z0(ctx)
        p = self.to_p0(ctx)
        c, log_a, sigma = self.landscape(o, opt_mask, qtype, probe)
        z0 = z
        B = z.size(0)
        h = z.new_zeros(B, self.A_log.numel())
        T = self.max_steps if n_steps is None else n_steps
        settled = torch.zeros(B, dtype=torch.bool, device=z.device)
        steps_used = torch.full((B,), T, dtype=torch.long, device=z.device)
        E0, _ = self.well_energy(z, c, log_a, sigma, opt_mask)
        U, grad, q = self.well_potential(z, c, log_a, sigma, opt_mask)
        gn = grad.norm(dim=-1)
        logits_traj, H_traj = [-E0], [0.5 * (p ** 2).sum(-1) + U]
        A = self.A_log.exp()
        for t in range(T):
            x = torch.cat([z, p, grad, gn[:, None]], -1)
            u = self.ssm_in(x)
            bu, dt_raw, fr_raw = u[:, :-2], u[:, -2:-1], u[:, -1:]
            dt = self.dt_max * torch.sigmoid(dt_raw)                 # selective step (Mamba's Delta)
            gamma = F.softplus(fr_raw)                                # friction
            h = torch.exp(-A * dt) * h + dt * bu                      # diagonal SSM memory of the trajectory
            force = self.force_scale * torch.tanh(self.ssm_out(h))
            g_eff = grad / (1 + gn[:, None] / self.force_max)         # soft cap: no slingshots
            p_new = p * torch.exp(-gamma * dt) - dt * g_eff + dt * force   # symplectic Euler, damped
            z_new = z + dt * p_new
            if halt:  # settled particles stay where they are (early exit per item)
                keep = settled[:, None]
                p_new = torch.where(keep, p, p_new)
                z_new = torch.where(keep, z, z_new)
            z, p = z_new, p_new
            U, grad, q = self.well_potential(z, c, log_a, sigma, opt_mask)
            gn = grad.norm(dim=-1)
            if return_traj or t == T - 1:
                logits_traj.append(-self.well_energy_at(q, log_a, opt_mask))
                H_traj.append(0.5 * (p ** 2).sum(-1) + U)
            kin = 0.5 * (p ** 2).sum(-1)
            now = (kin < self.settle_eps) & (gn < 10 * self.settle_eps) & ~settled
            steps_used = torch.where(now, torch.full_like(steps_used, t + 1), steps_used)
            settled = settled | now
            # Deliberately no `break` on settled.all(): that test is a GPU->host sync on every step.
            # A settled particle is already frozen by the where() above, so a longer horizon returns the
            # same numbers; the loop just runs to T without touching the host. Numbers in
            # bench/bench_fastpath.py, invariance in tests/test_engine_fastpath.py.
        _, d2 = self.well_energy(z, c, log_a, sigma, opt_mask)
        nearest = (d2 / (2 * sigma ** 2)).masked_fill(~opt_mask, 1e4).min(-1).values   # how far from any well bottom
        return {"logits": logits_traj[-1], "logits_traj": logits_traj, "H_traj": H_traj, "z0": z0, "zT": z,
                "s": s, "s_next": s_next, "centers": c, "log_a": log_a, "sigma": sigma,
                "kinetic": 0.5 * (p ** 2).sum(-1), "grad_norm": grad.norm(dim=-1), "nearest": nearest,
                "steps_used": steps_used, "gates": gates}

    def _set_gates(self, slots):
        """Per-question adapter gates [B, n_adapters] (None without adapters or in "off" mode)."""
        if len(self.gate_nets) == 0 or self.adapter_mode == "off":
            self._gate_state["g"] = None
            return None
        if self.adapter_mode == "on":
            g = slots.new_ones(slots.size(0), len(self.gate_nets))
        else:
            x = torch.cat([slots[:, 1], slots[:, 2]], -1)          # question reading + final token
            g = torch.cat([torch.sigmoid(net(x)) for net in self.gate_nets], -1)
        self._gate_state["g"] = g
        return g


# ----------------------------------------------------------------------------- gated adapters (PEFT on the engine)
ADAPTER_TARGETS = ("world.net.1", "probe.net.1", "impulse.net.1", "to_z0.net.1", "to_p0.net.1", "well.net.1", "well.net.4")


class GatedLoRALinear(nn.Module):
    """A frozen Linear plus low-rank adapters. Adapter j adds gate_j * scale * B_j A_j x, where gate_j (per question)
    comes from the engine: 0 leaves the base engine's output exactly unchanged."""

    def __init__(self, base: nn.Linear, state: Dict):
        super().__init__()
        self.base = base
        self.lora_A = nn.ParameterList()
        self.lora_B = nn.ParameterList()
        self.scales: List[float] = []
        self._state = state

    def add(self, rank: int, alpha: float):
        w = self.base.weight
        self.lora_A.append(nn.Parameter(torch.randn(rank, w.shape[1], device=w.device, dtype=w.dtype) / math.sqrt(w.shape[1])))
        self.lora_B.append(nn.Parameter(torch.zeros(w.shape[0], rank, device=w.device, dtype=w.dtype)))   # starts as a no-op
        self.scales.append(alpha / rank)

    def forward(self, x):
        y = self.base(x)
        g = self._state.get("g")
        if g is None:
            return y
        for j, (A, B, sc) in enumerate(zip(self.lora_A, self.lora_B, self.scales)):
            gj = g[:, j].reshape(g.shape[0], *([1] * (x.dim() - 1)))
            y = y + gj * sc * ((x @ A.t()) @ B.t())
        return y


def _target_parent(engine, path):
    mod = engine
    parts = path.split(".")
    for name in parts[:-1]:
        mod = getattr(mod, name)
    return mod, int(parts[-1])


def attach_adapter(engine: "VegaEngine", rank: int = 16, alpha: float = 16.0) -> int:
    """Add one gated adapter (low-rank updates on the engine's input layers + a gate network). The base engine's own
    weights are untouched; returns the adapter index."""
    for path in ADAPTER_TARGETS:
        parent, idx = _target_parent(engine, path)
        mod = parent[idx]
        if not isinstance(mod, GatedLoRALinear):
            mod = GatedLoRALinear(mod, engine._gate_state)
            parent[idx] = mod
        mod.add(rank, alpha)
    d_in = engine.feat_mu.shape[-1]
    dev = engine.feat_mu.device
    net = nn.Sequential(nn.LayerNorm(2 * d_in), nn.Linear(2 * d_in, 128), nn.GELU(), nn.Linear(128, 1)).to(dev)
    engine.gate_nets.append(net)
    return len(engine.gate_nets) - 1


def adapter_state(engine: "VegaEngine", j: int) -> Dict[str, torch.Tensor]:
    """Adapter j's tensors under index-free names (for saving one adapter on its own)."""
    out = {}
    for path in ADAPTER_TARGETS:
        parent, idx = _target_parent(engine, path)
        mod = parent[idx]
        out[path + ".lora_A"] = mod.lora_A[j].detach().float().cpu().contiguous()
        out[path + ".lora_B"] = mod.lora_B[j].detach().float().cpu().contiguous()
    for k, v in engine.gate_nets[j].state_dict().items():
        out["gate." + k] = v.detach().float().cpu().contiguous()
    return out


def load_adapter(engine: "VegaEngine", state: Dict[str, torch.Tensor], rank: int, alpha: float) -> int:
    j = attach_adapter(engine, rank, alpha)
    with torch.no_grad():
        for path in ADAPTER_TARGETS:
            parent, idx = _target_parent(engine, path)
            mod = parent[idx]
            mod.lora_A[j].copy_(state[path + ".lora_A"])
            mod.lora_B[j].copy_(state[path + ".lora_B"])
        engine.gate_nets[j].load_state_dict({k[5:]: v for k, v in state.items() if k.startswith("gate.")})
    return j


def route_adapters(gates: Optional[np.ndarray], threshold: float = 0.5) -> np.ndarray:
    """Which calibration each question uses: the most open adapter if its gate >= threshold, else -1 (base)."""
    if gates is None:
        return None
    g = np.asarray(gates)
    return np.where(g.max(1) >= threshold, g.argmax(1), -1)


def route_per_adapter(gates: Optional[np.ndarray], thresholds) -> np.ndarray:
    """Like :func:`route_adapters`, but each adapter is judged against its OWN threshold.

    ``route_adapters`` takes a single threshold for every adapter, which silently applies adapter 0's
    gate setting to adapters 1, 2, ... Each adapter's metadata carries its own ``gate_threshold``
    (read in ``_api.Vega.add_adapter``, stored as ``adapters[i]["threshold"]``), so a checkpoint
    that ships differing values then routes some questions to the wrong calibration -- and every
    probability for those questions is miscalibrated while still looking like a normal answer.

    A column beyond ``thresholds`` (a gate for an adapter this checkpoint never loaded) falls back to
    0.5 rather than to 0, which would shut it entirely. Ties go to the lowest index, so routing is
    deterministic when two adapters both qualify.
    """
    if gates is None:
        return None
    g = np.asarray(gates, dtype=np.float64)
    if g.ndim != 2:
        return None
    thr = np.full(g.shape[1], 0.5, dtype=np.float64)
    if thresholds is not None and len(thresholds):
        t = np.asarray(thresholds, dtype=np.float64)
        thr[:min(len(t), g.shape[1])] = t[:g.shape[1]]
    qualifies = g >= thr[None, :]
    any_ok = qualifies.any(1)
    best = np.where(qualifies, g, -np.inf).argmax(1)
    return np.where(any_ok, best, -1)


def thermal_features(out: Dict, opt_mask: torch.Tensor, max_steps: int) -> torch.Tensor:
    """Per-item physical state used by thermal calibration: residual kinetic energy, distance to the nearest well
    bottom, fraction of the step budget used, and log number of options."""
    k = opt_mask.sum(-1).float().clamp_min(2)
    return torch.stack([torch.log1p(out["kinetic"].float()), torch.log1p(out["nearest"].float()),
                        out["steps_used"].float() / max(1, max_steps), torch.log(k)], -1)


def engine_from_cfg(cfg: Dict, meta: Dict) -> VegaEngine:
    return VegaEngine(meta["d_model"], len(meta["feature_layers"]), len(meta["option_layers"]), cfg)


# ----------------------------------------------------------------------------- thermal calibration
N_THERMAL = 4


def thermal_tau(theta: np.ndarray, qtype: np.ndarray, phys: np.ndarray) -> np.ndarray:
    """theta [3, 1 + N_THERMAL]: per question type, log tau = b + w . phys. Clamped to [0.2, 20]."""
    th = theta[qtype]
    log_tau = th[:, 0] + (th[:, 1:] * phys).sum(-1)
    return np.exp(np.clip(log_tau, math.log(0.2), math.log(20.0)))


def softmax_np(z: np.ndarray, tau) -> np.ndarray:
    z = z / np.asarray(tau).reshape(-1, 1)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def fit_thermal(logits: np.ndarray, target: np.ndarray, qtype: np.ndarray, phys: np.ndarray,
                use_phys: bool = True, l2: float = 1e-3, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Minimise the (weighted) NLL of softmax(logits / tau(x)) on held-out items. Padded options carry -1e4 logits.

    use_phys=False fits a plain per-type temperature (the baseline thermal calibration must beat).
    """
    theta = np.zeros((3, 1 + N_THERMAL), dtype=np.float64)
    L = torch.tensor(logits, dtype=torch.float64)
    Tg = torch.tensor(target, dtype=torch.float64)
    Q = torch.tensor(qtype, dtype=torch.long)
    P = torch.tensor(phys, dtype=torch.float64)
    Wt = torch.tensor(np.ones(len(logits)) if weights is None else weights, dtype=torch.float64)
    for qt in range(3):
        sel = (Q == qt).nonzero(as_tuple=True)[0]
        if len(sel) < 30:
            continue
        w = torch.zeros(1 + N_THERMAL, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w], lr=0.5, max_iter=300, line_search_fn="strong_wolfe")
        Ls, Ts, Ps, Ws = L[sel], Tg[sel], P[sel], Wt[sel]

        def closure():
            opt.zero_grad()
            lt = w[0] + (Ps * w[1:]).sum(-1) * (1.0 if use_phys else 0.0)
            tau = torch.exp(lt.clamp(math.log(0.2), math.log(20.0)))
            nll = -((Ts * torch.log_softmax(Ls / tau[:, None], -1)).sum(-1) * Ws).sum() / Ws.sum()
            loss = nll + l2 * (w[1:] ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        theta[qt] = w.detach().numpy()
        if not use_phys:
            theta[qt, 1:] = 0.0
    return theta


def conformal_thresholds(probs: np.ndarray, label: np.ndarray, qtype: np.ndarray, alphas=(0.1, 0.05)) -> Dict:
    """Split-conformal (LAC) score thresholds per question type: set = {k : p_k >= 1 - qhat}."""
    out = {}
    for qt in range(3):
        sel = qtype == qt
        n = int(sel.sum())
        if n < 30:
            continue
        scores = 1.0 - probs[sel, :][np.arange(n), label[sel]]
        out[QTYPE_NAMES[qt]] = {}
        for a in alphas:
            lvl = min(1.0, math.ceil((n + 1) * (1 - a)) / n)
            out[QTYPE_NAMES[qt]][str(a)] = float(np.quantile(scores, lvl, method="higher"))
    return out


# ----------------------------------------------------------------------------- metrics
def ece_score(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error: the gap between stated confidence and observed accuracy.

    Equal-width binning over [0, 1], each bin weighted by how many predictions land in it. The
    first bin takes its lower edge inclusively so a confidence of exactly 0 is counted once.
    """
    conf, correct = np.asarray(conf, dtype=np.float64), np.asarray(correct, dtype=np.float64)
    if conf.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    # digitize puts each confidence in its bin; clip folds the right edge into the last bin
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            total += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(total)


def aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    if len(conf) == 0:
        return float("nan")
    order = np.argsort(-conf)
    risk = np.cumsum(1 - correct[order]) / np.arange(1, len(order) + 1)
    return float(risk.mean())


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    # average ranks for ties
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def amp_dtype(name: Optional[str]) -> torch.dtype:
    """The autocast dtype a config name asks for; fp16 is the fallback because the T4 has no bf16."""
    return torch.bfloat16 if str(name).lower() in ("bf16", "bfloat16") else torch.float16
