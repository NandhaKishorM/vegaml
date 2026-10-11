"""Vega 1 with the image path of its Qwen3.5 backbone (zero-shot: the engine was trained on text features only).

The backbone is a Qwen3_5ForConditionalGeneration (vision tower + language model). Vega's Perception loads it as a
causal LM, so only text reaches the engine. Here the full model is loaded, the image tokens take the place of the state
text in Vega's own prompt (same template, question-first prefix, option caps), and the same layers and spans are
pooled: state = mean over the vision block, question = mean over the question, final token, options = mean per option.
The engine, adapters and calibration are Vega's, unchanged.

    vv = VegaVision(model_dir, backbone="Qwen/Qwen3.5-0.8B", device="mps")
    vv.decide_image(pil_image, questions)       # image as state
    vv.decide_text(text, questions)             # text through this path; must equal Vega.decide (parity check)
"""
import json
import os
import sys
import time

import numpy as np
import torch

VISION_STATE = "<|vision_start|><|image_pad|><|vision_end|>"


class VegaVision:
    def __init__(self, model_dir, backbone, device="mps", dtype=torch.float32, adapters="auto", max_pixels=1024 * 32 * 32):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        from . import _common as dc
        from ._api import Vega

        self.dc = dc
        self.dev = torch.device(device)
        self.proc = AutoProcessor.from_pretrained(backbone, max_pixels=max_pixels)
        self.tok = self.proc.tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(backbone, dtype=dtype).to(self.dev).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        cfg = json.load(open(os.path.join(model_dir, "vega_config.json")))
        layers = dc.find_layers(self.model)
        self.feature_layers = dc.resolve_layers(cfg["feature_layers"], len(layers))
        self.option_layers = dc.resolve_layers(cfg["option_layers"], len(layers))
        self.capture = sorted(set(self.feature_layers) | set(self.option_layers))
        self._cap = {}
        last = max(self.capture)
        for i in self.capture:
            def hook(mod, inp, out, i=i):
                self._cap[i] = out[0] if isinstance(out, (tuple, list)) else out
                if i == last:
                    raise dc.StopForward()
            layers[i].register_forward_hook(hook)

        class _Shim:   # Vega only needs these attributes from its perception
            pass
        shim = _Shim()
        shim.tok, shim.feature_layers, shim.option_layers = self.tok, self.feature_layers, self.option_layers
        self.d = Vega(model_dir, backbone=backbone, device=device, adapters=adapters, perception=shim)
        self.cfg = self.d.cfg
        ids = self.tok(VISION_STATE, add_special_tokens=False)["input_ids"]
        assert len(ids) == 3, ids
        self.pad_id = ids[1]

    def _run(self, ids, pixel_values=None, grid=None):
        self._cap = {}
        kw = {"input_ids": ids}
        if pixel_values is not None:
            kw.update(pixel_values=pixel_values, image_grid_thw=grid, mm_token_type_ids=(ids == self.pad_id).long())
        try:
            self.model(**kw)
        except self.dc.StopForward:
            pass
        return dict(self._cap)

    def _chunk_pool(self, h, s0, s1, chunks, chunk_min_tokens, chunk_tokens=128):
        """Same as Perception._chunk_pool: the state span cut into k equal pieces, mean per piece."""
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

    def _features(self, state_text, qs, image=None, chunks=0, chunk_min_tokens=128, chunk_tokens=128):
        """Per question: one forward pass, pooled exactly like Perception._pooled_batch (B = 1)."""
        pv = grid = None
        if image is not None:
            im = self.proc.image_processor(images=[image], return_tensors="pt")
            pv, grid = im["pixel_values"].to(self.dev, self.model.dtype), im["image_grid_thw"].to(self.dev)
            n_img = int(grid.prod()) // (self.proc.image_processor.merge_size ** 2)
        feats, opts, n_tok, chs = [], [], 0, []
        for q in qs:
            ids, sp = self.dc.build_prompt(self.tok, state_text, q, self.cfg["max_len"], template=0,
                                           opt_max_tokens=self.cfg["opt_max_tokens"],
                                           query_first_max_state=self.cfg.get("query_first_max_state", 0))
            ids = list(ids)
            if image is not None:
                j = ids.index(self.pad_id)
                ids = ids[:j] + [self.pad_id] * n_img + ids[j + 1:]
                sh = lambda a: a + (n_img - 1) if a > j else a
                sp = {"state": (sp["state"][0], sh(sp["state"][1])), "question": tuple(map(sh, sp["question"])),
                      "options": [tuple(map(sh, o)) for o in sp["options"]]}
            n_tok += len(ids)
            cap = self._run(torch.tensor([ids], device=self.dev), pv, grid)
            h = {i: cap[i][0].float() for i in self.capture}
            mean = lambda i, a, e: h[i][a:e].mean(0)
            feats.append(torch.stack([torch.stack([mean(i, *sp["state"]), mean(i, *sp["question"]), h[i][len(ids) - 1]])
                                      for i in self.feature_layers]))
            opts.append(torch.stack([torch.stack([mean(i, a, e) for i in self.option_layers]) for a, e in sp["options"]]))
            if chunks:
                chs.append(self._chunk_pool(h[self.feature_layers[-1]], sp["state"][0], sp["state"][1],
                                            chunks, chunk_min_tokens, chunk_tokens))
        return feats, opts, n_tok, chs

    @torch.no_grad()
    def _decide(self, state_text, questions, image=None, alpha="0.1", risk=None):
        from ._common import QTYPES, route_adapters, softmax_np, thermal_features, thermal_tau
        t0 = time.time()
        qids = list(questions)
        if not qids:
            raise ValueError("decide_image/decide_text needs at least one question")
        qs = [self.d._internal(questions[k]) for k in qids]
        eng = self.d.engine
        # text states feed the engine's memory reader exactly like Vega.decide; the image path keeps the
        # zero-chunk behaviour it was benchmarked with (RVL-CDIP-N numbers stay reproducible)
        C = (eng.memory_chunks if eng.memory is not None else 0) if image is None else 0
        feats, olist, n_tok, chs = self._features(state_text, qs, image, chunks=C,
                                                  chunk_min_tokens=self.cfg.get("memory_min_tokens", 2048),
                                                  chunk_tokens=self.cfg.get("memory_chunk_tokens", 128))
        B, K = len(qs), max(o.size(0) for o in olist)
        opts = torch.zeros((B, K) + olist[0].shape[1:], device=self.dev)
        mask = torch.zeros((B, K), dtype=torch.bool, device=self.dev)
        for b, o in enumerate(olist):
            opts[b, :o.size(0)] = o
            mask[b, :o.size(0)] = True
        feats = torch.stack(feats)
        qtype = torch.tensor([QTYPES[q["t"]] for q in qs], device=self.dev)
        ch = torch.stack([c[0] for c in chs]) if C else None
        cm = torch.stack([c[1] for c in chs]) if C else None
        out = eng(feats, opts, mask, qtype, halt=self.d.halt, return_traj=False, chunks=ch, chunk_mask=cm)
        logits = out["logits"].float().cpu().numpy()
        phys = thermal_features(out, mask, eng.max_steps).cpu().numpy()
        qt = qtype.cpu().numpy()
        gates = out["gates"].float().cpu().numpy() if out.get("gates") is not None else None
        route = route_adapters(gates, self.d.adapters[0]["threshold"] if self.d.adapters else 0.5)
        route = np.full(B, -1) if route is None else route
        P = np.zeros_like(logits, dtype=np.float64)
        for rt in np.unique(route):
            sel = route == rt
            P[sel] = softmax_np(logits[sel], thermal_tau(self.d._cal(rt)["theta"], qt[sel], phys[sel]))
        risk = str(risk) if risk is not None else self.d.risk
        answers = {}
        for r, (qid, q) in enumerate(zip(qids, qs)):
            k = int(mask[r].sum())
            # score questions have one label per level; only boolean collapses to [no, yes].
            # Labelling every non-choice question [False, True] indexed off the end of this list
            # for any score question with more than two levels.
            labels = list(q["crit"].keys()) if q["t"] == "choice" else list(range(k)) if q["t"] == "score" \
                else [False, True]
            cal = self.d._cal(int(route[r]))
            p = P[r, :k]
            conf = float(p.max())
            best = int(p.argmax())
            th = cal["conformal"].get(q["t"], {}).get(alpha)
            # LAC set; the top option is always kept so the set is never empty. Same construction
            # as _api.Vega.decide, so an image answer carries what a text answer carries.
            pset = [labels[i] for i in range(k)
                    if th is not None and (p[i] >= max(1.0 - th, 1e-12) or i == best)]
            a = {"probs": {str(labels[i]): float(p[i]) for i in range(k)}, "confidence": conf,
                 "prediction_set": pset if th is not None else None,
                 "abstain": bool(conf < cal["abstain"].get(risk, 0.0)),
                 "unbound": bool(float(out["nearest"][r]) > cal["unbound"]),
                 "settle_steps": int(out["steps_used"][r]), "energy": float(-logits[r, :k].max())}
            if self.d.adapters:
                a["adapter"] = self.d.adapters[int(route[r])]["name"] if route[r] >= 0 else None
                a["gate"] = float(gates[r].max()) if gates is not None else 0.0
            if q["t"] == "choice":
                a["choice"] = labels[best]
            elif q["t"] == "score":
                a["score"] = best
                a["expected_score"] = float((np.arange(k) * p).sum())
            else:
                a["boolean"] = float(p[1])
            answers[qid] = a
        if self.dev.type == "mps":
            torch.mps.synchronize()
        return {"answers": answers, "latency_ms": round(1000 * (time.time() - t0), 1), "n_tokens": n_tok,
                "views": 1}

    def decide_image(self, image, questions, **kw):
        """Answer questions about a picture.

        `image` is a PIL Image, a path to one, or an HxWx3 numpy array.
        """
        return self._decide(VISION_STATE, questions, image=_as_rgb_image(image), **kw)

    def decide_text(self, text, questions, **kw):
        return self._decide(self.dc.serialize_state(text), questions, **kw)


def _as_rgb_image(image):
    """Accept a PIL image, a path, or an HxWx3 array and return a PIL RGB image.

    `decide_image` previously called `image.convert("RGB")` unconditionally, so a path or an array
    raised AttributeError instead of working.
    """
    if hasattr(image, "convert"):          # already a PIL image
        return image.convert("RGB")
    try:
        from PIL import Image
    except ImportError:                   # pragma: no cover - only without the vision extra
        raise ImportError("images need Pillow: pip install 'vegaml[vision]'") from None
    if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
        with Image.open(image) as im:
            return im.convert("RGB")
    return Image.fromarray(np.asarray(image).astype("uint8")).convert("RGB")
