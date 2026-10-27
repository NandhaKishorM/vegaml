"""Vega inference: decide(state, questions) -> calibrated typed answers, with the physics behind each one.

    d = Vega("/kaggle/working/vega_model")            # backbone read from vega_config.json (or backbone=...)
    d.decide("I was billed twice, refund me or I cancel.",
             {"team": {"type": "choice", "instructions": "Which team handles this?",
                       "criteria": {"billing": "payments and refunds", "technical": "bugs", "sales": "pricing"}},
              "churn": {"type": "boolean", "instructions": "Does the customer threaten to leave?"}})

Every answer carries: probabilities (thermally calibrated), confidence, a split-conformal prediction set, whether to
abstain at the configured risk, whether the particle ended unbound (far from every well: likely out of
distribution), how many integration steps it took to settle, and its final energy.

Adapters (model_dir/adapters/*.safetensors + .json) extend the frozen engine to new decision families without
changing it. adapters="auto" (default): a learned gate per question decides whether an adapter applies, and each
answer uses that adapter's calibration or the base one; "on": every adapter always on; "off": the base engine only.
Answers then also carry "adapter" (name or None) and "gate".
"""
import glob
import hashlib
import json
import os
import time

import numpy as np
import torch

from ._common import (
    QTYPES,
    Perception,
    VegaEngine,
    build_prompt,
    load_adapter,
    normalise_conformal,
    option_texts,
    prose_state,
    resolve_device,
    route_adapters,
    route_per_adapter,
    serialize_state,
    set_cpu_threads,
    softmax_np,
    thermal_features,
    thermal_tau,
)


class Vega:
    def __init__(self, model_dir, backbone=None, device=None, halt=True, calibration="default", adapters="auto",
                 perception=None, max_len=None, quantize=None, threads=None):
        """calibration="teacher" loads the teacher-matching calibration of a typed-decisions fine-tune (if present)
        instead of the default one fitted to the correct answers. perception: an already loaded Perception to share.
        max_len: prompt token limit, overriding the trained one (e.g. 65536 to read long records; the engine was
        trained at its configured length, so accuracy beyond it has to be measured).
        quantize="int8": dynamic int8 over the reader's linear layers, CPU only. A quarter of the weight bytes,
        at a latency and agreement cost measured in docs/cpu.md rather than assumed.
        threads: torch threads for a CPU decision. One is what we measure fastest, by a lot; left unset the
        process default stands and a note says so."""
        from safetensors.torch import load_file
        with open(os.path.join(model_dir, "vega_config.json"), encoding="utf-8") as f:
            self.cfg = json.load(f)
        if max_len:
            self.cfg["trained_max_len"] = self.cfg["max_len"]
            self.cfg["max_len"] = int(max_len)
        self.model_dir = model_dir
        self.calibration = calibration
        self.device = resolve_device(device)
        if self.device.type == "cpu":
            self.threads = set_cpu_threads(threads)
        # Resolution order, most specific first. cfg["backbone"] is whatever absolute path the
        # training machine used, so it is only usable if it happens to exist here; otherwise fall
        # back to cfg["backbone_name"], the Hub id, which is what makes a downloaded checkpoint
        # work without the caller naming a backbone.
        from .hub import backbone_hub_id, looks_like_hub_id
        local = os.path.join(model_dir, "backbone")
        if backbone is not None:
            bb = backbone
        elif os.path.isdir(local):
            bb = local
        elif os.path.isdir(self.cfg.get("backbone", "")):
            bb = self.cfg["backbone"]
        elif looks_like_hub_id(self.cfg.get("backbone", "")):
            # an attached dir stores the full repository id its port was fitted for
            # ("HuggingFaceTB/SmolLM2-1.7B-Instruct"); the bare backbone_name below would drop
            # the namespace, which only resolves for the shipped readers
            bb = self.cfg["backbone"]
        elif self.cfg.get("backbone_name"):
            # the config stores the bare reader name ("Qwen3.5-0.8B"); downloading it needs
            # the owning namespace, which the hub module knows for every shipped reader
            bb = backbone_hub_id(self.cfg["backbone_name"])
        else:
            bb = self.cfg["backbone"]
        self.backbone = bb     # the one actually used, so the vision path loads the same weights
        self.perc = perception or Perception(bb, self.device, self.cfg["feature_layers"], self.cfg["option_layers"],
                                            quantize=quantize)
        assert self.perc.feature_layers == self.cfg["feature_layers"], "backbone depth differs from training"
        self.engine = VegaEngine(self.cfg["d_model"], len(self.cfg["feature_layers"]), len(self.cfg["option_layers"]),
                                  self.cfg)
        self.engine.load_state_dict(load_file(os.path.join(model_dir, "engine.safetensors")))
        if self.cfg.get("port"):
            # A vega-U (universal) checkpoint: the engine lives in a shared space and this dir
            # carries the backbone's port into it. Raw features are standardized with the
            # extraction statistics and ported before the engine ever sees them.
            from ._universal import UniversalEngine
            self.engine = UniversalEngine(model_dir, self.cfg, self.engine)
        self.engine.to(self.device).eval()
        self.halt = halt
        cal = self.cfg.get("calibration_teacher" if calibration == "teacher" else "calibration") or {}
        if calibration == "teacher" and not cal:
            raise ValueError("this model has no teacher calibration block (only typed-decisions fine-tunes do)")
        self.theta = np.array(cal.get("theta", np.zeros((3, 5))), dtype=np.float64)
        self.conformal = normalise_conformal(cal.get("conformal", {}))
        self.abstain = cal.get("abstain_threshold", {})
        self.risk = cal.get("default_risk", "0.1")
        self.unbound = cal.get("unbound_threshold", float("inf"))
        if not cal:
            print("vega: no calibration block in vega_config.json, so probabilities are raw and "
                  "prediction_set is None. vegaml.calibrate(model_dir, examples, questions) "
                  "refits it on your own labelled examples.")
        self.adapters = []
        self._sha = None
        assert adapters in ("auto", "on", "off"), adapters
        if adapters != "off":
            for js in sorted(glob.glob(os.path.join(model_dir, "adapters", "*.json"))):
                meta = json.load(open(js, encoding="utf-8"))
                if adapters == "auto" and not meta.get("gate_trained"):
                    print("vega: adapter '%s' has no trained gate (fitted without replay data); load it with "
                          "adapters=\"on\" to use it" % meta["name"])
                    continue
                if meta.get("base_engine_sha256") and meta["base_engine_sha256"] != self.base_sha256():
                    print("vega: adapter '%s' was trained on a different base engine - skipped" % meta["name"])
                    continue
                self.add_adapter(load_file(js[:-len(".json")] + ".safetensors"), meta)
        self.set_adapters(adapters)

    def base_sha256(self):
        if self._sha is None:
            h = hashlib.sha256()
            with open(os.path.join(self.model_dir, "engine.safetensors"), "rb") as f:
                for chunk in iter(lambda: f.read(1 << 22), b""):
                    h.update(chunk)
            self._sha = h.hexdigest()
        return self._sha

    def _core(self):
        """The VegaEngine itself: behind the port wrapper on a vega-U checkpoint, or self.engine
        directly. Attribute WRITES must land here; reads fall through the wrapper anyway."""
        return getattr(self.engine, "engine", self.engine)

    def add_adapter(self, state, meta):
        """Attach one adapter (tensors + its json) to the engine; returns its index."""
        j = load_adapter(self._core(), state, meta["rank"], meta["alpha"])
        self.engine.to(self.device).eval()
        cal = (meta.get("calibration_teacher") if self.calibration == "teacher" else None) or meta["calibration"]
        self.adapters.append({"name": meta["name"], "meta": meta, "cal": {
            "theta": np.array(cal["theta"], dtype=np.float64),
            "conformal": normalise_conformal(cal.get("conformal", {})),
            "abstain": cal.get("abstain_threshold", {}), "unbound": cal.get("unbound_threshold", float("inf"))},
            "threshold": meta.get("gate_threshold", 0.5)})
        return j

    def set_adapters(self, mode):
        """"auto" (gated), "on" (all adapters always on) or "off" (base engine exactly)."""
        assert mode in ("auto", "on", "off"), mode
        if mode == "on" and len(self.adapters) > 1:
            print("vega: %d adapters all on at once; \"auto\" lets each question pick" % len(self.adapters))
        self._core().adapter_mode = mode if self.adapters else "off"

    def _cal(self, r):
        if r < 0:
            return {"theta": self.theta, "conformal": self.conformal, "abstain": self.abstain, "unbound": self.unbound}
        return self.adapters[r]["cal"]

    def _route(self, gates):
        """Which calibration each question uses, judged against each adapter's OWN gate threshold.

        A single threshold for all adapters silently applies adapter 0's setting to the others, so a
        checkpoint that ships per-adapter gate_thresholds routes to the wrong calibration.
        """
        if gates is None:
            return None
        if not self.adapters:
            return route_adapters(gates, 0.5)
        return route_per_adapter(gates, [a["threshold"] for a in self.adapters])

    @staticmethod
    def _internal(qdef):
        t = qdef["type"]
        if t not in QTYPES:
            raise ValueError("unknown question type %r: expected one of %s"
                             % (t, ", ".join(QTYPES)))
        crit = qdef.get("criteria")
        if t == "choice" and isinstance(crit, list):
            crit = {c: None for c in crit}
        ins = qdef["instructions"] if isinstance(qdef["instructions"], str) else json.dumps(qdef["instructions"])
        return {"t": t, "ins": ins, "crit": crit}

    def _view(self, text, qs, template, orders, state_ids=None):
        """One reading of the state under one surface form. Returns calibrated probabilities per question in ORIGINAL
        option order, plus the engine output / route / raw logits (in prompt order) for the physics fields."""
        if state_ids is None:
            state_ids = self.perc.tok(text, add_special_tokens=False)["input_ids"]
        items = []
        for q, order in zip(qs, orders):
            ids, spans = build_prompt(self.perc.tok, text, q, self.cfg["max_len"], template=template, option_order=order,
                                      opt_max_tokens=self.cfg["opt_max_tokens"],
                                      query_first_max_state=self.cfg.get("query_first_max_state", 0),
                                      state_ids=state_ids)
            items.append({"ids": np.asarray(ids, dtype=np.int32), "spans": spans})
        C = self.engine.memory_chunks if self.engine.memory is not None else 0
        pm = self.cfg.get("prefix_min_tokens", 4096)
        if C:   # long states are read once for all questions; chunk summaries feed the memory reader
            feats, opt_flat, chunks, cmask = self.perc.pooled(items, chunks=C, prefix_min=pm,
                                                              chunk_min_tokens=self.cfg.get("memory_min_tokens", 2048),
                                                              chunk_tokens=self.cfg.get("memory_chunk_tokens", 128))
        else:
            (feats, opt_flat), chunks, cmask = self.perc.pooled(items, prefix_min=pm), None, None
        K = max(len(it["spans"]["options"]) for it in items)
        B = len(items)
        opts = torch.zeros((B, K) + opt_flat.shape[1:], device=self.device)
        mask = torch.zeros((B, K), dtype=torch.bool, device=self.device)
        o = 0
        for b, it in enumerate(items):
            k = len(it["spans"]["options"])
            opts[b, :k] = opt_flat[o:o + k]
            mask[b, :k] = True
            o += k
        qtype = torch.tensor([QTYPES[q["t"]] for q in qs], device=self.device)
        out = self.engine(feats, opts, mask, qtype, halt=self.halt, return_traj=False, chunks=chunks, chunk_mask=cmask)
        logits = out["logits"].float().cpu().numpy()
        phys = thermal_features(out, mask, self.engine.max_steps).cpu().numpy()
        qt = qtype.cpu().numpy()
        gates = out["gates"].float().cpu().numpy() if out.get("gates") is not None else None
        route = self._route(gates)
        route = np.full(B, -1) if route is None else route
        P = np.zeros_like(logits, dtype=np.float64)
        for rt in np.unique(route):   # each question calibrated by the adapter it went through (or the base)
            sel = route == rt
            P[sel] = softmax_np(logits[sel], thermal_tau(self._cal(rt)["theta"], qt[sel], phys[sel]))
        P_orig = []
        for r, order in enumerate(orders):
            po = np.zeros(len(order))
            po[order] = P[r, :len(order)]
            P_orig.append(po)
        return {"P": P_orig, "out": out, "route": route, "gates": gates, "logits": logits, "mask": mask,
                "n_tokens": int(sum(len(it["ids"]) for it in items))}

    @torch.no_grad()
    def decide(self, state, questions, alpha="0.1", risk=None, views=1):
        """views = 1: one reading. views = 2 adds a reading with every choice / yes-no question's options in reverse
        order; views = 3 also adds the state rendered as prose with the second template. Probabilities are averaged
        (test-time averaging over surface forms: steadier answers, about views x the cost)."""
        t0 = time.time()
        qids = list(questions)
        qs = [self._internal(questions[k]) for k in qids]
        n_opt = [len(option_texts(q)) for q in qs]
        ident = [list(range(k)) for k in n_opt]
        text = serialize_state(state)
        st_ids = self.perc.tok(text, add_special_tokens=False)["input_ids"]
        runs = [self._view(text, qs, 0, ident, state_ids=st_ids)]
        if views >= 2:
            rev = [o[::-1] if q["t"] != "score" else o for q, o in zip(qs, ident)]
            runs.append(self._view(text, qs, 0, rev, state_ids=st_ids))
        if views >= 3:
            p_text = prose_state(state) if not isinstance(state, str) else text
            p_ids = st_ids if p_text == text else None
            runs.append(self._view(p_text, qs, 1, ident, state_ids=p_ids))
        base = runs[0]
        out, route, gates, logits, mask = base["out"], base["route"], base["gates"], base["logits"], base["mask"]
        risk = str(risk) if risk is not None else self.risk
        answers = {}
        for r, (qid, q) in enumerate(zip(qids, qs)):
            k = n_opt[r]
            p = np.mean([run["P"][r] for run in runs], 0)
            cal = self._cal(int(route[r]))
            labels = list(q["crit"].keys()) if q["t"] == "choice" else list(range(k)) if q["t"] == "score" else [False, True]
            conf = float(p.max())
            th = cal["conformal"].get(q["t"], {}).get(alpha)
            best = int(p.argmax())
            # LAC set; the top option is always kept so the set is never empty
            pset = [labels[i] for i in range(k) if th is not None and (p[i] >= max(1.0 - th, 1e-12) or i == best)]
            a = {"probs": {str(labels[i]): float(p[i]) for i in range(k)}, "confidence": conf,
                 "prediction_set": pset if th is not None else None,
                 "abstain": bool(conf < cal["abstain"].get(risk, 0.0)),
                 "unbound": bool(float(out["nearest"][r]) > cal["unbound"]),
                 "settle_steps": int(out["steps_used"][r]), "energy": float(-logits[r, :k].max())}
            if len(runs) > 1:
                a["view_agreement"] = float(np.mean([int(run["P"][r].argmax()) == best for run in runs]))
            if self.adapters:
                a["adapter"] = self.adapters[int(route[r])]["name"] if route[r] >= 0 else None
                a["gate"] = float(gates[r].max()) if gates is not None else 0.0
            if q["t"] == "choice":
                a["choice"] = labels[best]
            elif q["t"] == "score":
                a["score"] = best
                a["expected_score"] = float((np.arange(k) * p).sum())
            else:
                a["boolean"] = float(p[1])
            answers[qid] = a
        return {"answers": answers, "latency_ms": round(1000 * (time.time() - t0), 1),
                "n_tokens": sum(run["n_tokens"] for run in runs), "views": len(runs)}
