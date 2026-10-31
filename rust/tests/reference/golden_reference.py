# Golden reference for the Rust port: runs this repository's vegaml package on cases.json.
#
# Usage (CPU, fp32 reader, needs torch + transformers):
#   python rust/tests/reference/golden_reference.py <repo root> <checkpoint dir>
#       rust/tests/reference/cases.json golden.json [reader hub id]
# The reader defaults to Qwen/Qwen3.5-0.8B, the vega-U root reader.

import json
import sys

import numpy as np
import torch

pkg_parent, ckpt, cases_path, out_path = sys.argv[1:5]
reader = sys.argv[5] if len(sys.argv) > 5 else "Qwen/Qwen3.5-0.8B"
sys.path.insert(0, pkg_parent)

from vegaml._api import Vega  # noqa: E402
from vegaml._common import QTYPES, build_prompt, option_texts, softmax_np, thermal_features, thermal_tau  # noqa: E402


def internal(q):
    """The harness sends formatted option texts; rebuild criteria that format back to exactly them."""
    t, opts = q["type"], q["options"]
    if t == "choice":
        crit = {o: None for o in opts}
    elif t == "score":
        crit = [o.split(" - ", 1)[1] for o in opts]
    else:
        crit = None
    iq = {"t": t, "ins": q["instructions"], "crit": crit}
    assert option_texts(iq) == opts, (option_texts(iq), opts)
    return iq


v = Vega(ckpt, backbone=reader, device="cpu", adapters="off")
cfg = v.cfg
C = v.engine.memory_chunks if v.engine.memory is not None else 0
out = []
with torch.no_grad():
    for case in json.load(open(cases_path, encoding="utf-8"))["cases"]:
        q = internal(case["q"])
        ids, spans = build_prompt(v.perc.tok, case["state"], q, cfg["max_len"], template=0,
                                  opt_max_tokens=cfg["opt_max_tokens"],
                                  query_first_max_state=cfg.get("query_first_max_state", 0))
        items = [{"ids": np.asarray(ids, dtype=np.int32), "spans": spans}]
        feats, opt_flat, chunks, cmask = v.perc.pooled(
            items, chunks=C, prefix_min=cfg.get("prefix_min_tokens", 4096),
            chunk_min_tokens=cfg.get("memory_min_tokens", 2048),
            chunk_tokens=cfg.get("memory_chunk_tokens", 128))
        K = opt_flat.size(0)
        mask = torch.ones(1, K, dtype=torch.bool)
        qtype = torch.tensor([QTYPES[q["t"]]])
        res = v.engine(feats, opt_flat[None], mask, qtype, halt=True, return_traj=False,
                       chunks=chunks, chunk_mask=cmask)
        logits = res["logits"].float().numpy()
        phys = thermal_features(res, mask, v.engine.max_steps).numpy()
        probs = softmax_np(logits.astype(np.float64), thermal_tau(v.theta, qtype.numpy(), phys))[0]
        out.append({
            "name": case["name"], "q": case["q"], "state": case["state"], "ids": [int(i) for i in ids],
            "spans": {"state": list(spans["state"]), "question": list(spans["question"]),
                      "options": [list(s) for s in spans["options"]]},
            "feats": feats[0].reshape(-1).tolist(),
            "opts": [o.reshape(-1).tolist() for o in opt_flat],
            "chunks": [c.tolist() for c, m in zip(chunks[0], cmask[0], strict=True) if bool(m)],
            "logits": logits[0].tolist(), "probs": probs.tolist(),
            "steps_used": int(res["steps_used"][0]), "kinetic": float(res["kinetic"][0]),
            "nearest": float(res["nearest"][0]),
        })
        print(case["name"], "best", int(np.argmax(probs)), "p", round(float(probs.max()), 3),
              "tokens", len(ids), flush=True)
json.dump(out, open(out_path, "w", encoding="utf-8"))
print("wrote", len(out), "golden cases")
