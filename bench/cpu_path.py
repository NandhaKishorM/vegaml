"""What a decision costs on a CPU-only box, and which lever moves it.

Three things change CPU latency, and only one of them is the one people reach for first:

    --threads N      torch's intra-op threads. The right number depends on the READER, not on
                     the engine, and the shipped Qwen3.5 readers want one. Their linear-attention
                     layers fall back to a reference PyTorch implementation when the CUDA kernels
                     are absent, which is always on CPU, and that fallback does not thread.
    --reader ID      which reader inside a vega-U checkpoint. A plain-transformer reader has no
                     such fallback and scales normally.
    --variant int8   dynamic int8 over every Linear, CPU only: a quarter of the weight bytes, at
                     a latency cost that depends entirely on which int8 backend your torch has.

One process per configuration, so nothing shares a thread pool or an allocator. Reported: median
and mean latency, weight bytes, resident memory, and the answers themselves, because a faster path
that changes the answers is not a faster path.

    python bench/cpu_path.py --threads 1
    python bench/cpu_path.py --threads 4 --model nandakishorm/vega-U --reader google/gemma-3-1b-it
    python bench/cpu_path.py --threads 1 --variant int8

docs/cpu.md has what this measured on our own box, and the comparison is only meaningful on yours.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import sys
import time

QUESTIONS = {
    "tier": {
        "type": "choice",
        "instructions": "Which tier should take this request?",
        "criteria": {
            "fast": "short factual questions, lookups, classification, formatting, small talk",
            "standard": "everyday coding, writing, summarisation, single-step tool use",
            "general": "complex reasoning, multi-step work, long context, anything unclear",
        },
    }
}

# Short states, which is what a routing or screening decision actually looks like. --join glues
# them together when you want to measure a longer record.
STATES = [
    "What time does the London office open on Fridays?",
    "Rewrite this paragraph so it fits in a tweet.",
    "My card was charged twice for the same order and I want one of them back.",
    "Convert these dates to ISO 8601 and sort them.",
    "Design a migration plan from our monolith to services, with rollback at every step.",
    "Is 2026 a leap year?",
    "Summarise this contract and flag anything that limits our liability.",
    "Translate 'thank you for your patience' into Malayalam.",
    "The deploy failed with an exit code 137 and no logs. Where do I start?",
    "Add a docstring to this function.",
    "Our auditor wants evidence that every payment change was approved by a second person.",
    "Draft a two-line reply declining the meeting politely.",
]


def rss_gb():
    n = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round((n if sys.platform == "darwin" else n * 1024) / (1024 ** 3), 2)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="fp32", choices=["fp32", "int8"])
    p.add_argument("--threads", type=int, default=0, help="0 leaves torch's own default")
    p.add_argument("--model", default=None, help="checkpoint; a vega-U repo when using --reader")
    p.add_argument("--reader", default=None, help="which reader, e.g. google/gemma-3-1b-it")
    p.add_argument("--join", type=int, default=1, help="states glued into one, for a longer record")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--out", default=None, help="append this run to a JSON file")
    a = p.parse_args()

    if a.threads:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[var] = str(a.threads)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    import vegaml

    cases = (["\n\n".join(STATES[i:i + a.join]) for i in range(0, len(STATES), a.join)]
             if a.join > 1 else list(STATES))

    kw = {"device": "cpu", "quantize": ("int8" if a.variant == "int8" else None)}
    if a.threads:
        kw["threads"] = a.threads
    if a.reader:
        kw["backbone"] = a.reader
    t0 = time.time()
    v = vegaml.load(a.model, **kw)
    load_s = round(time.time() - t0, 2)

    # A quantized Linear keeps its weight in packed params, outside .parameters(), so both variants
    # are weighed the same way: every tensor the module graph actually owns.
    weights = sum(t.numel() * t.element_size()
                  for t in v.engine.perc.model.state_dict().values() if hasattr(t, "numel"))

    for text in cases[: a.warmup]:
        v.decide(text, QUESTIONS, mode="engine")

    rows = []
    for text in cases:
        t = time.perf_counter()
        ans = v.decide(text, QUESTIONS, mode="engine")["answers"]["tier"]
        rows.append({"ms": round(1000 * (time.perf_counter() - t), 1), "choice": ans["choice"],
                     "confidence": round(float(ans["confidence"]), 4), "chars": len(text)})
    ms = [r["ms"] for r in rows]
    report = {"variant": a.variant, "reader": a.reader or "default", "join": a.join,
              "threads": a.threads or torch.get_num_threads(), "cases": len(cases),
              "median_ms": round(statistics.median(ms), 1), "mean_ms": round(statistics.mean(ms), 1),
              "min_ms": min(ms), "max_ms": max(ms), "load_s": load_s,
              "weights_gb": round(weights / (1024 ** 3), 2), "rss_gb": rss_gb(),
              "mean_chars": round(statistics.mean(r["chars"] for r in rows)),
              "mean_confidence": round(statistics.mean(r["confidence"] for r in rows), 4),
              "quantized_engine": (torch.backends.quantized.engine if a.variant == "int8" else None),
              "torch": torch.__version__, "vegaml": vegaml.__version__,
              "platform": sys.platform, "cpu_count": os.cpu_count(), "rows": rows}
    if a.out:
        blob = {}
        if os.path.exists(a.out):
            with open(a.out, encoding="utf-8") as f:
                blob = json.load(f)
        blob.setdefault("runs", []).append(report)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(blob, f, indent=2)
    print(json.dumps({k: val for k, val in report.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
