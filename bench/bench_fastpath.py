"""Engine-only bench for the fast path: the physics loop with no backbone in the way.

The frozen backbone read dominates an end-to-end `decide()`, so an engine-side win is invisible
there. This loads the real 0.8B engine and feeds it synthetic features of the right shapes.

    VEGA_WORK=<work dir> python bench/bench_fastpath.py [--batches 1,8,32] [--reps 30]
"""
from __future__ import annotations

import argparse
import functools
import os
import statistics
import sys
import time

import torch

# The checkout this file lives in, so the bench measures the tree it sits in: a hard-coded path
# silently measured whichever worktree was named there, whatever the caller had checked out.
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import vegaml  # noqa: E402
from vegaml._common import VegaEngine  # noqa: E402


def load_engine(size="0.8b"):
    """`vegaml.load()` returns a text-level wrapper; the raw physics module is `.engine.engine`."""
    v = vegaml.load(size)
    for obj in (v, getattr(v, "engine", None), getattr(getattr(v, "engine", None), "engine", None)):
        if isinstance(obj, VegaEngine):
            return obj
    raise SystemExit("no VegaEngine found on %r" % type(v))


def make_batch(eng, B, K=61, device="cuda"):
    d_in = eng.feat_mu.shape[-1]
    Lf, Lo = eng.feat_mu.shape[0], eng.opt_mu.shape[0]
    g = torch.Generator(device=device if device == "cuda" else "cpu").manual_seed(0)
    feats = torch.randn(B, Lf, 3, d_in, device=device, generator=g)
    opts = torch.randn(B, K, Lo, d_in, device=device, generator=g)
    mask = torch.ones(B, K, dtype=torch.bool, device=device)
    qtype = torch.zeros(B, dtype=torch.long, device=device)
    return feats, opts, mask, qtype


def burn(seconds=5.0):
    """Idle clocks turn short kernels into a coin toss: this card sits at 210 MHz until it is worked.

    A few seconds of matmul puts it at boost before anything is timed.
    """
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    t0 = time.time()
    while time.time() - t0 < seconds:
        a @ b
    torch.cuda.synchronize()


def time_call(fn, reps):
    fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        xs.append(1000 * (time.perf_counter() - t0))
    return statistics.median(xs), min(xs)


def count_calls(eng, batch):
    n = [0]

    class Counter(torch.overrides.TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            n[0] += 1
            return func(*args, **(kwargs or {}))

    with Counter():
        eng(*batch, halt=True, return_traj=False)
    return n[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--size", default="0.8b")
    ap.add_argument("--tag", default="")
    ap.add_argument("--compile", action="store_true", help="also time a torch.compile'd forward")
    a = ap.parse_args()
    eng = load_engine(a.size)
    print("engine: %s steps, latent %d, %s parameters%s"
          % (eng.max_steps, eng.D,
             format(sum(t.numel() for t in eng.state_dict().values()), ","),
             (" [" + a.tag + "]") if a.tag else ""), flush=True)
    for B in [int(x) for x in a.batches.split(",")]:
        batch = make_batch(eng, B)
        row = ["B=%-3d" % B]
        for halt in (True, False):
            call = functools.partial(eng, *batch, halt=halt, return_traj=False)
            med, best = time_call(call, a.reps)
            row.append("halt=%-5s %7.2f/%7.2f ms" % (halt, med, best))
        row.append("torch calls %d" % count_calls(eng, batch))
        print("  ".join(row), flush=True)
        if a.compile:
            def call(_batch=batch):
                return eng(*_batch, halt=True, return_traj=False)

            compiled = torch.compile(call, dynamic=False)
            t0 = time.time()
            compiled()
            torch.cuda.synchronize()
            first = time.time() - t0
            # A compiled measurement is only trustworthy if it stopped recompiling: warm up, then read
            # dynamo's counters before and after the timed reps.
            burn()
            for _ in range(5):
                compiled()
            torch.cuda.synchronize()
            burn()
            before = dict(torch._dynamo.utils.counters["stats"])
            med, best = time_call(compiled, a.reps)
            after = dict(torch._dynamo.utils.counters["stats"])
            graphs = after.get("unique_graphs", 0) - before.get("unique_graphs", 0)
            print("      compiled halt=True  %7.2f/%7.2f ms  first %.1f s  recompiles during timing: %d"
                  % (med, best, first, graphs), flush=True)


if __name__ == "__main__":
    main()
