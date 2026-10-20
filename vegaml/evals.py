"""Measure the model on your own labelled decisions: accuracy, calibration, coverage, latency.

    vegaml-eval decisions.jsonl --questions questions.json

    from vegaml.evals import evaluate
    report = evaluate(v, rows, questions)

Each JSONL row is one labelled state: ``{"state": ..., "labels": {"churn": "true"}}``, with an
optional ``"questions"`` carrying per-row questions where the instruction itself varies. The
shared questions come from ``--questions`` (inline JSON or ``@file``), the same sheet
``decide()`` takes. The report says, per question and overall, how often the model was right,
how honest its probabilities were, whether the conformal sets covered at the rate they claim,
and what each decision cost. ``abstain_rate`` and ``unbound_rate`` are reported separately
because they call for different fixes: many abstains mean the task is hard for the engine
(fit heads on labels), many unbounds mean the traffic drifted off the calibrated map
(refit with fresh samples). Numbers like these on *your* data are the only ones that matter;
nobody's test split is your traffic.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Dict, List, Mapping, Optional, Sequence

from . import label_ids
from .calibrate import canon_probs, ece

__all__ = ["evaluate", "main"]


def _quantile(xs: List[float], q: float) -> float:
    xs = sorted(xs)
    i = min(int(q * (len(xs) - 1) + 0.5), len(xs) - 1)
    return xs[i]


def _canon(value, options: Sequence[str]) -> str:
    """The option id a value names, whatever its spelling: True, "True" and "true" are one."""
    by_fold = {o.lower(): o for o in options}
    return by_fold.get(str(value).lower(), str(value))


def _prediction(answer: Mapping, qtype: str, options: Sequence[str]) -> Optional[str]:
    """The predicted option id, read from the type-specific field or the probabilities."""
    if qtype == "boolean" and "boolean" in answer:
        return "true" if answer["boolean"] >= 0.5 else "false"
    if qtype == "score" and "score" in answer:
        return str(answer["score"])
    if qtype == "choice" and "choice" in answer:
        return str(answer["choice"])
    probs = answer.get("probs")
    if not probs:
        return None
    canon = canon_probs(probs, options)
    return max(canon, key=canon.get)


def evaluate(model, rows: Sequence[Mapping], questions: Mapping, mode: str = "engine",
             alpha: str = "0.1", views: int = 1) -> Dict:
    """One decide() per row, scored against the row's labels.

    Returns ``{"questions": {qid: {...}}, "overall": {...}}``. Per question: ``n``,
    ``accuracy``, ``ece``, ``mean_confidence``, ``abstain_rate``, ``coverage`` (how often the
    label was inside the prediction set) with ``mean_set_size`` beside it, since an always-full
    set covers trivially. Overall: the same pooled, plus ``latency_ms_p50``/``p95`` and
    ``n_tokens`` per decision.
    """
    if not rows:
        raise ValueError("no rows")
    for i, row in enumerate(rows):
        if "state" not in row or "labels" not in row:
            raise ValueError("row %d needs \"state\" and \"labels\"" % i)
    per_q: Dict[str, Dict[str, List]] = {}
    latencies: List[float] = []
    tokens: List[int] = []
    for row in rows:
        asked = {qid: (row.get("questions") or {}).get(qid, questions[qid])
                 for qid in row["labels"] if qid in questions}
        if not asked:
            continue
        started = time.perf_counter()
        out = model.decide(row["state"], asked, mode=mode, alpha=alpha, views=views)
        latencies.append(out.get("latency_ms") or 1000 * (time.perf_counter() - started))
        tokens.append(out.get("n_tokens", 0))
        for qid in asked:
            label = str(row["labels"][qid])
            options = label_ids(asked[qid])
            if label not in set(options):
                raise ValueError("label %r for %r is not one of its options %s"
                                 % (label, qid, options))
            answer = out["answers"][qid]
            d = per_q.setdefault(qid, {"right": [], "conf": [], "abstain": [],
                                       "unbound": [], "covered": [], "set_size": []})
            predicted = _prediction(answer, asked[qid]["type"], options)
            d["right"].append(predicted == label)
            conf = answer.get("confidence")
            if conf is not None:
                d["conf"].append((float(conf), predicted == label))
            d["abstain"].append(bool(answer.get("abstain")))
            d["unbound"].append(bool(answer.get("unbound")))
            pset = answer.get("prediction_set")
            if pset is not None:
                d["covered"].append(label in {_canon(p, options) for p in pset})
                d["set_size"].append(len(pset))
    if not per_q:
        raise ValueError("no row labels a question in the sheet")

    def summarise(d: Dict[str, List]) -> Dict:
        n = len(d["right"])
        s = {"n": n, "accuracy": round(sum(d["right"]) / n, 4),
             "abstain_rate": round(sum(d["abstain"]) / n, 4),
             "unbound_rate": round(sum(d["unbound"]) / n, 4)}
        if d["conf"]:
            s["mean_confidence"] = round(sum(c for c, _ in d["conf"]) / len(d["conf"]), 4)
            s["ece"] = round(ece(d["conf"]), 4)
        if d["covered"]:
            s["coverage"] = round(sum(d["covered"]) / len(d["covered"]), 4)
            s["mean_set_size"] = round(sum(d["set_size"]) / len(d["set_size"]), 4)
        return s

    pooled: Dict[str, List] = {k: sum((d[k] for d in per_q.values()), [])
                               for k in ("right", "conf", "abstain", "unbound", "covered",
                                         "set_size")}
    overall = summarise(pooled)
    overall["latency_ms_p50"] = round(_quantile(latencies, 0.5), 1)
    overall["latency_ms_p95"] = round(_quantile(latencies, 0.95), 1)
    overall["n_tokens_per_decision"] = round(sum(tokens) / len(tokens), 1)
    overall["mode"] = mode
    return {"questions": {qid: summarise(d) for qid, d in sorted(per_q.items())},
            "overall": overall}


# ---------------------------------------------------------------------------- the command
def _read_rows(path: str) -> List[Dict]:
    f = None
    try:
        f = sys.stdin if path == "-" else open(path, encoding="utf-8")
        rows = []
        for i, line in enumerate(f):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise SystemExit("vegaml-eval: line %d is not JSON: %s" % (i + 1, e)) from e
        return rows
    except OSError as e:
        raise SystemExit("vegaml-eval: cannot read %r: %s" % (path, e.strerror)) from None
    finally:
        if f is not None and f is not sys.stdin:
            f.close()


def _read_questions(spec: str) -> Dict:
    if spec.startswith("@"):
        try:
            with open(spec[1:], encoding="utf-8") as f:
                text = f.read()
        except OSError as e:
            raise SystemExit("vegaml-eval: --questions cannot read %r: %s"
                             % (spec[1:], e.strerror)) from None
    else:
        text = spec
    try:
        qs = json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit("vegaml-eval: --questions is not JSON: %s" % e) from e
    if not isinstance(qs, dict) or not qs:
        raise SystemExit("vegaml-eval: --questions must be a non-empty object of questions")
    return qs


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="vegaml-eval",
        description="Measure a checkpoint on labelled decisions: one JSONL row per state.")
    ap.add_argument("rows", help="JSONL of {\"state\": ..., \"labels\": {id: label}}; '-' reads stdin")
    ap.add_argument("--questions", required=True, metavar="JSON|@FILE",
                    help="the question sheet decide() takes, shared by every row")
    ap.add_argument("--model", default=None, help="checkpoint name or path; the 0.8b by default")
    ap.add_argument("--mode", default="engine", choices=["auto", "engine", "ttt"])
    ap.add_argument("--alpha", default="0.1", help="conformal miscoverage level the sets are read at")
    ap.add_argument("--views", type=int, default=1, help="readings averaged per decision (1-3)")
    ap.add_argument("--device", default=None, help="cpu, cuda, mps; picked automatically if absent")
    args = ap.parse_args(argv)

    rows = _read_rows(args.rows)
    questions = _read_questions(args.questions)
    import vegaml
    model = vegaml.load(args.model, device=args.device)
    try:
        report = evaluate(model, rows, questions, mode=args.mode,
                          alpha=args.alpha, views=args.views)
    except ValueError as e:
        raise SystemExit("vegaml-eval: %s" % e) from e
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
