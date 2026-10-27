"""The vegaml-calibrate command: refit on your own labels, or check whether you need to.

    vegaml-calibrate fit --labels labelled.jsonl --questions @questions.json \
        --target-precision 0.97 --out calibration.json

    vegaml-calibrate check --answers last-week.jsonl --calibration calibration.json

Two things a pilot has to be able to do without us. `fit` reads a JSONL file of labelled rows,
refits the temperature, the conformal threshold and the review band, writes one file and prints
what changed. `check` reads a JSONL file of answers your deployment already produced, needs no
labels at all, and says whether the calibration still fits the traffic; it exits non-zero when it
does not, so a cron entry and a pager are the whole alerting setup.

The labels file has one JSON object per line:

    {"state": "...", "labels": {"churn": "true"}}
    {"state": {"ticket": "..."}, "labels": {"team": "billing"}, "questions": {...}}

`questions` per line is optional and overrides the flags for that row, exactly as in `vegaml
--jsonl`. The answers file for `check` takes decide() results as they come out of the API, one per
line, or a bare {question_id: answer} object.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

from .cli import _parse_typed, _read_text

_FAIL_ON = ("recalibrate", "watch")


def _questions(args) -> Dict[str, Dict]:
    """The question sheet from the typed flags and --questions, as in the vegaml command."""
    questions: Dict[str, Dict] = {}
    _parse_typed(getattr(args, "boolean", None), "boolean", questions)
    _parse_typed(getattr(args, "choice", None), "choice", questions)
    _parse_typed(getattr(args, "score", None), "score", questions)
    if getattr(args, "questions", None):
        raw = args.questions
        if raw.startswith("@"):
            raw = _read_text(raw[1:], "--questions")
        try:
            extra = json.loads(raw)
        except json.JSONDecodeError as e:
            raise SystemExit("vegaml-calibrate: --questions is not valid JSON: %s" % e) from None
        if not isinstance(extra, dict):
            raise SystemExit("vegaml-calibrate: --questions must be a JSON object of {id: question}")
        for qid, q in extra.items():
            if qid in questions:
                raise SystemExit("vegaml-calibrate: question id %r given more than once" % qid)
            questions[qid] = q
    return questions


def _jsonl(path: str, what: str):
    """Every JSON object in a JSONL file (or stdin), with the line number in any error."""
    handle = sys.stdin if path == "-" else None
    if handle is None:
        try:
            handle = open(path, encoding="utf-8")
        except OSError as e:
            raise SystemExit("vegaml-calibrate: %s cannot read %r: %s"
                             % (what, path, e.strerror)) from None
    try:
        for n, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield n, json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit("vegaml-calibrate: %s line %d is not valid JSON: %s"
                                 % (what, n, e)) from None
    finally:
        if handle is not sys.stdin:
            handle.close()


def _examples(path: str) -> List[Tuple]:
    out: List[Tuple] = []
    for n, item in _jsonl(path, "--labels"):
        if not isinstance(item, dict) or "state" not in item or "labels" not in item:
            raise SystemExit("vegaml-calibrate: --labels line %d needs {\"state\": ..., "
                             "\"labels\": {...}}" % n)
        if not isinstance(item["labels"], dict) or not item["labels"]:
            raise SystemExit("vegaml-calibrate: --labels line %d has no labels" % n)
        row = [item["state"], {k: str(v) for k, v in item["labels"].items()}]
        if item.get("questions"):
            row.append(item["questions"])
        out.append(tuple(row))
    if not out:
        raise SystemExit("vegaml-calibrate: --labels held no rows")
    return out


def _load(args):
    # Imported here so --help and a mistyped path never pay for torch.
    import vegaml

    kw = {}
    if args.device:
        kw["device"] = args.device
    if args.max_len:
        kw["max_len"] = args.max_len
    if args.threads:
        kw["threads"] = args.threads
    if args.quantize:
        kw["quantize"] = args.quantize
    return vegaml.load(args.model, **kw)


# --------------------------------------------------------------------------------- fit
def _fit(args) -> int:
    questions = _questions(args)
    examples = _examples(args.labels)
    if not questions:
        # Every row carrying its own sheet is legitimate; no sheet anywhere is not.
        from_rows: Dict[str, Dict] = {}
        for row in examples:
            if len(row) > 2:
                from_rows.update(row[2])
        questions = from_rows
    if not questions:
        raise SystemExit("vegaml-calibrate: no questions. Use --boolean/--choice/--score, "
                         "--questions @file.json, or a \"questions\" key on every labels line")
    from .calibrate import calibrate

    v = _load(args)
    cal = calibrate(v, examples, questions, alpha=args.alpha, min_examples=args.min_examples,
                    views=args.views, target_precision=args.target_precision,
                    review_rate=args.review_rate, max_review_rate=args.max_review_rate,
                    band_confidence=args.band_confidence, band_default=args.band_default)
    cal.save(args.out)
    if args.json:
        json.dump({"report": cal.report, "reference": cal.reference}, sys.stdout, indent=2,
                  sort_keys=True)
        sys.stdout.write("\n")
    else:
        _print_fit(cal, args.out, len(examples))
    # A question nobody could calibrate is the reason to look at the output, so it sets the code.
    return 0 if all(r.get("calibrated") for r in cal.report.values()) else 1


def _print_fit(cal, out_path: str, n_rows: int) -> None:
    print("%d labelled rows, %d question(s)\n" % (n_rows, len(cal.report)))
    for qid in sorted(cal.report):
        r = cal.report[qid]
        print("  %s" % qid)
        if not r.get("calibrated"):
            print("    not calibrated: %s" % r.get("reason", "unknown"))
            continue
        print("    %d examples, temperature %.3f, qhat %.3f" % (r["examples"], r["temperature"],
                                                                r["qhat"]))
        print("    ECE %.3f -> %.3f on the fitting examples" % (r["ece_before"], r["ece_after"]))
        ref = cal.reference.get(qid, {})
        print("    accuracy %.3f, mean confidence %.3f, answer mix %s"
              % (ref.get("accuracy", float("nan")), ref.get("mean_confidence", float("nan")),
                 ", ".join("%s %.2f" % kv for kv in sorted(ref.get("predicted_mix", {}).items()))))
        if ref.get("abstention_reachable") is False:
            print("    the conformal set on this question can never be empty, so the published "
                  "abstain flag cannot fire here; the review band below is the escalation signal")
        b = r.get("band")
        if not b:
            continue
        if not b["reachable"]:
            print("    review band: %s" % b["reason"])
            continue
        held = b.get("held_out")
        print("    review band: review below confidence %.3f, which reviews %.1f%% of traffic"
              % (b["threshold"], 100 * b["review_rate"]))
        print("    the %.1f%% answered automatically scored %.3f, at least %.3f with %d%% confidence"
              % (100 * (1 - b["review_rate"]), b["precision"], b["precision_lb"],
                 round(100 * b["confidence"])))
        if held and held.get("precision") is not None:
            print("    held out, the same rule scored %.3f at a %.1f%% review rate"
                  % (held["precision"], 100 * held["review_rate"]))
    print("\nwritten to %s" % out_path)


# --------------------------------------------------------------------------------- check
def _check(args) -> int:
    from .calibrate import Calibration
    from .monitor import Monitor

    try:
        cal = Calibration.load(args.calibration)
    except OSError as e:
        raise SystemExit("vegaml-calibrate: --calibration cannot read %r: %s"
                         % (args.calibration, e.strerror)) from None
    # --state on the first run has nothing to load yet, which is the normal way a daily check
    # starts rather than an error to report.
    carried = bool(args.state) and os.path.exists(args.state)
    try:
        mon = (Monitor.load_state(args.state, min_answers=args.min_answers) if carried
               else Monitor.from_calibration(cal, min_answers=args.min_answers))
    except (ValueError, OSError) as e:
        raise SystemExit("vegaml-calibrate: %s" % e) from None
    for n, item in _jsonl(args.answers, "--answers"):
        if not isinstance(item, dict):
            raise SystemExit("vegaml-calibrate: --answers line %d is not a JSON object" % n)
        try:
            mon.observe(item)
        except ValueError as e:
            raise SystemExit("vegaml-calibrate: --answers line %d: %s" % (n, e)) from None
    report = mon.report()
    if args.state:
        mon.save_state(args.state)
    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        _print_check(report, args)
    return 1 if _RANK[report["status"]] >= _RANK[args.fail_on] else 0


_RANK = {"insufficient": 0, "ok": 1, "watch": 2, "recalibrate": 3}


def _print_check(report: Dict, args) -> None:
    print("status: %s   (%d answers)\n" % (report["status"], report["answers_seen"]))
    for qid in sorted(report["questions"]):
        q = report["questions"][qid]
        print("  %s: %s" % (qid, q["status"]))
        if "psi" in q:
            print("    %d answers, PSI %.3f" % (q["n"], q["psi"]))
            print("    mix now       %s" % ", ".join("%s %.2f" % kv
                                                     for kv in sorted(q["predicted_mix"].items())))
            print("    mix calibrated%s" % ("  " + ", ".join("%s %.2f" % kv
                                            for kv in sorted(q["reference_mix"].items()))))
            if "confidence_delta" in q:
                print("    confidence %.3f against %.3f (%+.3f)"
                      % (q["mean_confidence"], q["reference_confidence"], q["confidence_delta"]))
            if "review_rate" in q:
                ref = q.get("reference_review_rate")
                print("    review load %.1f%%%s"
                      % (100 * q["review_rate"],
                         "" if ref is None else " against %.1f%%" % (100 * ref)))
        print("    %s" % q.get("reason", ""))
    if _RANK[report["status"]] >= _RANK["watch"]:
        print("\nto refit, label a few hundred recent rows and run:\n"
              "  vegaml-calibrate fit --labels recent.jsonl --questions @questions.json \\\n"
              "      --target-precision 0.97 --out %s" % args.calibration)


# --------------------------------------------------------------------------------- argv
def _common(sub) -> None:
    sub.add_argument("--model", default=None,
                     help="checkpoint: nothing for the 0.8B, '4b', a Hub id or a path")
    sub.add_argument("--device", default=None, help="cpu, cuda, cuda:N or mps; automatic when absent")
    sub.add_argument("--max-len", type=int, default=None, help="prompt token limit override")
    sub.add_argument("--threads", type=int, default=None,
                     help="torch threads for a CPU read; 1 is what we measure fastest")
    sub.add_argument("--quantize", default=None, choices=["int8"],
                     help="int8 weights for the reader, CPU only (docs/cpu.md)")
    sub.add_argument("--json", action="store_true", help="print the report as JSON")


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="vegaml-calibrate",
        description="Refit a calibration on your own labels, or check whether it still fits.")
    subs = ap.add_subparsers(dest="cmd", required=True)

    f = subs.add_parser("fit", help="refit temperature, conformal threshold and review band")
    f.add_argument("--labels", required=True, metavar="FILE",
                   help="JSONL of {\"state\": ..., \"labels\": {...}}; '-' reads stdin")
    f.add_argument("--out", default="calibration.json", metavar="FILE",
                   help="where to write the calibration (default calibration.json)")
    f.add_argument("--boolean", action="append", metavar="ID=INSTRUCTIONS")
    f.add_argument("--choice", action="append", metavar="ID=INSTRUCTIONS=A,B,C")
    f.add_argument("--score", action="append", metavar="ID=INSTRUCTIONS=LOW,MID,HIGH")
    f.add_argument("--questions", metavar="JSON|@FILE",
                   help="questions as a JSON object, inline or @file.json")
    f.add_argument("--alpha", type=float, default=0.1,
                   help="conformal miscoverage the prediction sets should hold (default 0.1)")
    f.add_argument("--min-examples", type=int, default=10,
                   help="labelled examples a question needs before anything is fitted for it")
    f.add_argument("--views", type=int, default=1, help="readings averaged per example (1-3)")
    f.add_argument("--target-precision", type=float, default=None, metavar="P",
                   help="fit the review band so the automatic share holds precision P")
    f.add_argument("--review-rate", type=float, default=None, metavar="R",
                   help="instead fix the share sent to review at R and report what it buys")
    f.add_argument("--max-review-rate", type=float, default=1.0, metavar="R",
                   help="cap on the share a precision target is allowed to send to review")
    f.add_argument("--band-confidence", type=float, default=0.95, choices=[0.9, 0.95, 0.975, 0.99],
                   help="confidence the precision target must hold with (default 0.95)")
    f.add_argument("--band-default", type=float, default=None, metavar="T",
                   help="review line for questions that got no band of their own")
    _common(f)

    c = subs.add_parser("check", help="has the traffic drifted away from this calibration")
    c.add_argument("--answers", required=True, metavar="FILE",
                   help="JSONL of decide() results; '-' reads stdin")
    c.add_argument("--calibration", required=True, metavar="FILE",
                   help="the calibration.json to check against")
    c.add_argument("--state", default=None, metavar="FILE",
                   help="carry counters across runs in FILE, so a daily check accumulates")
    c.add_argument("--min-answers", type=int, default=50,
                   help="answers a question needs before a verdict is called (default 50)")
    c.add_argument("--fail-on", default="recalibrate", choices=list(_FAIL_ON),
                   help="exit 1 at this verdict or worse (default recalibrate)")
    c.add_argument("--json", action="store_true", help="print the report as JSON")

    args = ap.parse_args(argv)
    if args.cmd == "fit":
        if args.target_precision is not None and args.review_rate is not None:
            raise SystemExit("vegaml-calibrate: --target-precision or --review-rate, not both")
        return _fit(args)
    return _check(args)


if __name__ == "__main__":
    raise SystemExit(main())
