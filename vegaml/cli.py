"""The vegaml command: one decision from the shell, or a JSONL file of them.

    vegaml "I was billed twice, refund me or I cancel." \
        --choice "team=Which team handles this?=billing,technical,sales" \
        --boolean "churn=Does the customer threaten to leave?"

    cat tickets.jsonl | vegaml --jsonl - --boolean "urgent=Does this need a reply today?"

The state is the positional argument; "-" (or nothing, when stdin is piped) reads it from stdin.
Questions come from the typed flags above, from --questions as inline JSON, or from
--questions @file.json; the forms combine, and a duplicate id across them is an error rather
than a silent override. The result is the decide() dictionary printed as JSON, one object per
line in --jsonl mode, so the output pipes straight into jq or the next process.

The checkpoint loads once, before the first line is answered, which is what makes --jsonl
worth having: a thousand states pay for one load.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, Optional


def _parse_typed(specs, qtype: str, questions: Dict[str, Dict]) -> None:
    """Fold "id=instructions" / "id=instructions=options" flag values into the questions dict."""
    for spec in specs or []:
        parts = spec.split("=", 2)
        if len(parts) < 2 or not parts[0].strip():
            tail = "=a,b,c" if qtype == "choice" else "=low,mid,high" if qtype == "score" else ""
            raise SystemExit("vegaml: --%s expects \"id=instructions%s\", got %r" % (qtype, tail, spec))
        qid, ins = parts[0].strip(), parts[1].strip()
        if qid in questions:
            raise SystemExit("vegaml: question id %r given more than once" % qid)
        q: Dict = {"type": qtype, "instructions": ins}
        if qtype == "choice":
            if len(parts) < 3 or not parts[2].strip():
                raise SystemExit("vegaml: --choice %r names no options; append =a,b,c" % qid)
            q["criteria"] = [o.strip() for o in parts[2].split(",") if o.strip()]
        elif qtype == "score":
            levels = [o.strip() for o in parts[2].split(",")] if len(parts) > 2 else []
            levels = [o for o in levels if o]
            if len(levels) < 2:
                raise SystemExit("vegaml: --score %r needs its ordered levels; append =low,mid,high" % qid)
            q["criteria"] = levels
        questions[qid] = q


def _questions(args) -> Dict[str, Dict]:
    questions: Dict[str, Dict] = {}
    _parse_typed(args.boolean, "boolean", questions)
    _parse_typed(args.choice, "choice", questions)
    _parse_typed(args.score, "score", questions)
    if args.questions:
        raw = args.questions
        if raw.startswith("@"):
            with open(raw[1:], encoding="utf-8") as f:
                raw = f.read()
        try:
            extra = json.loads(raw)
        except json.JSONDecodeError as e:
            raise SystemExit("vegaml: --questions is not valid JSON: %s" % e) from None
        if not isinstance(extra, dict):
            raise SystemExit("vegaml: --questions must be a JSON object of {id: question}")
        for qid, q in extra.items():
            if qid in questions:
                raise SystemExit("vegaml: question id %r given more than once" % qid)
            questions[qid] = q
    if not questions:
        raise SystemExit("vegaml: no questions. Use --boolean/--choice/--score or --questions; "
                         "see vegaml --help")
    return questions


def _state(args) -> str:
    if args.state in (None, "-"):
        text = sys.stdin.read()
        if not text.strip():
            raise SystemExit("vegaml: no state. Pass it as the first argument or pipe it on stdin")
        return text
    return args.state


def _load(args):
    # Imported here, not at module top: --help and flag errors should never pay for torch.
    import vegaml
    kw = {}
    if args.device:
        kw["device"] = args.device
    if args.max_len:
        kw["max_len"] = args.max_len
    return vegaml.load(args.model, **kw)


def _emit(result: Dict, pretty: bool, out) -> None:
    out.write(json.dumps(result, indent=2 if pretty else None, sort_keys=pretty))
    out.write("\n")


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="vegaml",
        description="Typed decisions from the shell: state in, calibrated JSON answers out.")
    ap.add_argument("state", nargs="?", help="the state to decide about; '-' or absent reads stdin")
    ap.add_argument("--boolean", action="append", metavar="ID=INSTRUCTIONS",
                    help="a yes/no question (repeatable)")
    ap.add_argument("--choice", action="append", metavar="ID=INSTRUCTIONS=A,B,C",
                    help="a pick-one question with its options (repeatable)")
    ap.add_argument("--score", action="append", metavar="ID=INSTRUCTIONS=LOW,MID,HIGH",
                    help="an ordered-level question; the levels are its criteria, in order (repeatable)")
    ap.add_argument("--questions", metavar="JSON|@FILE",
                    help="questions as a JSON object, inline or @file.json; combines with the flags")
    ap.add_argument("--jsonl", metavar="FILE",
                    help="batch mode: FILE (or '-') has one JSON object per line, {\"state\": ..., "
                         "\"questions\": {...}?}; lines without questions use the flags'")
    ap.add_argument("--model", default=None,
                    help="checkpoint: nothing for the 0.8B, '4b', a Hub id or a path")
    ap.add_argument("--mode", default=None, choices=["auto", "engine", "ttt", "both"])
    ap.add_argument("--alpha", default="0.1", help="conformal miscoverage level the sets are read at")
    ap.add_argument("--views", type=int, default=1, help="readings averaged per decision (1-3)")
    ap.add_argument("--device", default=None,
                    help="cpu, cuda, cuda:N or mps; picked automatically when absent")
    ap.add_argument("--max-len", type=int, default=None, help="prompt token limit override")
    ap.add_argument("--pretty", action="store_true", help="indent the JSON instead of one compact object")
    args = ap.parse_args(argv)

    default_questions = None
    if args.jsonl is None or args.boolean or args.choice or args.score or args.questions:
        default_questions = _questions(args)

    v = _load(args)
    if args.jsonl is None:
        result = v.decide(_state(args), default_questions, mode=args.mode, alpha=args.alpha, views=args.views)
        _emit(result, args.pretty, sys.stdout)
        return 0

    lines = sys.stdin if args.jsonl == "-" else open(args.jsonl, encoding="utf-8")
    try:
        for n, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit("vegaml: --jsonl line %d is not valid JSON: %s" % (n, e)) from None
            if not isinstance(item, dict) or "state" not in item:
                raise SystemExit("vegaml: --jsonl line %d needs {\"state\": ...}" % n)
            qs = item.get("questions") or default_questions
            if not qs:
                raise SystemExit("vegaml: --jsonl line %d has no questions and none came from flags" % n)
            result = v.decide(item["state"], qs, mode=args.mode, alpha=args.alpha, views=args.views)
            _emit(result, args.pretty, sys.stdout)
    finally:
        if lines is not sys.stdin:
            lines.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
