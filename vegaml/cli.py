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
    """Fold "id=instructions" / "id=instructions=options" flag values into the questions dict.

    The id is the text before the FIRST `=`; the options, for a question that takes them, are the
    text after the LAST one. Everything between is the instruction, `=` included: an instruction is
    free-form prose that the model actually reads, so "sev=Score this =0 to 4" asks about
    "Score this =0 to 4", not "Score this".
    """
    for spec in specs or []:
        head, _, rest = spec.partition("=")
        if not rest or not head.strip():
            tail = "=a,b,c" if qtype == "choice" else "=low,mid,high" if qtype == "score" else ""
            raise SystemExit("vegaml: --%s expects \"id=instructions%s\", got %r" % (qtype, tail, spec))
        qid, rest = head.strip(), rest.strip()
        if qid in questions:
            raise SystemExit("vegaml: question id %r given more than once" % qid)
        if qtype in ("choice", "score"):
            ins, _, opts = rest.rpartition("=")
            # no second `=` at all: the whole remainder is the instruction and the options are missing
            if not ins:
                ins, opts = rest, ""
        else:
            ins, opts = rest, None
        q: Dict = {"type": qtype, "instructions": ins}
        if qtype == "choice":
            if not opts.strip():
                raise SystemExit("vegaml: --choice %r names no options; append =a,b,c" % qid)
            q["criteria"] = [o.strip() for o in opts.split(",") if o.strip()]
            if not q["criteria"]:
                raise SystemExit("vegaml: --choice %r names no options; append =a,b,c" % qid)
        elif qtype == "score":
            levels = [o.strip() for o in opts.split(",")] if opts else []
            levels = [o for o in levels if o]
            if len(levels) < 2:
                raise SystemExit("vegaml: --score %r needs its ordered levels; append =low,mid,high" % qid)
            q["criteria"] = levels
        questions[qid] = q


def _read_text(path: str, flag: str) -> str:
    """Read a file a flag pointed at, reporting a bad path in the same voice as every other
    input error here. An unreadable path is the user's mistake, not a crash to dump."""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        raise SystemExit("vegaml: %s cannot read %r: %s" % (flag, path, e.strerror)) from None


def _questions(args) -> Dict[str, Dict]:
    questions: Dict[str, Dict] = {}
    _parse_typed(args.boolean, "boolean", questions)
    _parse_typed(args.choice, "choice", questions)
    _parse_typed(args.score, "score", questions)
    if args.questions:
        raw = args.questions
        if raw.startswith("@"):
            raw = _read_text(raw[1:], "--questions")
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
    if args.threads:
        kw["threads"] = args.threads
    if args.quantize:
        kw["quantize"] = args.quantize
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
    ap.add_argument("--threads", type=int, default=None,
                    help="torch threads for a CPU decision; 1 is what we measure fastest")
    ap.add_argument("--quantize", default=None, choices=["int8"],
                    help="int8 weights for the reader, CPU only: a quarter of the bytes, see docs/cpu.md")
    ap.add_argument("--review-below", type=float, default=None, metavar="T",
                    help="route any answer under this confidence to review (route=\"review\")")
    ap.add_argument("--pretty", action="store_true", help="indent the JSON instead of one compact object")
    args = ap.parse_args(argv)

    # The batch file is opened before the checkpoint is loaded, so a mistyped path costs one short
    # error rather than a model load the user already paid for to be told nothing.
    lines = None
    if args.jsonl is not None:
        if args.jsonl == "-":
            lines = sys.stdin
        else:
            try:
                lines = open(args.jsonl, encoding="utf-8")
            except OSError as e:
                raise SystemExit("vegaml: --jsonl cannot read %r: %s"
                                 % (args.jsonl, e.strerror)) from None

    default_questions = None
    if args.jsonl is None or args.boolean or args.choice or args.score or args.questions:
        default_questions = _questions(args)

    v = _load(args)
    if args.jsonl is None:
        result = v.decide(_state(args), default_questions, mode=args.mode, alpha=args.alpha,
                          views=args.views, review_below=args.review_below)
        _emit(result, args.pretty, sys.stdout)
        return 0

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
            result = v.decide(item["state"], qs, mode=args.mode, alpha=args.alpha,
                              views=args.views, review_below=args.review_below)
            _emit(result, args.pretty, sys.stdout)
    finally:
        if lines is not sys.stdin:
            lines.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
