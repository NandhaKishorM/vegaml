"""The escalation band: send the uncertain middle to a human, on a line you set.

    python examples/15_escalation_band.py

Every engine answer carries an `abstain` flag, and on the published calibration it fires only when
the conformal prediction set comes out empty, which on most question sheets it never does. So the
band reads confidence instead. No labels, no fitting, one line per question.

docs/escalation.md fits the same line to a precision target when you do have labels, and runs the
drift watch that tells you when to refit it.
"""
import vegaml

QUESTIONS = {
    "payment_change": {
        "type": "boolean",
        "instructions": "Does this message ask to change where money is sent?",
    },
    "urgency_pressure": {
        "type": "choice",
        "instructions": "How is the sender pressing for speed?",
        "criteria": {
            "none": "no time pressure at all",
            "stated": "a deadline is mentioned plainly",
            "manufactured": "urgency used to prevent the usual checks",
        },
    },
}

EMAILS = [
    "Hi, as discussed, please note our new bank details for the invoice. Wire today, the "
    "supplier audit closes at 5 and the CFO is on a flight.",
    "Thanks for the updated statement. No action needed, just filing it for the quarter.",
]


def main() -> None:
    # threads=1 is the setting for the shipped reader on CPU; docs/cpu.md has the sweep.
    v = vegaml.load(device="cpu", threads=1)

    # One review line per question. The payment question is the one that moves money, so it is
    # held to a stricter line than the one that only describes tone.
    v.set_band({"payment_change": 0.8, "urgency_pressure": 0.6})

    for email in EMAILS:
        out = v.decide(email, QUESTIONS)
        print("\n%s" % email[:70])
        print("  band: %d automatic, %d to review" % (out["band"]["auto"], out["band"]["review"]))
        for qid, answer in out["answers"].items():
            print("  %-18s %-13s %s  (%s)" % (
                qid,
                answer.get("choice", answer.get("boolean")),
                answer["route"],
                answer["route_reason"]))
            # The flag the band replaces, printed next to it so the difference is visible.
            print("  %-18s abstain=%s  prediction_set=%s"
                  % ("", answer.get("abstain"), answer.get("prediction_set")))


if __name__ == "__main__":
    main()
