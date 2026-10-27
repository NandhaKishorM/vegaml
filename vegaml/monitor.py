"""Drift watch: when your traffic has moved away from what the calibration was fitted on.

    from vegaml.monitor import Monitor

    mon = Monitor.from_calibration(cal)
    for result in stream:
        mon.observe(result)
    print(mon.report()["status"])        # "ok", "watch" or "recalibrate"

Our own drift runs are the reason this exists. A covariate shift, the same decisions written in
different words, left the calibration intact: expected calibration error actually improved, 0.116
to 0.055, and conformal coverage held at 90.0% against a 90% target. A prior shift, the same words
carrying a different mix of answers, broke it on all five corpora we tried. So the signal worth
watching is not how the text looks, which is hard to measure and did not matter, but the mix of
answers coming out, which is one counter per option and did.

The statistic is the population stability index, which is what model-risk teams in banking already
read, with its conventional bands: below 0.1 is stable, 0.1 to 0.25 is worth watching, 0.25 and
above means refit. Mean confidence, review rate and the unbound rate are watched beside it: a
band whose review load has doubled is costing a pilot money whatever the mix says, and a rising
unbound rate is the engine saying the records now land far from every well it knows, which is the
one signal here that sees new material rather than only a new mix.

Nothing here needs labels. That is the point: labels are what recalibration needs, and this is what
tells you to go and collect them.
"""
from __future__ import annotations

import json
import math
from typing import Dict, Mapping, Optional

__all__ = ["Monitor", "psi", "PSI_WATCH", "PSI_ALERT", "STATUSES"]

PSI_WATCH = 0.1
PSI_ALERT = 0.25
STATUSES = ("ok", "watch", "recalibrate", "insufficient")
_RANK = {"insufficient": 0, "ok": 1, "watch": 2, "recalibrate": 3}

# A share this small is treated as this small rather than as zero: the index takes a logarithm of
# the ratio, and one option that never came up would otherwise make it infinite.
_FLOOR = 1e-3


def psi(reference: Mapping[str, float], current: Mapping[str, float]) -> float:
    """Population stability index between two answer mixes, each a share per option.

    Symmetric in the sense that matters here (swapping the arguments gives the same number) and
    zero only when the two mixes agree. Options missing from either side count as absent, not as
    an error, so a mix that has stopped producing an option is measured rather than rejected.
    """
    keys = set(reference) | set(current)
    total = 0.0
    for k in keys:
        e = max(float(reference.get(k, 0.0)), _FLOOR)
        a = max(float(current.get(k, 0.0)), _FLOOR)
        total += (a - e) * math.log(a / e)
    return total


class Monitor:
    """Counts answers as they go by and says whether the calibration still fits them.

    ``reference`` is :attr:`vegaml.calibrate.Calibration.reference`: per question, the answer mix,
    mean confidence and review rate the calibration was fitted on. ``min_answers`` is how many
    answers a question needs before the monitor will call anything at all, since a mix drawn from
    twenty answers moves on its own.
    """

    def __init__(self, reference: Mapping[str, Mapping], min_answers: int = 50,
                 psi_watch: float = PSI_WATCH, psi_alert: float = PSI_ALERT,
                 confidence_drop: float = 0.05, review_rate_rise: float = 0.05,
                 unbound_rise: float = 0.05):
        self.reference = {k: dict(v) for k, v in reference.items()}
        self.min_answers = int(min_answers)
        self.psi_watch, self.psi_alert = float(psi_watch), float(psi_alert)
        self.confidence_drop = float(confidence_drop)
        self.review_rate_rise = float(review_rate_rise)
        self.unbound_rise = float(unbound_rise)
        self.counts: Dict[str, Dict] = {}

    @classmethod
    def from_calibration(cls, cal, **kw) -> "Monitor":
        """A monitor against what :func:`vegaml.calibrate.calibrate` recorded while fitting."""
        reference = getattr(cal, "reference", None)
        if not reference:
            raise ValueError("this calibration carries no reference mix, so there is nothing to "
                             "compare live traffic against. Refit it with a current vegaml "
                             "(calibrate() records the mix) or build Monitor(reference) by hand.")
        return cls(reference, **kw)

    # ------------------------------------------------------------------ counting
    def observe(self, result: Mapping) -> None:
        """Count one decide() result, or one bare ``{question_id: answer}`` mapping."""
        answers = result.get("answers") if "answers" in result else result
        if not isinstance(answers, Mapping):
            raise ValueError("expected a decide() result or a {question_id: answer} mapping")
        for qid, ans in answers.items():
            if not isinstance(ans, Mapping):
                continue
            c = self.counts.setdefault(qid, {"n": 0, "labels": {}, "confidence": 0.0,
                                             "review": 0, "routed": 0,
                                             "unbound": 0, "bounded": 0})
            chosen = self._chosen(ans)
            if chosen is None:
                continue
            c["n"] += 1
            c["labels"][chosen] = c["labels"].get(chosen, 0) + 1
            if ans.get("confidence") is not None:
                c["confidence"] += float(ans["confidence"])
            if ans.get("route") in ("auto", "review"):
                c["routed"] += 1
                c["review"] += int(ans["route"] == "review")
            if ans.get("unbound") is not None:
                c["bounded"] += 1
                c["unbound"] += int(bool(ans["unbound"]))

    @staticmethod
    def _chosen(ans: Mapping) -> Optional[str]:
        """Which option this answer picked, whichever answer type and readout produced it."""
        for key in ("choice", "score"):
            if ans.get(key) is not None:
                return str(ans[key])
        if ans.get("boolean") is not None:
            return "true" if float(ans["boolean"]) >= 0.5 else "false"
        probs = ans.get("probs")
        if isinstance(probs, Mapping) and probs:
            return str(max(probs, key=lambda k: probs[k])).lower()
        return None

    # ------------------------------------------------------------------ reporting
    def report(self) -> Dict:
        """Per question: the mix now, its distance from the reference, and what to do.

        ``status`` at the top is the worst of them, so one field decides whether to page anyone.
        """
        questions: Dict[str, Dict] = {}
        for qid in sorted(set(self.counts) | set(self.reference)):
            questions[qid] = self._one(qid)
        known = [q["status"] for q in questions.values() if q["status"] != "insufficient"]
        status = max(known, key=lambda s: _RANK[s]) if known else "insufficient"
        return {"status": status, "questions": questions,
                "answers_seen": sum(c["n"] for c in self.counts.values())}

    def _one(self, qid: str) -> Dict:
        ref = self.reference.get(qid)
        c = self.counts.get(qid)
        if ref is None:
            return {"status": "insufficient", "n": (c or {}).get("n", 0),
                    "reason": "no reference for %r: this question was not part of the calibration" % qid}
        if c is None or c["n"] == 0:
            return {"status": "insufficient", "n": 0, "reason": "no answers seen for %r yet" % qid}
        n = c["n"]
        mix = {k: v / n for k, v in c["labels"].items()}
        ref_mix = ref.get("predicted_mix", {})
        out: Dict = {"n": n, "predicted_mix": {k: round(v, 4) for k, v in sorted(mix.items())},
                     "reference_mix": {k: round(float(v), 4) for k, v in sorted(ref_mix.items())},
                     "psi": round(psi(ref_mix, mix), 4)}
        out["mean_confidence"] = round(c["confidence"] / n, 4)
        if ref.get("mean_confidence") is not None:
            out["reference_confidence"] = round(float(ref["mean_confidence"]), 4)
            out["confidence_delta"] = round(out["mean_confidence"] - out["reference_confidence"], 4)
        if c["routed"]:
            out["review_rate"] = round(c["review"] / c["routed"], 4)
            if ref.get("review_rate") is not None:
                out["reference_review_rate"] = round(float(ref["review_rate"]), 4)
                out["review_rate_delta"] = round(out["review_rate"] - out["reference_review_rate"], 4)
        if c.get("bounded"):
            out["unbound_rate"] = round(c["unbound"] / c["bounded"], 4)
            if ref.get("unbound_rate") is not None:
                out["reference_unbound_rate"] = round(float(ref["unbound_rate"]), 4)
                out["unbound_delta"] = round(out["unbound_rate"] - out["reference_unbound_rate"], 4)
        unseen = sorted(k for k in mix if k not in ref_mix)
        reasons = []
        status = "ok"

        def raise_to(level, why):
            nonlocal status
            reasons.append(why)
            if _RANK[level] > _RANK[status]:
                status = level

        if unseen:
            # An option the calibration never produced is not drift in degree, it is a different
            # question: no temperature was fitted for it and no threshold was measured on it.
            raise_to("recalibrate", "option(s) %s are being answered that the calibration never "
                                    "produced" % ", ".join(repr(u) for u in unseen[:5]))
        if out["psi"] >= self.psi_alert:
            raise_to("recalibrate", "answer mix has moved: PSI %.3f is at or above %.2f"
                                    % (out["psi"], self.psi_alert))
        elif out["psi"] >= self.psi_watch:
            raise_to("watch", "answer mix is drifting: PSI %.3f is at or above %.2f"
                              % (out["psi"], self.psi_watch))
        d = out.get("confidence_delta")
        if d is not None and d <= -2 * self.confidence_drop:
            raise_to("recalibrate", "mean confidence has fallen %.3f below the calibration's %.3f"
                                    % (-d, out["reference_confidence"]))
        elif d is not None and d <= -self.confidence_drop:
            raise_to("watch", "mean confidence is %.3f below the calibration's %.3f"
                              % (-d, out["reference_confidence"]))
        r = out.get("review_rate_delta")
        if r is not None and r >= 2 * self.review_rate_rise:
            raise_to("recalibrate", "review load has risen %.1f points over the %.1f%% the band was "
                                    "fitted at" % (100 * r, 100 * out["reference_review_rate"]))
        elif r is not None and r >= self.review_rate_rise:
            raise_to("watch", "review load is %.1f points over the %.1f%% the band was fitted at"
                              % (100 * r, 100 * out["reference_review_rate"]))
        u = out.get("unbound_delta")
        if u is not None and u >= 2 * self.unbound_rise:
            raise_to("recalibrate", "unbound rate has risen %.1f points over the %.1f%% these "
                                    "questions were fitted at, so the records are landing far "
                                    "from anything the engine knows"
                                    % (100 * u, 100 * out["reference_unbound_rate"]))
        elif u is not None and u >= self.unbound_rise:
            raise_to("watch", "unbound rate is %.1f points over the %.1f%% these questions were "
                              "fitted at" % (100 * u, 100 * out["reference_unbound_rate"]))
        if n < self.min_answers:
            # Measured and reported, but not acted on: a mix drawn from a handful of answers moves
            # by itself, and paging on that is how a drift alert gets switched off for good.
            out["status"] = "insufficient"
            out["reason"] = ("%d answers is under the %d this monitor will call a verdict on; the "
                             "numbers above are measured, not acted on" % (n, self.min_answers))
            return out
        out["status"] = status
        out["reason"] = "; ".join(reasons) if reasons else (
            "answer mix, confidence and review load are all where the calibration left them")
        return out

    # ------------------------------------------------------------------ persistence
    def save_state(self, path) -> None:
        """The counters, so a long-running watch survives a restart."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"reference": self.reference, "counts": self.counts}, f, indent=2,
                      sort_keys=True)

    @classmethod
    def load_state(cls, path, **kw) -> "Monitor":
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        mon = cls(payload["reference"], **kw)
        mon.counts = payload.get("counts", {})
        return mon

    def __repr__(self) -> str:
        return "Monitor(questions=%d, answers_seen=%d)" % (
            len(self.reference), sum(c["n"] for c in self.counts.values()))
