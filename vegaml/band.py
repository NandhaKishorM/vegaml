"""The review band: which answers a human sees, set by a threshold or by a precision target.

    from vegaml.band import Band

    band = Band.from_thresholds({"payment_change": 0.72}, default=0.6)   # no labels needed
    out = band.apply(v.decide(state, questions))
    out["answers"]["payment_change"]["route"]            # "auto" or "review"

    from vegaml.calibrate import calibrate                # or fitted to a target, on labels
    cal = calibrate(v, examples, questions, target_precision=0.97)
    cal.report["payment_change"]["band"]["review_rate"]   # what that target costs in review load

Abstention on the published calibration fires only when the conformal set comes out empty, which
on a question with k options needs qhat < 1 - 1/k. Across the six corpora we published it never
fired once, and on two of them it could not have fired at all. A band no operator can trigger is
not an escalation policy, so nothing here reads the conformal set. The band cuts on confidence, at
a line you set yourself or fit to the precision the automatic band has to hold.

Fitting to a target uses a one-sided 95% lower bound on the automatic band's precision rather than
the precision it happened to score, because the threshold is chosen on the same rows that score it.
Picking the best-looking cut on a small sample is how a 97% target ships as a 91% pilot. The lower
bound makes a narrow band pay for being narrow, and :func:`crossfit` reports what the same rule
scores on rows it did not see.
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = ["Band", "fit_threshold", "sweep", "crossfit", "wilson_lower", "ROUTES"]

ROUTES = ("auto", "review", "unbanded")

# One-sided normal quantiles, for the confidence levels anyone actually asks for.
_Z = {0.90: 1.2816, 0.95: 1.6449, 0.975: 1.9600, 0.99: 2.3263}

# A band narrower than this says nothing about its own precision, whatever it scored.
_MIN_AUTO = 10


def wilson_lower(hits: int, n: int, confidence: float = 0.95) -> float:
    """One-sided lower confidence bound on a success rate, by Wilson's interval.

    Wilson rather than the textbook normal interval because the band that matters is the one near
    precision 1.0, where the normal interval runs past 1 and, at hits == n, collapses to a width of
    zero: 12 for 12 would certify any target at all.
    """
    if n <= 0:
        return 0.0
    z = _Z.get(round(confidence, 3))
    if z is None:
        raise ValueError("confidence must be one of %s, got %r"
                         % (", ".join(str(c) for c in sorted(_Z)), confidence))
    p = hits / n
    z2 = z * z
    centre = p + z2 / (2 * n)
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4 * n * n))
    return max(0.0, (centre - half) / (1.0 + z2 / n))


def sweep(rows: Sequence[Tuple[float, bool]], confidence: float = 0.95) -> List[Dict]:
    """Every threshold worth considering, with what the automatic band does at each.

    ``rows`` are ``(confidence, was_correct)``, one per answered example. A threshold ``t`` sends
    every answer whose confidence is below ``t`` to review, so the candidates are 0.0 (nothing
    reviewed) and each distinct confidence in the rows. Returned lowest threshold first.
    """
    if not rows:
        raise ValueError("no rows")
    n = len(rows)
    candidates = sorted({0.0} | {float(c) for c, _ in rows})
    out = []
    for t in candidates:
        auto = [ok for c, ok in rows if c >= t]
        hits = sum(1 for ok in auto if ok)
        out.append({"threshold": round(t, 6), "n": n, "n_auto": len(auto),
                    "review_rate": round(1.0 - len(auto) / n, 4),
                    "precision": round(hits / len(auto), 4) if auto else None,
                    "precision_lb": round(wilson_lower(hits, len(auto), confidence), 4)})
    return out


def fit_threshold(rows: Sequence[Tuple[float, bool]], target_precision: Optional[float] = None,
                  review_rate: Optional[float] = None, max_review_rate: float = 1.0,
                  confidence: float = 0.95, min_auto: int = _MIN_AUTO) -> Dict:
    """The confidence line to review below, and what it is measured to do.

    Give ``target_precision`` and you get the widest automatic band whose precision holds at that
    target with ``confidence`` to spare, subject to ``max_review_rate``. Give ``review_rate``
    instead and you get the line that sends exactly that share to review, with its precision
    reported rather than targeted: a review budget, for a pilot that has a fixed number of
    reviewers and wants to know what they are buying.

    When no line can hold the target, nothing is quietly downgraded: ``reachable`` is False, the
    threshold goes above every observed confidence so every answer escalates, and
    ``best_achievable`` names the target that this much data can support. Lower the target, widen
    the review budget or label more examples; do not ship a band that does not hold.
    """
    if (target_precision is None) == (review_rate is None):
        raise ValueError("give exactly one of target_precision or review_rate")
    points = sweep(rows, confidence)
    top = max(c for c, _ in rows)

    if review_rate is not None:
        if not 0.0 <= review_rate < 1.0:
            raise ValueError("review_rate is a share of traffic in [0, 1), got %r" % (review_rate,))
        # The widest band that still reviews at least the requested share: a budget is a ceiling on
        # how many reviewers are needed, so overshooting it is the failure, not undershooting.
        allowed = [p for p in points if p["review_rate"] >= review_rate] or [points[-1]]
        best = min(allowed, key=lambda p: (p["review_rate"], p["threshold"]))
        return dict(best, reachable=True, target_precision=None, confidence=confidence,
                    requested_review_rate=review_rate)

    if not 0.0 < target_precision < 1.0:
        raise ValueError("target_precision is a rate in (0, 1), got %r" % (target_precision,))
    ok = [p for p in points
          if p["n_auto"] >= min_auto and p["review_rate"] <= max_review_rate
          and p["precision_lb"] >= target_precision]
    if ok:
        best = min(ok, key=lambda p: (p["review_rate"], p["threshold"]))
        return dict(best, reachable=True, target_precision=target_precision, confidence=confidence)
    feasible = [p for p in points if p["n_auto"] >= min_auto and p["review_rate"] <= max_review_rate]
    hint = max(feasible, key=lambda p: (p["precision_lb"], p["n_auto"])) if feasible else None
    return {"threshold": round(top + 1e-6, 6), "n": len(rows), "n_auto": 0, "review_rate": 1.0,
            "precision": None, "precision_lb": 0.0, "reachable": False,
            "target_precision": target_precision, "confidence": confidence,
            "best_achievable": hint,
            "reason": ("no confidence line holds precision %.3f at %d%% confidence on %d examples%s; "
                       "every answer goes to review until the target is lowered or more examples "
                       "are labelled"
                       % (target_precision, round(100 * confidence), len(rows),
                          "" if hint is None else (", the best being %.3f at a %.0f%% review rate"
                                                   % (hint["precision_lb"], 100 * hint["review_rate"]))))}


def crossfit(rows: Sequence[Tuple[float, bool]], folds: int = 5, **fit_kw) -> Optional[Dict]:
    """What the same fitting rule scores on rows it did not choose its threshold on.

    The threshold in :func:`fit_threshold` is chosen on the rows that then measure it, so its
    precision is in-sample however careful the bound. This refits the rule on each 4/5 of the rows
    and pools what it does on the held-out fifth, which costs no data and is the number to quote.
    Returns None when there are too few rows to split.
    """
    if len(rows) < 2 * folds:
        return None
    hits = auto = total = 0
    thresholds = []
    for f in range(folds):
        te = [r for i, r in enumerate(rows) if i % folds == f]
        tr = [r for i, r in enumerate(rows) if i % folds != f]
        if not te or not tr:
            continue
        t = fit_threshold(tr, **fit_kw)["threshold"]
        thresholds.append(t)
        for c, ok in te:
            total += 1
            if c >= t:
                auto += 1
                hits += int(ok)
    if not total or not thresholds:
        return None
    return {"folds": len(thresholds), "n": total, "n_auto": auto,
            "review_rate": round(1.0 - auto / total, 4),
            "precision": round(hits / auto, 4) if auto else None,
            "threshold_mean": round(sum(thresholds) / len(thresholds), 4),
            "threshold_spread": round(max(thresholds) - min(thresholds), 4)}


class Band:
    """A review line per question: below it the answer is a human's, above it the machine's.

    ``questions`` maps a question id to the dict :func:`fit_threshold` returns, or to anything
    carrying a ``threshold``. ``default`` covers questions with no line of their own; without one
    they come back ``route="unbanded"``, so a question nobody configured is never silently
    auto-approved on a policy it was not measured under.
    """

    def __init__(self, questions: Optional[Mapping[str, Mapping]] = None,
                 default: Optional[float] = None, report: Optional[Mapping] = None):
        self.questions: Dict[str, Dict] = {k: dict(v) for k, v in (questions or {}).items()}
        for qid, q in self.questions.items():
            if "threshold" not in q:
                raise ValueError("the band for %r names no threshold" % qid)
        self.default = None if default is None else float(default)
        self.report: Dict = dict(report or {})

    # ------------------------------------------------------------------ building
    @classmethod
    def from_thresholds(cls, thresholds: Mapping[str, float], default: Optional[float] = None) -> "Band":
        """A band from confidence lines you chose yourself, no labels involved."""
        return cls({qid: {"threshold": float(t), "source": "configured"}
                    for qid, t in thresholds.items()}, default=default)

    def threshold_for(self, qid: str) -> Optional[float]:
        q = self.questions.get(qid)
        return self.default if q is None else float(q["threshold"])

    # ------------------------------------------------------------------ routing
    def route(self, answer: Mapping, qid: str) -> Dict:
        """``{"route", "review", "review_threshold", "route_reason"}`` for one answer."""
        t = self.threshold_for(qid)
        if t is None:
            return {"route": "unbanded", "review": False, "review_threshold": None,
                    "route_reason": ("no review line is configured for %r, so no escalation policy "
                                     "covers this answer" % qid)}
        conf = answer.get("confidence")
        if conf is None:
            return {"route": "unbanded", "review": False, "review_threshold": t,
                    "route_reason": "this answer carries no confidence to compare against the line"}
        conf = float(conf)
        review = conf < t
        return {"route": "review" if review else "auto", "review": review,
                "review_threshold": round(t, 6),
                "route_reason": ("confidence %.3f is %s the %.3f review line"
                                 % (conf, "below" if review else "at or above", t))}

    def apply(self, result: Mapping) -> Dict:
        """A copy of a decide() result with every answer routed, plus a ``band`` summary."""
        out = dict(result)
        counts = {"auto": 0, "review": 0, "unbanded": 0}
        unbanded: List[str] = []
        for key in ("answers", "ttt"):
            answers = result.get(key)
            if not isinstance(answers, Mapping):
                continue
            routed = {}
            for qid, ans in answers.items():
                decided = self.route(ans, qid)
                counts[decided["route"]] += 1
                if decided["route"] == "unbanded" and qid not in unbanded:
                    unbanded.append(qid)
                routed[qid] = dict(ans, **decided)
            out[key] = routed
        out["band"] = {"auto": counts["auto"], "review": counts["review"],
                       "unbanded": unbanded, "default": self.default}
        return out

    # ------------------------------------------------------------------ persistence
    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"questions": self.questions, "default": self.default,
                       "report": self.report}, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path) -> "Band":
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return cls(payload.get("questions"), payload.get("default"), payload.get("report"))

    def __repr__(self) -> str:
        return "Band(questions=%d, default=%s)" % (len(self.questions), self.default)
