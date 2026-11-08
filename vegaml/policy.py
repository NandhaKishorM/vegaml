"""Trained policies: load one, check it, decide with it, and get a route on every answer.

    import vegaml

    policy = vegaml.load_policy("acme-policy.zip")    # Embedding Studio's download, or a folder
    print(policy.verified)                            # {"status": "signed", "signed_by": ...}
    out = policy.decide("I was charged twice")        # the policy's own questions
    out["route"]                                      # "auto" or "review", for the whole record
    out["answers"]["team"]["route_reason"]            # why, in a sentence

    policy = vegaml.fit_policy(vegaml.load(), "labels.csv", questions)   # or fit one yourself
    policy.save("my-policy/")

A policy is the trained layer on top of the frozen reader: one small head per question
(``heads.npz``), the calibration those heads were measured with (``calibration.json``, review
band included) and the questions they answer (``questions.json``). Studio adds a manifest
(``manifest.json``) and a signature (``manifest.sig``).

**Verified before anything is installed.** :func:`load_policy` checks, in this order:

1. every file the manifest names has exactly the SHA-256 it states, so a changed or swapped
   file is refused by name;
2. a manifest that names a signing key carries a signature that verifies against a key this
   release trusts (:data:`TRUSTED_KEYS`), so a policy cannot be re-signed by anybody else;
3. the engine loaded is the one the heads were trained on: heads fitted on one reader's
   features are meaningless on another's.

Any failure raises :class:`PolicyError`. ``verify="signed"`` also refuses an unsigned policy;
``verify=False`` skips all three. A folder written by :meth:`Policy.save` carries a manifest with
hashes and no key: it is checked for integrity and reported as unsigned.

**The route.** Every answer leaves :meth:`Policy.decide` marked ``"auto"`` or ``"review"``, and
the record is ``"auto"`` only when every answer is. Four things send an answer to a person, and
none of them is a confidence number someone had to guess:

* ``outbound``: the record lies outside what the policy learned. The routing head (the first
  question) measures the share of the record's features its projection does not hold; past the
  limit measured when the policy was trained, every answer of the record goes to review.
* not calibrated: a question the policy's calibration does not cover, or one asked with an
  option the policy was never trained on (``calibrated: False``). Its numbers are not ones the
  calibration has measured.
* abstain: the conformal set at the policy's ``alpha`` is empty or holds more than one option.
* the band: the policy's review line for that question, placed where precision held on its
  calibration rows.

Pass ``review_below`` to :meth:`Policy.decide` to draw your own line instead: the line then
decides with ``outbound``, and ``abstain`` and ``calibrated`` are reported without routing.

**Gate on ``route``, never on ``confidence``.** A fitted head once returned 1.000 on every one of
sixteen held-out records, both of its mistakes included, while the route on the same records was
right, because the route comes from the conformal set, the band and the outbound check rather
than from the top probability.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
import os
import re
import zipfile
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from . import label_ids
from .ttt import TTTHead

__all__ = ["Policy", "PolicyError", "load_policy", "fit_policy", "save_policy", "load_rows",
           "min_rows", "normalise_questions", "verify", "read", "manifest_for", "residual",
           "TRUSTED_KEYS", "HEAD_ARRAYS"]

#: Public keys whose signatures vegaml accepts, by key id (the first 16 hex characters of the
#: SHA-256 of the raw public key). Embedding Space signs every policy Studio exports with the
#: private half, which never leaves its servers. The same key is served at
#: ``/v1/policy-signing-key`` on the Studio API.
TRUSTED_KEYS: Dict[str, str] = {
    "589c4677fa61c61c": "dZMFedqlr5VlF5yt+sT0L86hrmprsLPCF6CLyIaA38U=",
}

FILES = ("heads.npz", "calibration.json", "questions.json", "manifest.json", "manifest.sig")

#: What a fitted head is, in full: standardisation, projection and the linear readout.
HEAD_ARRAYS = ("mu", "sd", "c", "P", "s", "W", "b")

#: Rows held back from the heads so the calibration is measured on rows they never saw.
CALIBRATION_SHARE = 0.25
#: The calibration needs ten rows to say anything.
MIN_CALIBRATION_ROWS = 10


class PolicyError(ValueError):
    """The policy failed verification; the message names what and why."""


# ----------------------------------------------------------------------------- the files
def key_id(public_key_b64: str) -> str:
    return hashlib.sha256(base64.b64decode(public_key_b64)).hexdigest()[:16]


def read(path) -> Dict[str, bytes]:
    """The policy's files as bytes, from a folder or a ``.zip`` (files at the root of the archive
    or inside one top-level folder, as Studio's download has them). Nothing is extracted."""
    path = os.fspath(path)
    out: Dict[str, bytes] = {}
    if os.path.isdir(path):
        for name in FILES:
            p = os.path.join(path, name)
            if os.path.isfile(p):
                with open(p, "rb") as f:
                    out[name] = f.read()
    elif os.path.isfile(path) and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                name = info.filename.rsplit("/", 1)[-1]
                if name in FILES and info.filename.count("/") <= 1:
                    if name in out:
                        raise PolicyError("%s holds more than one %s" % (path, name))
                    out[name] = z.read(info)
    elif not os.path.exists(path):
        raise PolicyError("no policy at %s" % path)
    else:
        raise PolicyError("%s is neither a policy folder nor a .zip" % path)
    if "heads.npz" not in out:
        raise PolicyError("%s has no heads.npz; it is not a trained policy" % path)
    return out


def manifest_for(files: Mapping[str, bytes], **fields: Any) -> bytes:
    """A manifest naming the SHA-256 of each file, plus whatever fields are given. Serialised once
    and kept as bytes: a signature covers these exact bytes, so nobody re-serialises them."""
    body = dict(fields)
    body["files"] = {n: hashlib.sha256(b).hexdigest() for n, b in sorted(files.items())
                     if n not in ("manifest.json", "manifest.sig")}
    return json.dumps(body, indent=2, sort_keys=True).encode("utf-8")


def verify(files: Mapping[str, bytes], trusted: Optional[Mapping[str, str]] = None,
           require_signature: bool = False) -> Dict[str, Any]:
    """Check the manifest's hashes and signature. Returns what was verified; raises PolicyError."""
    trusted = TRUSTED_KEYS if trusted is None else trusted
    raw = files.get("manifest.json")
    if raw is None:
        if require_signature:
            raise PolicyError("this policy has no manifest, so its origin cannot be verified")
        return {"status": "unverified", "reason": "no manifest"}
    try:
        manifest = json.loads(raw)
    except ValueError as e:
        raise PolicyError("manifest.json is not valid JSON: %s" % e) from e
    if not isinstance(manifest, dict):
        raise PolicyError("manifest.json is not a JSON object")
    hashes = manifest.get("files")
    if hashes is None:
        hashes = {}
    if not isinstance(hashes, dict) or not all(
            isinstance(name, str) and isinstance(want, str) for name, want in hashes.items()):
        raise PolicyError("manifest.json's files is not a map of file names to SHA-256 hashes")
    for name, want in hashes.items():
        have = files.get(name)
        if have is None:
            raise PolicyError("the manifest lists %s but the policy does not contain it" % name)
        got = hashlib.sha256(have).hexdigest()
        if got != want:
            raise PolicyError("%s has been changed: its SHA-256 is %s, the manifest says %s"
                              % (name, got[:16], want[:16]))
    for name in ("heads.npz", "calibration.json", "questions.json"):
        if name in files and name not in hashes:
            raise PolicyError("%s is in the policy but not in its manifest" % name)
    kid = manifest.get("key_id")
    if not kid:
        if require_signature:
            raise PolicyError("this policy is not signed, so its origin cannot be verified")
        return {"status": "integrity", "signed_by": None, "manifest": manifest}
    sig = files.get("manifest.sig")
    if sig is None:
        raise PolicyError("the manifest names signing key %s but manifest.sig is missing" % kid)
    pub = trusted.get(kid)
    if pub is None:
        raise PolicyError("signed with key %s, which this release of vegaml does not trust; "
                          "upgrade vegaml, or the policy did not come from Embedding Space" % kid)
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as e:
        raise PolicyError("verifying a signed policy needs the cryptography package: "
                          "pip install cryptography") from e
    try:
        Ed25519PublicKey.from_public_bytes(base64.b64decode(pub)).verify(
            base64.b64decode(sig.strip()), raw)
    except (InvalidSignature, ValueError) as e:
        raise PolicyError("the signature does not match the manifest: the policy was altered "
                          "after Embedding Space signed it, or it was signed by someone else") from e
    return {"status": "signed", "signed_by": "Embedding Space (key %s)" % kid, "manifest": manifest}


def expected_engine(files: Mapping[str, bytes], manifest: Mapping[str, Any]) -> Optional[str]:
    """The engine SHA-256 the policy was trained on: the manifest's, else the calibration's own."""
    if manifest.get("engine_sha256"):
        return str(manifest["engine_sha256"])
    try:
        cal = json.loads(files.get("calibration.json") or b"{}")
    except ValueError:
        return None
    return ((cal.get("provenance") or {}).get("engine_sha256")) or None


def check_engine(expected: Optional[str], actual: Optional[str]) -> None:
    if expected and actual and expected != actual:
        raise PolicyError("this policy was trained on engine %s... but the loaded engine is %s...; "
                          "load the reader it was trained for (the manifest names it)"
                          % (expected[:12], actual[:12]))


def _manifest(files: Mapping[str, bytes]) -> Dict[str, Any]:
    try:
        m = json.loads(files.get("manifest.json") or b"{}")
    except ValueError:
        return {}
    return m if isinstance(m, dict) else {}


# ----------------------------------------------------------------------------- questions
def normalise_questions(questions: Mapping[str, Mapping]) -> Dict[str, Dict]:
    """Questions in vegaml's spelling, checked, so a malformed one is refused by name.

    Accepts laya's spelling as well, which is what a Studio download's ``questions.json`` uses:
    laya's yes/no type is ``noul`` where vegaml's is ``boolean``, and laya names a yes/no's two
    sides in ``labels`` where vegaml uses ``criteria``. A ``choice`` given as a list of options is
    turned into vegaml's ``{option: description}`` map.
    """
    if not isinstance(questions, Mapping):
        raise ValueError("questions must be a map of question id to question, got %s"
                         % type(questions).__name__)
    out: Dict[str, Dict] = {}
    for qid, q in questions.items():
        if not isinstance(q, Mapping):
            raise ValueError("question %r must be a map with type, instructions and criteria" % (qid,))
        t = q.get("type")
        if t == "noul":
            t = "boolean"
        if t not in ("choice", "score", "boolean"):
            raise ValueError("question %r has type %r; it must be choice, score or boolean" % (qid, t))
        ins = q.get("instructions")
        if not isinstance(ins, str) or not ins.strip():
            raise ValueError("question %r needs its instructions as text" % (qid,))
        vq: Dict[str, Any] = {"type": t, "instructions": ins}
        crit = q.get("criteria")
        if t == "boolean":
            lab = q.get("labels")
            if isinstance(crit, Mapping):
                vq["criteria"] = dict(crit)
            elif isinstance(lab, Mapping):
                vq["criteria"] = {"false": lab.get("false", "no"), "true": lab.get("true", "yes")}
        elif t == "score":
            if not isinstance(crit, Sequence) or isinstance(crit, str) or len(crit) < 2:
                raise ValueError("score question %r needs criteria as a list of at least two levels"
                                 % (qid,))
            vq["criteria"] = list(crit)
        else:
            if isinstance(crit, Mapping):
                vq["criteria"] = dict(crit)
            elif isinstance(crit, Sequence) and not isinstance(crit, str):
                vq["criteria"] = {str(c): None for c in crit}
            else:
                raise ValueError("choice question %r needs criteria: its options" % (qid,))
            if len(vq["criteria"]) < 2:
                raise ValueError("choice question %r needs at least two options" % (qid,))
        out[str(qid)] = vq
    return out


def _finite(value: Any, where: str) -> float:
    """One number off an answer, refused rather than rounded when it is not one.

    ``float()`` reads True as 1.0 and waves nan and inf through, and every comparison against a
    NaN is False, so a NaN confidence used to clear a review line untouched and leave as "auto".
    A corrupt distribution is not a decision, and rounding it is how it becomes one."""
    if isinstance(value, bool):
        raise ValueError("%s is %r: answers must carry finite numbers, not a bool" % (where, value))
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("%s is %r: answers must carry finite numbers" % (where, value)) from exc
    if not math.isfinite(number):
        raise ValueError("%s is %r: answers must carry finite numbers" % (where, value))
    return number


def _check_answers(result: Mapping) -> None:
    for qid, a in (result.get("answers") or {}).items():
        for k, v in (a.get("probs") or {}).items():
            _finite(v, "the probability of %r on %r" % (k, qid))
        if a.get("confidence") is not None:
            _finite(a["confidence"], "the confidence on %r" % (qid,))


# ----------------------------------------------------------------------------- heads
def residual(head: TTTHead, x) -> float:
    """The share of a record's standardised features the head's projection does not hold.

    Near 0 for a record like the ones the head was fitted on; it grows as the record moves into
    directions the head never saw. This is the outbound measure."""
    Z = (np.asarray(x, dtype=np.float32).reshape(1, -1) - head.mu) / head.sd
    Zc = Z - head.c
    total = float((Zc ** 2).sum())
    if total <= 0:
        return 0.0
    inside = float(((Zc @ head.P) ** 2).sum())
    return max(0.0, 1.0 - inside / total)


class GuardedHead(TTTHead):
    """A head that remembers how far outside its space the last record it read was."""

    last_residual: Optional[float] = None

    def predict(self, x, allowed=None):
        self.last_residual = residual(self, x)
        return super().predict(x, allowed)


def _head_arrays(head: TTTHead) -> Dict[str, np.ndarray]:
    return {name: np.asarray(getattr(head, name)) for name in HEAD_ARRAYS}


# ----------------------------------------------------------------------------- calibration
def _stale(result: Mapping, cal, questions: Mapping) -> Dict[str, List[str]]:
    """Questions asked with an option the calibration was never fitted on, with those options.

    Adding an option to a trained question leaves the calibration naming the old option set, and
    applying it raised "probability key names no option" on every decision until a retrain."""
    out: Dict[str, List[str]] = {}
    if cal is None:
        return out
    for qid in (result.get("answers") or {}):
        fitted = cal.questions.get(qid)
        if fitted is None or qid not in questions:
            continue
        known = {str(o).lower() for o in fitted["labels"]}
        new = [o for o in label_ids(questions[qid]) if str(o).lower() not in known]
        if new:
            out[qid] = new
    return out


def _pad(result: Mapping, cal) -> Dict:
    """Zero-fill any fitted option an answer never mentioned (the row did not offer it):
    Calibration.apply indexes every label it was fitted on."""
    out = dict(result)
    out["answers"] = {}
    for qid, ans in result["answers"].items():
        fitted = cal.questions.get(qid)
        if fitted is None or not ans.get("probs"):
            out["answers"][qid] = ans
            continue
        have = {str(k).lower() for k in ans["probs"]}
        missing = [o for o in fitted["labels"] if str(o).lower() not in have]
        if missing:
            ans = dict(ans, probs=dict(ans["probs"], **{o: 0.0 for o in missing}))
        out["answers"][qid] = ans
    return out


def _apply_calibration(result: Mapping, cal, stale: Mapping) -> Dict:
    """The calibration on every answer it covers; stale answers pass through untouched."""
    answers = result["answers"]
    sub = dict(result, answers={q: a for q, a in answers.items() if q not in stale})
    done = cal.apply(_pad(sub, cal))
    merged = dict(done)
    merged["answers"] = {q: (answers[q] if q in stale else done["answers"][q]) for q in answers}
    return merged


# ----------------------------------------------------------------------------- the route
def _pair(a: float, b: float) -> Tuple[str, str]:
    """Two numbers printed with just enough digits to tell them apart. A saturated head puts the
    confidence at 0.99995 and its bar at 0.99996, and "1.000 is below 1.000" explains nothing."""
    for digits in range(3, 8):
        x, y = "%.*f" % (digits, a), "%.*f" % (digits, b)
        if x != y:
            return x, y
    return "%.7f" % a, "%.7f" % b


def settle(result: Dict, review_below: Optional[float] = None,
           stale: Optional[Mapping[str, List[str]]] = None) -> Dict:
    """``abstain``, ``route`` and ``route_reason`` on every answer, and ``route`` on the record.

    In place, and returned. See the module docstring for the four reasons."""
    stale = stale or {}
    line = None if review_below is None else _finite(review_below, "review_below")
    for qid, a in (result.get("answers") or {}).items():
        if not isinstance(a, dict):
            continue
        pset = a.get("prediction_set")
        if pset is not None or a.get("abstain") is not None:
            a["abstain"] = not (bool(pset) and len(pset) == 1 and not a.get("abstain"))
        reasons = []
        if a.get("outbound"):
            reasons.append("the record is unlike anything the policy learned from")
        if line is not None:
            conf = a.get("confidence")
            if conf is None or float(conf) < line:
                reasons.append("confidence %s is below your line %.3f"
                               % ("unknown" if conf is None else "%.3f" % float(conf), line))
        else:
            if qid in stale:
                reasons.append("options changed since the policy was trained (%s not in it), so "
                               "it is not calibrated; train it again" % ", ".join(stale[qid]))
            elif a.get("calibrated") is False:
                reasons.append("the policy's calibration does not cover this question; "
                               "train it to calibrate")
            if a.get("abstain"):
                if pset and len(pset) > 1:
                    reasons.append("it cannot narrow the answer to one option (%s)"
                                   % " or ".join(str(x) for x in pset))
                else:
                    conf, bar = a.get("confidence"), a.get("auto_at")
                    if isinstance(conf, (int, float)) and isinstance(bar, (int, float)):
                        c, b = _pair(float(conf), float(bar))
                        reasons.append("confidence %s is below %s, the bar the policy's held-out "
                                       "rows set for an automatic answer" % (c, b))
                    else:
                        reasons.append("confidence %s is below the bar the policy's held-out rows "
                                       "set for an automatic answer"
                                       % ("%.3f" % conf if isinstance(conf, (int, float)) else "unknown"))
            if a.get("route") == "review" and a.get("route_reason"):
                conf, line_t = a.get("confidence"), a.get("review_threshold")
                if isinstance(conf, (int, float)) and isinstance(line_t, (int, float)):
                    c, t = _pair(float(conf), float(line_t))
                    reasons.append("band: confidence %s is below the %s review line" % (c, t))
                else:
                    reasons.append("band: " + str(a["route_reason"]))
        if reasons:
            a["route"] = "review"
            a["route_reason"] = "; ".join(reasons)
        else:
            a["route"] = "auto"
            if line is not None:
                a["route_reason"] = "confidence is at or above your line %.3f" % line
            elif not a.get("route_reason") or "no review line" in str(a.get("route_reason")):
                a["route_reason"] = "calibrated, one option in the conformal set, inside the policy's space"
    result["route"] = record_route(result.get("answers") or {})
    return result


def record_route(answers: Mapping) -> str:
    """"auto" only when every answer is: a record is what a person picks up."""
    return "review" if any(isinstance(a, Mapping) and a.get("route") == "review"
                           for a in answers.values()) else "auto"


# ----------------------------------------------------------------------------- the policy
class Policy:
    """A model with a trained policy installed. Built by :func:`load_policy` or :func:`fit_policy`."""

    def __init__(self, model, calibration=None, questions: Optional[Mapping] = None,
                 novelty: Optional[Tuple[str, float]] = None,
                 verified: Optional[Dict[str, Any]] = None, source: Optional[str] = None,
                 manifest: Optional[Dict[str, Any]] = None):
        self.model = model
        self.calibration = calibration
        self.questions = normalise_questions(questions) if questions else {}
        self.novelty = novelty
        self.verified = verified or {"status": "not loaded from a file"}
        self.source = source
        self.manifest = manifest or {}
        self.fit_report: Dict[str, Any] = {}

    def __repr__(self) -> str:
        return "Policy(%s, questions=%s, calibrated=%s, outbound=%s)" % (
            self.source or "fitted here", sorted(self.questions), self.calibration is not None,
            None if self.novelty is None else round(self.novelty[1], 4))

    @property
    def heads(self) -> List[str]:
        return sorted(self.model.heads)

    def decide(self, state: Any = None, questions: Optional[Mapping] = None, *, image: Any = None,
               audio: Any = None, video: Any = None, review_below: Optional[float] = None,
               alpha: Optional[str] = None, **kw) -> Dict:
        """Answer the questions about one record, with a route on every answer and on the record.

        One of ``state`` (text or a JSON-able object), ``image``, ``audio`` or ``video``.
        ``questions`` defaults to the policy's own. ``review_below`` replaces the policy's own
        routing with your line (outbound still applies). Other keywords go to the model's decide
        (``views``, ``asr``, ``language``, ``sampling_rate``, ``frames``).

        The heads and the calibration were trained on text. A picture, or a clip decided from its
        frames, is answered by the vision tower's zero-shot readout, which they do not cover: those
        answers come back ``calibrated: False`` and go to review unless you give ``review_below``.
        """
        given = sum(x is not None for x in (state, image, audio, video))
        if given != 1:
            raise ValueError("decide takes exactly one of state, image, audio or video; got %d" % given)
        qs = normalise_questions(questions) if questions is not None else self.questions
        if not qs:
            raise ValueError("this policy carries no questions; pass questions=")
        a = alpha or (str(self.calibration.alpha) if self.calibration is not None else "0.1")
        routing = self.novelty[0] if self.novelty else None
        head = self.model.heads.get(routing, (None,))[0] if routing else None
        if head is not None:
            head.last_residual = None
        if video is not None:
            raw = self.model.decide_video(video, qs, alpha=a, **kw)
            modality = "video"
        elif image is not None:
            raw = self.model.decide_image(image, qs, alpha=a)
            modality = "image"
        elif audio is not None:
            raw = self.model.decide_audio(audio, qs, alpha=a, **kw)
            modality = "audio"
        else:
            raw = self.model.decide(state, qs, alpha=a, **kw)
            modality = "text"
        _check_answers(raw)
        pixels = modality == "image" or (raw.get("video") or {}).get("path") == "frames"
        stale: Dict[str, List[str]] = {}
        if self.calibration is not None and not pixels:
            stale = _stale(raw, self.calibration, qs)
            raw = _apply_calibration(raw, self.calibration, stale)
        covered = set(self.calibration.questions) if self.calibration is not None else set()
        for qid, ans in raw["answers"].items():
            ans["calibrated"] = bool(not pixels and qid in covered and qid not in stale)
            if ans["calibrated"]:
                # the confidence an answer needs for its conformal set to hold it at all
                ans["auto_at"] = round(1.0 - float(self.calibration.questions[qid]["qhat"]), 6)
        out = dict(raw, modality=modality)
        if head is not None and not pixels and head.last_residual is not None:
            r = float(head.last_residual)
            out["outbound"] = {"residual": round(r, 4), "limit": round(self.novelty[1], 4),
                               "question": routing, "outside": r > self.novelty[1]}
            if r > self.novelty[1]:
                for ans in out["answers"].values():
                    ans["outbound"] = True
        settle(out, review_below=review_below, stale=stale)
        if self.calibration is not None:
            out["calibration_version"] = self.calibration.version()
        return out

    def thresholds(self) -> Dict[str, Any]:
        """What decides ``route`` right now, per question, so nothing about it is hidden."""
        cal = self.calibration
        out: Dict[str, Any] = {
            "alpha": None if cal is None else cal.alpha,
            "outbound": None if not self.novelty else {"question": self.novelty[0],
                                                      "limit": self.novelty[1]},
            "band": None if cal is None or cal.band is None else
            {"default": cal.band.default, "questions": cal.band.questions},
            "calibration": {} if cal is None else
            {q: {k: v for k, v in c.items() if k != "labels"} for q, c in cal.questions.items()},
            "heads": {q: {"cv_accuracy": cv, "serving": self.model._head_worth_using(q, self.questions[q])
                          if q in self.questions else False}
                      for q, (h, cv) in self.model.heads.items()},
        }
        return out

    def save(self, path) -> Dict[str, Any]:
        """The policy to a folder that :func:`load_policy` reads. See :func:`save_policy`."""
        return save_policy(self.model, path, calibration=self.calibration, questions=self.questions,
                           novelty=self.novelty)


# ----------------------------------------------------------------------------- loading
def _reader_for(manifest: Mapping[str, Any], meta: Mapping[str, Any]) -> Optional[str]:
    """What to hand :func:`vegaml.load` for the reader a policy was trained on."""
    reader = str(manifest.get("reader") or meta.get("model") or "")
    if re.search(r"4b", reader, re.I):
        return "4b"
    if not reader or reader in ("nandakishorm/vega-U", "0.8b", "True"):
        return None
    if os.path.isdir(reader):
        return reader
    return None


def load_policy(path, model=None, verify: Union[bool, str] = True, trusted=None,
                **load_kw) -> Policy:
    """Install a trained policy from a folder or Studio's downloaded ``.zip``, verified first.

    ``model`` is a loaded :class:`vegaml.Vega`; without one, the reader the policy names is
    loaded (``load_kw`` go to :func:`vegaml.load`, e.g. ``device="cpu"``). ``verify=True``
    checks hashes, signature and engine; ``verify="signed"`` also refuses an unsigned policy;
    ``verify=False`` checks nothing. Returns a :class:`Policy`; ``policy.verified`` says what
    was checked.
    """
    from .calibrate import Calibration

    files = read(path)
    z = np.load(io.BytesIO(files["heads.npz"]), allow_pickle=False)
    try:
        meta = json.loads(str(z["meta"]))
    except (KeyError, ValueError) as e:
        raise PolicyError("heads.npz carries no readable meta; it is not a trained policy") from e
    report: Dict[str, Any] = {"status": "skipped"}
    if verify:
        report = verify_files(files, trusted=trusted, require_signature=(verify == "signed"))
    manifest = report.pop("manifest", None) or _manifest(files)
    if model is None:
        from . import load as _load

        model = _load(_reader_for(manifest, meta), **load_kw)
    if verify:
        try:
            actual = model.engine.base_sha256()
        except Exception:  # an engine that cannot hash itself is reported, not guessed
            actual = None
        expected = expected_engine(files, manifest)
        check_engine(expected, actual)
        report["engine"] = "matches" if (expected and actual) else "not checked"
    limits: Dict[str, float] = {}
    for qid, m in (meta.get("heads") or {}).items():
        head = GuardedHead(m["labels"], k=m.get("k", 128), l2=m.get("l2", 1e-2))
        try:
            for name in HEAD_ARRAYS:
                setattr(head, name, z["%s::%s" % (qid, name)])
        except KeyError as e:
            raise PolicyError("heads.npz has no %s for question %r" % (e, qid)) from e
        head.ready = True
        head.engine_accuracy = m.get("engine_accuracy")
        model.heads[qid] = (head, m.get("cv"))
        if m.get("novelty_max") is not None:
            limits[qid] = _finite(m["novelty_max"], "the off-topic limit of %r" % qid)
    cal = None
    if "calibration.json" in files:
        cal = Calibration.from_payload(json.loads(files["calibration.json"]))
    questions = json.loads(files["questions.json"]) if "questions.json" in files else {}
    # The routing question is the policy's first; the limit is measured on its head.
    order = list(questions) or list(meta.get("heads") or {})
    routing = next((q for q in order if q in limits), None)
    return Policy(model, calibration=cal, questions=questions,
                  novelty=(routing, limits[routing]) if routing else None,
                  verified=report, source=os.fspath(path), manifest=manifest)


verify_files = verify


def save_policy(model, path, calibration=None, questions: Optional[Mapping] = None,
                novelty: Optional[Tuple[str, float]] = None) -> Dict[str, Any]:
    """The model's fitted heads, the calibration and the questions to a folder, with a manifest
    of hashes (unsigned) so :func:`load_policy` can tell if the folder was changed."""
    os.makedirs(path, exist_ok=True)
    arrays, meta = {}, {"heads": {}, "format": "vegaml-policy/1"}
    try:
        meta["model"] = model.engine.cfg.get("backbone_name")
    except Exception:
        meta["model"] = None
    for qid, (head, cv) in model.heads.items():
        if not getattr(head, "ready", False):
            continue
        for name, arr in _head_arrays(head).items():
            arrays["%s::%s" % (qid, name)] = arr
        meta["heads"][qid] = {"labels": [str(x) for x in head.labels], "k": int(head.k),
                              "l2": float(head.l2), "cv": None if cv is None else float(cv),
                              "engine_accuracy": head.engine_accuracy}
        if novelty and novelty[0] == qid:
            meta["heads"][qid]["novelty_max"] = float(novelty[1])
    arrays["meta"] = np.array(json.dumps(meta))
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    files = {"heads.npz": buf.getvalue()}
    if calibration is not None:
        files["calibration.json"] = json.dumps(calibration.to_payload(), indent=2,
                                               sort_keys=True).encode("utf-8")
    if questions:
        files["questions.json"] = json.dumps(normalise_questions(questions), indent=2).encode("utf-8")
    try:
        engine = model.engine.base_sha256()
    except Exception:
        engine = None
    files["manifest.json"] = manifest_for(files, format="vegaml-policy/1", engine_sha256=engine,
                                          reader=meta["model"])
    for name, blob in files.items():
        with open(os.path.join(path, name), "wb") as f:
            f.write(blob)
    return {"path": os.fspath(path), "heads": sorted(meta["heads"]),
            "calibrated": calibration is not None, "outbound": novelty}


# ----------------------------------------------------------------------------- fitting
def load_rows(path, questions: Optional[Mapping] = None) -> List[Dict[str, Any]]:
    """Labelled rows from a ``.csv`` or ``.jsonl``.

    A CSV has a ``state`` column and one column per question id holding that row's answer (an
    option id for ``choice``, a level index for ``score``, yes/no or true/false for
    ``boolean``); an empty cell means the row carries no answer for that question. JSONL rows are
    ``{"state", "gold"}`` (``"questions"`` optional, per row)."""
    path = os.fspath(path)
    if path.lower().endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "state" not in reader.fieldnames:
            raise ValueError("%s needs a `state` column; the other columns are question ids" % path)
        cols = [c for c in reader.fieldnames if c != "state"]
        if questions:
            unknown = [c for c in cols if c not in questions]
            if unknown:
                raise ValueError("column(s) %s in %s match no question id in %s"
                                 % (unknown, path, sorted(questions)))
        rows = []
        for rec in reader:
            gold = {c: rec[c].strip() for c in cols if (rec.get(c) or "").strip()}
            if (rec.get("state") or "").strip() and gold:
                rows.append({"state": rec["state"], "gold": gold})
    return rows


def gold_label(q: Mapping, gold: Any) -> str:
    """A row's answer as the label a head is fitted on, or a ValueError naming why it cannot be.

    The quiet failure this refuses: a yes/no written as 1.0 reads as the string "1.0", which is
    not a true-word, so the row landed on the false side of its own label."""
    if isinstance(gold, Mapping):
        if "label" in gold:
            gold = gold["label"]
        elif gold.get("probabilities"):
            gold = max(gold["probabilities"], key=gold["probabilities"].get)
    t = q["type"]
    if t == "boolean":
        if isinstance(gold, bool):
            return "true" if gold else "false"
        if isinstance(gold, (int, float)) and gold in (0, 1):
            return "true" if gold == 1 else "false"
        if isinstance(gold, str):
            s = gold.strip().lower()
            if s in ("true", "yes", "1", "t", "y"):
                return "true"
            if s in ("false", "no", "0", "f", "n"):
                return "false"
        raise ValueError("a boolean answer must be true/false, yes/no or 0/1, got %r" % (gold,))
    if t == "score":
        crit = list(q["criteria"])
        if isinstance(gold, bool):
            raise ValueError("a score answer is a level index or a level description, not %r" % (gold,))
        if isinstance(gold, str) and not gold.strip().isdigit():
            if gold not in crit:
                raise ValueError("a score answer must be a level index in [0, %d] or one of %r, got %r"
                                 % (len(crit) - 1, crit, gold))
            return str(crit.index(gold))
        try:
            level = float(gold)
        except (TypeError, ValueError, OverflowError):
            level = None
        if level is None or not math.isfinite(level) or not level.is_integer() \
                or not 0 <= level <= len(crit) - 1:
            raise ValueError("a score answer must be a whole level index in [0, %d], got %r"
                             % (len(crit) - 1, gold))
        return str(int(level))
    labels = label_ids(q)
    if str(gold) in labels:
        return str(gold)
    raise ValueError("a choice answer must be one of the question's options %r, got %r" % (labels, gold))


def _examples(rows: Any, questions: Optional[Mapping]) -> Tuple[list, Dict]:
    if isinstance(rows, (str, os.PathLike)):
        rows = load_rows(rows, questions)
    examples, sheet = [], {}
    for r in rows:
        qs = normalise_questions(r.get("questions") or questions or {})
        if not qs:
            raise ValueError("a row has no questions and none were given")
        sheet.update(qs)
        gold = r.get("gold") if r.get("gold") is not None else r.get("expected")
        if not isinstance(gold, Mapping):
            raise ValueError("each row needs `gold` as {question_id: answer}")
        unknown = [qid for qid in gold if qid not in qs]
        if unknown:
            raise ValueError("answer key(s) %s match no question id in %s" % (unknown, sorted(qs)))
        labels = {qid: gold_label(qs[qid], g) for qid, g in gold.items()}
        if labels:
            examples.append((r["state"], labels, qs))
    return examples, sheet


def min_rows(questions: Mapping, folds: int = 5, calibrate: bool = True) -> int:
    """The fewest labelled rows :func:`fit_policy` accepts for these questions.

    Heads need ``folds`` rows per option (the widest question decides) and never fewer than
    ``2 * folds``; with ``calibrate`` another quarter is held out to calibrate on, at least ten.
    A five-way choice needs 35, a yes/no question 20. These are floors."""
    qs = normalise_questions(questions)
    widest = max((len(label_ids(q)) for q in qs.values()), default=2)
    heads = max(2 * folds, folds * widest)
    if not calibrate:
        return heads
    return heads + max(MIN_CALIBRATION_ROWS,
                       math.ceil(heads * CALIBRATION_SHARE / (1 - CALIBRATION_SHARE)))


def _split(examples: Sequence[Tuple], share: float) -> Tuple[list, list]:
    """A calibration share taken across the first question's labels in proportion, at least one
    row from every label that has two; deterministic, so a refit is the same split."""
    by_label: Dict[str, list] = {}
    for ex in examples:
        by_label.setdefault(str(next(iter(ex[1].values()), "")), []).append(ex)
    target = min(len(examples) // 2, max(MIN_CALIBRATION_ROWS, round(len(examples) * share)))
    take = {lab: (1 if len(m) >= 2 else 0) for lab, m in by_label.items()}
    remaining = target - sum(take.values())
    order = sorted(by_label, key=lambda lab: -(len(by_label[lab]) * share - take[lab]))
    while remaining > 0:
        moved = False
        for lab in order:
            if remaining <= 0:
                break
            if take[lab] < len(by_label[lab]) // 2:
                take[lab] += 1
                remaining -= 1
                moved = True
        if not moved:
            break
    fit_part, cal_part = [], []
    for lab, members in by_label.items():
        k = take[lab]
        cal_part.extend(members[len(members) - k:])
        fit_part.extend(members[:len(members) - k])
    return fit_part, cal_part


def fit_policy(model, rows: Any, questions: Optional[Mapping] = None, *, folds: int = 5,
               target_precision: float = 0.95, alpha: float = 0.1,
               calibration_rows: Any = None) -> Policy:
    """Heads, calibration, review band and off-topic limit from your labelled rows, as a Policy.

    A small head per question over the frozen features (seconds on a GPU for a few hundred rows;
    the reader is never touched), then a temperature and conformal threshold per question and a
    review band held to ``target_precision``, all measured on rows the heads never saw: a quarter
    of ``rows`` taken evenly across the first question's labels, or ``calibration_rows`` if you
    give them. Calibrating on the fitting rows is refused: a head is right on its own rows, so
    every confidence becomes 1.000 and the review line cannot be placed.

    The off-topic limit is the largest residual of the routing head (the first question) over
    the calibration rows; a record past it goes to review. Studio also raises it to the residual
    of generic off-topic messages, which a few hundred of your own rows cannot measure.
    """
    from .calibrate import calibrate as _calibrate

    examples, sheet = _examples(rows, questions)
    if not examples:
        raise ValueError("no trainable rows")
    qs = normalise_questions(questions) if questions else sheet
    if calibration_rows is not None:
        cal_examples, cal_sheet = _examples(calibration_rows, questions)
        sheet.update(cal_sheet)
        need = min_rows(qs, folds, calibrate=False)
        if len(examples) < need or len(cal_examples) < MIN_CALIBRATION_ROWS:
            raise ValueError("fit_policy needs at least %d rows for the heads and %d to calibrate on; "
                             "got %d and %d" % (need, MIN_CALIBRATION_ROWS, len(examples),
                                               len(cal_examples)))
        fit_examples = examples
    else:
        need = min_rows(qs, folds, calibrate=True)
        if len(examples) < need:
            raise ValueError("fit_policy needs at least %d labelled rows for these questions (%d for "
                             "the heads, the rest held out to calibrate on); got %d"
                             % (need, min_rows(qs, folds, calibrate=False), len(examples)))
        fit_examples, cal_examples = _split(examples, CALIBRATION_SHARE)
    report = model.fit(fit_examples, sheet, folds=folds)
    for qid, (head, cv) in list(model.heads.items()):
        if not isinstance(head, GuardedHead) and getattr(head, "ready", False):
            guarded = GuardedHead(head.labels, k=head.k, l2=head.l2)
            guarded.__dict__.update(head.__dict__)
            model.heads[qid] = (guarded, cv)
    cal = _calibrate(model, cal_examples, sheet, alpha=float(alpha),
                     target_precision=float(target_precision), readout="auto")
    order = list(qs)
    routing = next((q for q in order if q in model.heads), None)
    novelty = None
    if routing is not None:
        head = model.heads[routing][0]
        rs = [residual(head, model.features(ex[0], ex[2].get(routing, sheet[routing])))
              for ex in cal_examples if routing in ex[1]]
        if rs:
            novelty = (routing, float(max(rs)))
    policy = Policy(model, calibration=cal, questions=qs, novelty=novelty,
                    verified={"status": "fitted here"})
    policy.fit_report = {
        "heads": report, "n_rows": len(examples), "n_fit": len(fit_examples),
        "n_calibrate": len(cal_examples),
        "serving": sorted(q for q, r in report.items() if r.get("used_by_auto")),
        "calibration": {q: {k: r.get(k) for k in ("temperature", "qhat", "ece_before", "ece_after",
                                                  "examples") if k in r}
                        for q, r in cal.report.items()},
        "outbound": None if novelty is None else {"question": novelty[0], "limit": round(novelty[1], 4)},
    }
    return policy
