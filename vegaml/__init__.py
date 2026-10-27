"""vegaml: typed decisions from a frozen language model and a small physics engine.

    import vegaml

    v = vegaml.load()              # the 0.8B, the default
    v = vegaml.load("4b")          # the 4B, same repository
    v.decide("I was billed twice, refund me or I cancel.",
             {"team": {"type": "choice", "instructions": "Which team handles this?",
                       "criteria": {"billing": "payments and refunds", "technical": "bugs"}}})

Two readouts over the same frozen features, chosen with `mode`:

* ``"engine"`` is the physics engine and its adapters. Zero-shot: it answers a question it has never
  seen, and every answer carries a calibrated probability, a conformal set and an abstain flag.
  Every benchmark figure published for this model is measured in this mode.
* ``"ttt"`` is a small head fitted here, on your labelled examples, over the same features. Sharper
  where your label space matches the examples, and worthless outside them. It carries no calibration
  and no abstain flag, so it will answer confidently when it should not.
* ``"auto"`` (the default) is "ttt" for each question that has a head worth using and "engine" for
  the rest. Fit nothing and it is exactly the engine, flags and all, which is why it is safe as the
  default and why it still reproduces the published figures. Call :meth:`fit` and the questions you
  fitted switch over without your having to remember a second argument. Every answer says which
  readout produced it, in ``readout``.
* ``"both"`` returns both, side by side, which is the only honest way to compare them.

An escalation band is separate from both readouts and sits over either: ``set_band`` (or
``decide(review_below=...)``) marks every answer "auto" or "review" by its confidence, which
is the escalation signal that works without labels and without the conformal set coming out
empty. :func:`vegaml.calibrate.calibrate` fits that line to a precision target when you do
have labels.

"Worth using" is not "fitted": :meth:`fit` refuses a head whose cross-validated accuracy is at or
below chance, since such a head has learned nothing and would dress noise up as a sharper answer.
``"auto"`` goes further and uses only heads whose cross-validated accuracy beats both chance and
the engine's own accuracy on the same examples, so a head that loses to the engine or was fitted on
too few examples to cross-validate is left to the engine.
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .band import Band
from .hub import default_token, fetch, reader_dir
from .ttt import TTTHead

# MODES and label_ids are imported by name from outside this package (the bench adapter, the tests),
# so they belong in __all__ rather than merely being present as module attributes.
__all__ = ["load", "Vega", "TTTHead", "Band", "fetch", "default_token", "label_ids", "MODES",
           "attach", "calibrate", "__version__"]
__version__ = "0.14.0"

MODES = ("auto", "engine", "ttt", "both")


class Vega:
    """A loaded checkpoint, with both readouts over one set of frozen features.

    `mode` sets the default readout for :meth:`decide`; "auto" unless you change it.
    """

    def __init__(self, engine, mode: str = "auto", band: Optional[Band] = None):
        if mode not in MODES:
            raise ValueError("mode must be one of %s, got %r" % (", ".join(MODES), mode))
        self.engine = engine
        self.mode = mode
        self.band = band
        self.heads: Dict[str, Tuple[TTTHead, Optional[float]]] = {}
        self._vision = None      # the vision tower, loaded on first decide_image()
        self._tokens_read = 0    # prompt tokens features() has read since decide() last reset it

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, repo_or_path: Optional[str] = None, backbone: Optional[str] = None,
                        device: Optional[str] = None, adapters: str = "auto",
                        mode: str = "auto", revision: Optional[str] = None,
                        token: Union[str, bool, None] = None,
                        max_len: Optional[int] = None,
                        quantize: Optional[str] = None,
                        threads: Optional[int] = None) -> "Vega":
        """Load a checkpoint from the Hub or from a local directory.

        `token` is optional. Left as None it picks up `HF_TOKEN` from the environment if one is
        set, which is what makes a private or gated repository load; with nothing set, the public
        checkpoints load anonymously as before. Pass ``token=False`` to ignore the environment.

        `quantize="int8"` loads the reader with dynamic int8 linear layers: a quarter of the weight
        bytes, for a box that cannot hold the float reader. CPU only, and both what it costs in
        latency and what it costs in agreement with the float reader are measured rather than
        assumed; see docs/cpu.md.

        `threads` is the setting that matters on a CPU box, and the right value belongs to the
        reader. On the shipped Qwen3.5 readers one thread measured 4.9x faster than torch's default
        of four, with identical answers, because their linear-attention layers run a reference
        implementation on CPU that does not thread; a plain-transformer reader preferred four on the
        same box. bench/cpu_path.py sweeps it, docs/cpu.md has our numbers.
        """
        # Imported here, not at module import: torch and transformers are only needed once a
        # checkpoint is actually being loaded.
        from ._api import Vega as _Engine

        reader = os.path.basename(backbone.rstrip("/")) if backbone else None
        path = fetch(repo_or_path, revision=revision, token=token, reader=reader)
        # On a universal (vega-U) checkpoint, backbone= names WHICH reader: the dir carrying
        # that reader's port and calibration is selected by its name, so
        # load("nandakishorm/vega-U", backbone="google/gemma-3-1b-it") is the whole call. On a
        # classic checkpoint it keeps its old meaning, an alternate location of the same
        # backbone, and backbone=None lets the engine read backbone_name out of the
        # checkpoint's own config, which is what makes load("4b") pull Qwen3.5-4B.
        if backbone:
            path = reader_dir(path, backbone)
        return cls(_Engine(path, backbone=backbone, device=device, adapters=adapters,
                           max_len=max_len, quantize=quantize, threads=threads), mode=mode)

    # ------------------------------------------------------------------ escalation
    def set_band(self, band) -> "Vega":
        """Set the review band every :meth:`decide` routes against, or None to clear it.

        Takes a :class:`vegaml.band.Band`, a ``{question_id: threshold}`` mapping, or one number
        used as the line for every question. The band marks each answer "auto" or "review" by its
        confidence, which is the only escalation signal that covers a TTT head (those answers carry
        no conformal set and no abstain flag at all) and the only one that fires on an engine answer
        whose conformal set cannot be empty.
        """
        if band is None or isinstance(band, Band):
            self.band = band
        elif isinstance(band, Mapping):
            self.band = Band.from_thresholds(band)
        else:
            self.band = Band(default=float(band))
        return self

    # ------------------------------------------------------------------ reading
    def features(self, state, question: Mapping):
        """Pooled frozen features for one (state, question): what both readouts see."""
        import numpy as np
        import torch

        from ._common import build_prompt, option_texts, serialize_state

        v = self.engine
        qi = v._internal(question)
        k = len(option_texts(qi))
        text = state if isinstance(state, str) else serialize_state(state)
        ids, spans = build_prompt(v.perc.tok, text, qi, v.cfg["max_len"],
                                  option_order=list(range(k)),
                                  opt_max_tokens=v.cfg["opt_max_tokens"],
                                  query_first_max_state=v.cfg.get("query_first_max_state", 0))
        with torch.no_grad():
            f, _ = v.perc.pooled([{"ids": np.asarray(ids, dtype=np.int32), "spans": spans}],
                                 prefix_min=v.cfg.get("prefix_min_tokens", 4096))
        # Counted here rather than returned, so the return value stays a bare vector for callers.
        # One counter per instance: concurrent decide() calls on one Vega would mix their counts.
        self._tokens_read += len(ids)
        return f[0].float().cpu().numpy().ravel()

    # ------------------------------------------------------------------ fitting
    def fit(self, examples: Sequence[Tuple], questions: Mapping, folds: int = 5,
            min_examples: int = 6, progress=None) -> Dict[str, Dict]:
        """Fit one TTT head per question from labelled examples.

        `examples` is a sequence of ``(state, {question_id: label})``, or
        ``(state, {question_id: label}, {question_id: question})`` when the question itself differs
        between examples. The third element matters more often than it looks: a head reads the
        features of the whole prompt, so if what distinguishes your examples sits in a question's
        ``instructions`` rather than in the state, as it does in most intent classification, then
        reusing one question for every example hands them all an identical prompt, identical
        features, and a head fitted on noise. Pass the per-example question and that cannot happen.

        Fitting reads the frozen backbone once per example for the head and, when the head beats
        chance, scores the engine on the same cross-validated examples to check whether
        ``mode="auto"`` should switch. That is seconds on a GPU for a few dozen examples, and it is
        paid once.

        A head whose cross-validated accuracy is at or below chance is **not kept**: it has learned
        nothing, and serving it would dress noise up as a sharper answer. A head that beats chance
        is kept for deliberate ``mode="ttt"``, and ``mode="auto"`` uses it only when its
        cross-validated accuracy also beats the engine's own accuracy on the same examples.

        ``progress`` is called as ``progress(stage, question_id, done, total)`` while the rows are
        read, where stage is "features" or "engine". It exists because reading the rows is the
        whole cost and it takes minutes: anything with a progress bar in front of a user needs to
        know how far along it is, and the alternative is a bar that sits at zero for the longest
        part of the job.
        """
        report: Dict[str, Dict] = {}
        for qid, q in questions.items():
            X, y, rows = [], [], []
            wanted = sum(1 for e in examples if qid in e[1])
            for example in examples:
                state, labels = example[0], example[1]
                if qid not in labels:
                    continue
                per_example = example[2] if len(example) > 2 else None
                asked = (per_example or {}).get(qid, q)
                X.append(self.features(state, asked))
                y.append(str(labels[qid]))
                rows.append((state, asked))
                if progress:
                    progress("features", qid, len(X), wanted)
            if len(X) < min_examples:
                self.heads.pop(qid, None)
                report[qid] = {"fitted": False, "examples": len(X),
                               "reason": "needs at least %d labelled examples" % min_examples}
                continue
            labels_for_q = label_ids(q)
            # A label outside the question's own option set cannot be predicted by a head spanning
            # that set. Report it here, where there is a report to put it in, rather than raising a
            # bare KeyError from inside the fit.
            bad = sorted({str(v) for v in y if str(v) not in set(labels_for_q)})
            if bad:
                self.heads.pop(qid, None)
                report[qid] = {"fitted": False, "examples": len(X),
                               "reason": ("label(s) %s in the examples are not options of this question, "
                                          "whose options are %s" % (", ".join(repr(b) for b in bad[:5]),
                                                                     labels_for_q))}
                continue
            if len(set(y)) < 2:
                self.heads.pop(qid, None)
                report[qid] = {"fitted": False, "examples": len(X),
                               "reason": ("every example carries the same label (%r); there is nothing to "
                                          "distinguish" % y[0])}
                continue
            head = TTTHead(labels_for_q).fit(X, y)
            cv = head.cross_val_accuracy(X, y, folds=folds)
            chance = 1.0 / max(len(labels_for_q), 2)
            # cv is None when there are too few examples to cross-validate. That is "unknown", not
            # "fine": the head is kept so mode="ttt" can still use it deliberately, but "auto"
            # leaves those questions to the engine rather than trust an unmeasured head.
            beats_chance = None if cv is None else cv > chance
            engine_acc = None
            if beats_chance:
                scored_rows = [rows[i] for i in head.cross_val_rows]
                scored_y = [y[i] for i in head.cross_val_rows]
                engine_acc = self._engine_accuracy(qid, scored_rows, scored_y, progress)
                head.engine_accuracy = engine_acc
            beats_engine = bool(beats_chance and engine_acc is not None and cv > engine_acc)
            if beats_chance is not False:
                self.heads[qid] = (head, cv)
            else:
                self.heads.pop(qid, None)
            report[qid] = {"fitted": beats_chance is not False, "examples": len(X),
                           "cv_accuracy": cv, "engine_accuracy": engine_acc,
                           "chance": round(chance, 3),
                           "at_or_below_chance": None if cv is None else not beats_chance,
                           "used_by_auto": beats_engine}
            if beats_chance is False:
                report[qid]["reason"] = ("cross-validated accuracy %.3f is at or below the chance "
                                         "level %.3f, so the head was discarded" % (cv, chance))
            elif beats_chance and not beats_engine:
                report[qid]["reason"] = ("cross-validated accuracy %.3f does not beat the engine's "
                                         "%.3f on these examples, so mode='auto' will leave this "
                                         "question to the engine" % (cv, engine_acc))
            if cv is None:
                report[qid]["cv_note"] = ("too few examples for %d-fold cross-validation, so the "
                                          "head's accuracy is unknown; %d or more would measure it. "
                                          "mode='auto' will leave this question to the engine"
                                          % (folds, 2 * folds))
        return report

    def _engine_accuracy(self, qid: str, rows: Sequence[Tuple], y: Sequence[str],
                         progress=None) -> float:
        """The engine's own accuracy on the rows the head was cross-validated on.

        This is a second read of each of those rows, which is why ``fit`` costs about two reads
        per labelled example rather than one, and why the progress callback reports it separately.
        """
        from .evals import _prediction

        hit = 0
        for i, ((state, asked), label) in enumerate(zip(rows, y, strict=True), 1):
            ans = self.engine.decide(state, {qid: asked})["answers"][qid]
            hit += int(_prediction(ans, asked["type"], label_ids(asked)) == label)
            if progress:
                progress("engine", qid, i, len(rows))
        return hit / len(rows)

    # ------------------------------------------------------------------ deciding
    def decide(self, state, questions: Mapping, mode: Optional[str] = None, alpha: str = "0.1",
               views: int = 1, review_below: Optional[float] = None) -> Dict:
        """Answer every question from one read of the state.

        With ``mode="auto"``, the default, each question is answered by its head when it has one
        worth using and by the engine otherwise, and every answer carries ``readout`` saying which.
        Fit nothing and this is the engine's own result, unchanged. With ``"engine"`` it is always
        the engine's. With ``"ttt"`` every answer comes from a fitted head, and a question without
        one is an error rather than a silent fall back. With ``"both"`` the result carries
        ``answers`` (engine) and ``ttt`` side by side.

        Every mode returns ``latency_ms``, ``n_tokens`` and ``views``. A head reads the frozen
        backbone too, one pass per question, so its time and tokens are counted with the engine's.

        ``review_below`` is a one-call review band, overriding :meth:`set_band` for this call: every
        answer whose confidence falls below it comes back ``route="review"``.
        """
        mode = mode or self.mode
        if mode not in MODES:
            raise ValueError("mode must be one of %s, got %r" % (", ".join(MODES), mode))
        started = time.perf_counter()
        self._tokens_read = 0
        if mode == "ttt":
            missing = [q for q in questions if q not in self.heads]
            if missing:
                raise RuntimeError(
                    "no TTT head for %s. Call fit(examples, questions) first, or use "
                    "mode='engine'." % ", ".join(sorted(missing)))
            out = {"mode": "ttt", "answers": self._ttt_answers(state, questions)}
        elif mode == "auto":
            out = self._auto(state, questions, alpha=alpha, views=views)
        else:
            out = self.engine.decide(state, questions, alpha=alpha, views=views)
            if mode == "engine":
                out["mode"] = "engine"
                return self._banded(out, review_below)
            out["mode"] = "both"
            out["ttt"] = self._ttt_answers(state, questions, skip_missing=True)
        # These used to come only from engine.decide(), so a request answered wholly by heads had
        # none of them, and a mixed one left the heads' own reads out of both numbers.
        out["latency_ms"] = round(1000 * (time.perf_counter() - started), 1)
        out["n_tokens"] = out.get("n_tokens", 0) + self._tokens_read
        out.setdefault("views", 1)          # a head reads the state once
        return self._banded(out, review_below)

    def _banded(self, out: Dict, review_below: Optional[float]) -> Dict:
        """``out`` routed by the call's band, this instance's band, or not at all."""
        band = self.band if review_below is None else Band(default=float(review_below))
        return band.apply(out) if band is not None else out

    def _auto(self, state, questions: Mapping, alpha: str, views: int) -> Dict:
        by_head = [qid for qid in questions if self._head_worth_using(qid, questions[qid])]
        by_engine = {qid: q for qid, q in questions.items() if qid not in by_head}
        out: Dict = {"mode": "auto", "answers": {}}
        if by_engine:
            out = self.engine.decide(state, by_engine, alpha=alpha, views=views)
            out["mode"] = "auto"
            for qid in by_engine:
                out["answers"][qid]["readout"] = "engine"
        for qid, answer in self._ttt_answers(state, {q: questions[q] for q in by_head}).items():
            answer["readout"] = "ttt"
            out["answers"][qid] = answer
        # Questions come back in the order they were asked, not grouped by readout.
        out["answers"] = {qid: out["answers"][qid] for qid in questions}
        return out

    def _head_worth_using(self, qid: str, question: Mapping) -> bool:
        head_cv = self.heads.get(qid)
        if head_cv is None:
            return False
        head, cv = head_cv
        options = label_ids(question)
        if cv is None or options != head.labels:
            return False
        bar = 1.0 / max(len(options), 2)
        if head.engine_accuracy is not None and head.engine_accuracy > bar:
            bar = head.engine_accuracy
        return cv > bar

    def _ttt_answers(self, state, questions: Mapping, skip_missing: bool = False) -> Dict:
        answers: Dict[str, Dict] = {}
        for qid, q in questions.items():
            hv = self.heads.get(qid)
            if hv is None:
                if skip_missing:
                    continue
                raise RuntimeError("no TTT head for %r" % qid)
            head, cv = hv
            allowed = label_ids(q)
            # A head is fitted per question, spanning that question's own options. Asking it a
            # question with a different option set used to fail deep inside predict() with
            # "max() arg is an empty sequence", naming neither the head nor the question.
            if not set(allowed) & set(head.labels):
                raise RuntimeError(
                    "the TTT head for %r was fitted on %s, but this question offers %s. Refit with "
                    "fit(examples, questions) for the question as it is now written."
                    % (qid, head.labels, allowed))
            probs = head.predict(self.features(state, q), allowed=allowed)
            best = max(probs, key=probs.get)
            answer = {"probs": probs, "confidence": probs[best], "cv_accuracy": cv,
                      "readout": "ttt"}
            # The engine keys each answer type differently, and mode="auto" is documented as
            # reproducing the engine's answer shape. A head answer was always keyed "choice", so a
            # boolean question answered by a head lost its "boolean" field and switching readout
            # changed the caller's code.
            t = q["type"]
            if t == "boolean":
                answer["boolean"] = probs["true"]
                answer["choice"] = "true" if best == "true" else "false"
            elif t == "score":
                level = int(best)
                answer["score"] = level
                answer["expected_score"] = float(sum(int(k) * v for k, v in probs.items()))
                answer["choice"] = level
            else:
                answer["choice"] = best
            answers[qid] = answer
        return answers

    # ------------------------------------------------------------------ images
    def decide_image(self, image, questions: Mapping, mode: Optional[str] = None,
                     alpha: str = "0.1", risk=None, **kw) -> Dict:
        """The same typed interface over a picture, through the backbone's frozen vision encoder.

        The vision tower is loaded on first use, so a text-only caller never pays for it. The image
        path runs the engine, so it answers in "engine" shape whatever `mode` says: a TTT head reads
        text features and has nothing to say about pixels. Every answer is marked
        `readout: "engine"` so a caller reading a mixed result is not misled.

        `image` is a PIL Image, a path to one, or an HxWx3 numpy array.
        """
        mode = mode or self.mode
        if mode not in MODES:
            raise ValueError("mode must be one of %s, got %r" % (", ".join(MODES), mode))
        if self._vision is None:
            from ._vision import VegaVision

            # The same backbone the engine resolved for itself, not a re-derived guess: a checkpoint
            # carrying a local backbone/ directory must load that, not a Hub id.
            self._vision = VegaVision(self.engine.model_dir,
                                      backbone=self.engine.backbone,
                                      device=str(getattr(self.engine, "device", "cpu")), **kw)
        out = self._vision.decide_image(image, questions, alpha=alpha, risk=risk)
        out["mode"] = "engine"
        for answer in out["answers"].values():
            answer["readout"] = "engine"
        return out

    def decide_image_text(self, image, questions: Mapping, **kw) -> Dict:
        """Parity check: answer the same questions through the vision path with the image withheld.

        Useful for measuring what the picture itself contributed, the way the README's MMMU-Pro
        "image withheld" row was measured.
        """
        if self._vision is None:
            from ._vision import VegaVision

            self._vision = VegaVision(self.engine.model_dir,
                                      backbone=self.engine.backbone,
                                      device=str(getattr(self.engine, "device", "cpu")))
        return self._vision.decide_text("", questions, **kw)

    def __repr__(self) -> str:
        return "Vega(mode=%r, heads=%d, max_len=%d)" % (
            self.mode, len(self.heads), self.engine.cfg.get("max_len", 0))


def label_ids(question: Mapping) -> List[str]:
    """The option ids a readout must span, for any of the three answer types."""
    t = question["type"]
    if t == "boolean":
        return ["false", "true"]
    if t == "score":
        return [str(i) for i in range(len(question.get("criteria") or []))]
    return [str(k) for k in question["criteria"]]


def attach(backbone: str, texts, out_dir: str, **kw) -> Dict:
    """Attach YOUR OWN frozen reader to the universal engine, from your own text.

    Self-supervised: the checkpoint's root reader generates the shared-space targets, so no
    labels and none of the original training data are needed. Writes a complete model dir that
    :func:`load` loads like any shipped reader, the engine unchanged byte for byte. Labels
    enter only through ``examples=`` here or :func:`calibrate` afterwards. See
    :mod:`vegaml._attach` for the full story, including the probe prompts and the hardware it
    takes (in short: enough GPU memory for the new reader itself)."""
    from ._attach import attach as _attach

    return _attach(backbone, texts, out_dir, **kw)


def calibrate(model, examples, questions, **kw) -> Dict:
    """Refit a checkpoint dir's calibration on your own labelled examples.

    Calibration never transfers between readers or distributions, so this is the step after
    :func:`attach`, and worth running on a shipped reader too when your data looks nothing like
    the published splits. Roughly 500 examples per question type is usable; 1,000 to 2,000 is
    where the numbers settle. The block is written into the dir and applied in place."""
    from ._attach import calibrate as _calibrate

    return _calibrate(model, examples, questions, **kw)


def load(repo_or_path: Optional[str] = None, **kw) -> Vega:
    """Load a checkpoint.

    With no argument you get the 0.8B, which is the default and the one every published figure
    measures. ``load("4b")`` gets the 4B from the same repository. Anything else is treated as a
    Hub repository id or a local path. See :meth:`Vega.from_pretrained` for the other arguments.
    """
    return Vega.from_pretrained(repo_or_path, **kw)


# vegaml/calibrate.py (the legacy calibration module) shares its name with the calibrate()
# function above, and the FIRST import of a submodule binds it onto the package, clobbering
# whatever held the name. evals imports it, and fit imports evals, so one fit() call used to
# turn vegaml.calibrate into a module. Spend that one binding here, at import time (it has
# to happen after label_ids exists, which the submodule imports), and take the name back:
# every later "from .calibrate import ..." resolves through sys.modules and never touches
# the package attribute again.
# "from . import calibrate" would NOT do it: that form prefers the package attribute,
# which is already the function, and never loads the file. import_module always loads.
import importlib as _importlib  # noqa: E402

_calibrate_fn = calibrate
_importlib.import_module(".calibrate", __name__)       # binds the module onto the package
calibrate = _calibrate_fn                              # and the function takes the name back
del _calibrate_fn, _importlib
