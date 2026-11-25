"""Continuous learning: a vega model that keeps learning from the decisions you confirm, on the machine it
runs on.

    from vegaml import Learner
    learner = Learner(questions, examples=rows, workdir="./vega-state")
    out = learner.decide("I was charged twice this month")      # {"answers": {...}, "version": 3, ...}
    learner.feedback("I was charged twice this month", {"team": "billing", "urgent": True})

Each question has a head: a small classifier over vega's frozen features, fitted on the starting set (your
labelled
examples, if any) and refitted as confirmed answers arrive (every `every` answers, in a background thread).
A head answers only when its
cross-validated accuracy beats vega's own answer on the same rows; otherwise vega answers that question
itself. The
starting set gives way to your own answers one for one, so after enough of them the heads are fitted on
yours alone.
"""

import collections
import concurrent.futures as cf
import datetime as dt
import hashlib
import io
import json
import math
import os
import queue
import threading
import time
import zipfile

import numpy as np

from .reader import Reader, label_of
from .records import DEFAULT_BUDGET, Record, as_record

FORMAT = "vega-learn/1"


def _key(item):
    """The cache key of what vega reads: a text, or a picture record (its own content hash)."""
    if isinstance(item, Record):
        return item.key
    return hashlib.sha1(item.encode("utf-8")).hexdigest()


class Store:
    """Per question: features (float16), label, vega's own label ("" when unknown), and where each row came
    from."""

    def __init__(self, qids, d=0):
        self.d = d
        self.X = {q: np.zeros((0, d), np.float16) for q in qids}
        self.y = {q: [] for q in qids}
        self.e = {q: [] for q in qids}
        self.src = {q: [] for q in qids}
        self.k = {q: [] for q in qids}  # what each row was read from ("" when unknown: an old bundle)

    def add(self, qid, X, y, e, src, keys=None):
        """Append rows. With `keys`, a row for a record already in the store replaces it: a corrected
        answer takes the place of the one it corrects (newest wins, whether that was a starting-set row or
        an earlier answer of yours), so the two never pull the head opposite ways."""
        X = np.asarray(X, np.float16)
        y, e = list(y), list(e)
        keys = [""] * len(y) if keys is None else [str(k or "") for k in keys]
        if keys and any(keys):
            last = {}
            for i, k in enumerate(keys):
                if k:
                    last[k] = i
            keep_new = [i for i, k in enumerate(keys) if not k or last[k] == i]  # within the batch, last wins
            if len(keep_new) < len(keys):
                X, y, e, keys = (
                    X[keep_new],
                    [y[i] for i in keep_new],
                    [e[i] for i in keep_new],
                    [keys[i] for i in keep_new],
                )
            replaced = set(last)
            keep = [i for i, k in enumerate(self.k[qid]) if not k or k not in replaced]
            if len(keep) < len(self.k[qid]):
                self.X[qid] = self.X[qid][keep]
                self.y[qid] = [self.y[qid][i] for i in keep]
                self.e[qid] = [self.e[qid][i] for i in keep]
                self.src[qid] = [self.src[qid][i] for i in keep]
                self.k[qid] = [self.k[qid][i] for i in keep]
        if self.X[qid].shape[1] == 0 and len(self.X[qid]) == 0:
            self.X[qid] = np.zeros((0, X.shape[1]), np.float16)
            self.d = X.shape[1]
        self.X[qid] = np.concatenate([self.X[qid], X])  # a new array: the previous one stays valid for undo
        self.y[qid] = self.y[qid] + y
        self.e[qid] = self.e[qid] + e
        self.src[qid] = self.src[qid] + [src] * len(y)
        self.k[qid] = self.k[qid] + keys

    def rows(self, qid, phase_out=True, max_rows=5000):
        """Indices to fit on: the starting set gives way one for one to the customer's own rows, newest
        kept."""
        src = self.src[qid]
        start = [i for i, s in enumerate(src) if s == "start"]
        yours = [i for i, s in enumerate(src) if s != "start"]
        if phase_out:
            start = start[: max(0, len(start) - len(yours))]
        idx = start + yours
        return idx[-max_rows:]

    def counts(self, qid):
        n = len(self.src[qid])
        s = sum(1 for x in self.src[qid] if x == "start")
        return {"starting_set": s, "yours": n - s}

    def copy(self):
        c = Store(list(self.X), self.d)
        c.X, c.y, c.e, c.src, c.k = dict(self.X), dict(self.y), dict(self.e), dict(self.src), dict(self.k)
        return c

    def dump(self):
        arr = {}
        for q in self.X:
            arr["X/" + q] = self.X[q]
            arr["y/" + q] = np.array(self.y[q], dtype=np.str_)
            arr["e/" + q] = np.array(self.e[q], dtype=np.str_)
            arr["s/" + q] = np.array(self.src[q], dtype=np.str_)
            arr["k/" + q] = np.array(self.k[q], dtype=np.str_)
        buf = io.BytesIO()
        np.savez(buf, **arr)
        return buf.getvalue()

    @classmethod
    def load(cls, raw, qids):
        z = np.load(io.BytesIO(raw), allow_pickle=False)
        s = cls(qids)
        for q in qids:
            if "X/" + q not in z.files:
                continue
            s.X[q] = z["X/" + q].astype(np.float16)
            s.d = s.X[q].shape[1] or s.d
            s.y[q] = [str(v) for v in z["y/" + q]]
            s.e[q] = [str(v) for v in z["e/" + q]]
            s.src[q] = [str(v) for v in z["s/" + q]] if "s/" + q in z.files else ["start"] * len(s.y[q])
            s.k[q] = [str(v) for v in z["k/" + q]] if "k/" + q in z.files else [""] * len(s.y[q])
        return s


def head_kind():
    """Which learner turns confirmed examples into answers: "ttt" (the logistic head, the default) or "wells"
    (landscape learning: one well per answer in the energy landscape, see wells.py). Chosen per process by
    VEGA_LEARN_HEAD ("landscape" is accepted for "wells"), so a server can be switched and switched back
    without touching what its learners have stored.

    The head stays the default because landscape learning is not better everywhere: level with the head on
    questions with a few answers, behind it on questions with 60 or more answers until it has seen several
    hundred examples, and faster to fit only past about a thousand examples."""
    kind = os.environ.get("VEGA_LEARN_HEAD", "ttt").strip().lower() or "ttt"
    kind = "wells" if kind == "landscape" else kind
    if kind not in ("ttt", "wells"):
        raise ValueError("VEGA_LEARN_HEAD is %r; it can be ttt or wells (landscape)" % kind)
    return kind


def fit_head(question, X, y, e, folds=5):
    """{"head", "cv", "engine", "usable", "rows"} or None when there is nothing to fit."""
    from vegaml import label_ids

    labels = label_ids(question)
    if len(y) < 10 or len(set(y)) < 2:
        return None
    X = np.asarray(X, np.float32)
    if head_kind() == "wells":
        from .wells import WellsHead

        head = WellsHead(labels).fit(X, y)
    else:
        from vegaml.ttt import TTTHead

        head = TTTHead(labels).fit(X, y)
    cv = head.cross_val_accuracy(X, y, folds=folds)
    scored = [i for i in head.cross_val_rows if e[i]]
    chance = 1.0 / max(len(labels), 2)
    eng = (sum(e[i] == y[i] for i in scored) / len(scored)) if scored else chance
    head.engine_accuracy = eng
    # The bar is balanced accuracy (mean recall over the answers present), not accuracy: on a question
    # where one answer is rare (a jailbreak, a fraud), a head that always gives the common answer scores
    # high on accuracy while catching none of the rare ones, and must not take over from vega for it.
    preds = getattr(head, "cross_val_preds", {})
    cv_bal = _balanced([y[i] for i in preds], list(preds.values())) if preds else None
    eng_bal = _balanced([y[i] for i in scored], [e[i] for i in scored]) if scored else None
    k = len({y[i] for i in preds}) or 2
    bar = max(eng_bal if eng_bal is not None else 0.0, 1.0 / k)
    usable = cv is not None and cv > max(eng, chance) and cv_bal is not None and cv_bal > bar
    return {
        "head": head,
        "cv": cv,
        "engine": eng,
        "cv_balanced": cv_bal,
        "engine_balanced": eng_bal,
        "usable": usable,
        "rows": len(y),
        "kind": head_kind(),
    }


def _balanced(gold, pred):
    """Mean recall over the answers in `gold`: always giving the commonest answer scores 1/k."""
    by = {}
    for g, p in zip(gold, pred, strict=True):
        by.setdefault(g, []).append(g == p)
    return sum(sum(v) / len(v) for v in by.values()) / len(by) if by else None


def answer_from(question, probs):
    """A head's probabilities in vegaml's answer shape, plus "answer": the label as you would write it
    back."""
    best = max(probs, key=probs.get)
    a = {"probs": probs, "confidence": probs[best], "readout": "head"}
    t = question["type"]
    if t == "boolean":
        a["boolean"] = probs.get("true", 0.0)
        a["answer"] = best == "true"
    elif t == "score":
        a["score"] = int(best)
        a["expected_score"] = float(sum(int(k) * v for k, v in probs.items()))
        a["answer"] = int(best)
    else:
        a["choice"] = best
        a["answer"] = best
    return a


def temper(probs, temperature):
    """A head's probabilities with its fitted temperature applied (logits divided by it): T > 1 softens an
    over-confident head, T < 1 sharpens an under-confident one. None or 1 leaves them as they are."""
    if not temperature or abs(float(temperature) - 1.0) < 1e-9:
        return probs
    t = float(temperature)
    logs = {k: math.log(max(float(v), 1e-12)) / t for k, v in probs.items()}
    m = max(logs.values())
    e = {k: math.exp(v - m) for k, v in logs.items()}
    z = sum(e.values())
    return {k: v / z for k, v in e.items()}


def _nll(rows, temperature):
    return -sum(math.log(max(temper(p, temperature)[y], 1e-12)) for p, y in rows) / len(rows)


def fit_temperature(rows):
    """The temperature that minimises the negative log-likelihood of the right answers.
    rows: [(probs, label)]."""
    grid = [0.25 * 1.08**i for i in range(60)]  # 0.25 to about 24
    return min(grid, key=lambda t: _nll(rows, t))


def calibration_error(confs, right, bins=10):
    """Expected calibration error: the gap between confidence and accuracy, averaged over confidence bins
    weighted by how many answers fall in each."""
    n, total = len(confs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs) if lo < c <= hi or (b == 0 and c == 0)]
        if idx:
            conf = sum(confs[i] for i in idx) / len(idx)
            acc = sum(right[i] for i in idx) / len(idx)
            total += len(idx) / n * abs(conf - acc)
    return total


def start_line(confs, right, trust=0.95, min_rows=30):
    """The lowest confidence above which answers were right at least `trust` of the time, over at least
    `min_rows` answers; None when no line reaches it."""
    pts = sorted(zip(confs, right, strict=True), key=lambda x: -x[0])
    best, hit = None, 0
    for i, (c, ok) in enumerate(pts, 1):
        hit += ok
        if i >= min_rows and hit / i >= trust:
            best = c
    return best


def engine_answer(question, a):
    a = dict(a, readout="engine")
    t = question["type"]
    lab = label_of(question, a)
    a["answer"] = (lab == "true") if t == "boolean" else (int(lab) if t == "score" else lab)
    return a


NEEDED = ("vega/learn.json", "vega/start.npz")


def read_bundle(path):
    """{name: bytes} for the two documents of a learner bundle: a .zip, or a folder, holding vega/learn.json
    and
    vega/start.npz at its root or inside one top-level folder."""
    path = os.path.expanduser(os.fspath(path))
    got = {}
    if os.path.isdir(path):
        roots = [path] + [
            os.path.join(path, d) for d in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path, d))
        ]
        for r in roots:
            if os.path.isfile(os.path.join(r, "vega", "learn.json")):
                for n in NEEDED:
                    with open(os.path.join(r, *n.split("/")), "rb") as f:
                        got[n] = f.read()
                break
    elif os.path.isfile(path) and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            names = [i.filename for i in z.infolist() if not i.is_dir()]
            hits = [n for n in names if n == NEEDED[0] or (n.endswith("/" + NEEDED[0]) and n.count("/") == 2)]
            if len(hits) == 1:
                top = hits[0][: -len(NEEDED[0])]
                for n in NEEDED:
                    if top + n in names:
                        got[n] = z.read(top + n)
    else:
        raise FileNotFoundError("no learner bundle at %s" % path)
    missing = [n for n in NEEDED if n not in got]
    if missing:
        raise ValueError("%s is not a learner bundle: it has no %s" % (path, " or ".join(missing)))
    return got


def starting_store(reader, questions, rows, progress=None):
    """A store holding the starting set: rows are [(text, {question id: label})], labels as a person writes
    them.
    Questions with no rows start with vega's own answers and learn from yours. Returns (store, report)."""
    progress = progress or (lambda stage, done=0, total=0: None)
    qids = list(questions)
    clean, skipped = [], {q: 0 for q in qids}
    for text, labels in rows:
        got = {}
        for q, v in labels.items():
            if q not in questions:
                continue
            try:
                got[q] = label_of(questions[q], v)
            except (ValueError, TypeError):
                skipped[q] += 1
        if got:
            clean.append((text, got))
    store = Store(qids)
    if clean:
        texts = [t for t, _ in clean]
        progress("reading the starting set", 0, len(texts))
        feats = reader.features(texts, questions)
        eng = []
        for i, t in enumerate(texts):
            eng.append(reader.engine_labels(t, questions))
            if i % 50 == 0:
                progress("vega's own answers", i, len(texts))
        for qi, q in enumerate(qids):
            idx = [i for i, (_, lab) in enumerate(clean) if q in lab]
            if idx:
                store.add(
                    q,
                    feats[idx, qi],
                    [clean[i][1][q] for i in idx],
                    [eng[i][q] for i in idx],
                    "start",
                    keys=[_key(texts[i]) for i in idx],
                )
    report = {}
    for q in qids:
        idx = store.rows(q, phase_out=False)
        h = (
            fit_head(
                questions[q], store.X[q][idx], [store.y[q][i] for i in idx], [store.e[q][i] for i in idx]
            )
            if idx
            else None
        )
        report[q] = {
            "rows": len(idx),
            "skipped_rows": skipped[q],
            "starts_with": "a head fitted on the starting set" if h and h["usable"] else "vega's own answers",
            "cross_validated_accuracy": None if not h or h["cv"] is None else round(h["cv"], 3),
            "vega_alone_accuracy": None if not h else round(h["engine"], 3),
        }
    progress("done", 1, 1)
    return store, report


def bundle_bytes(reader, questions, store, name=None, modality="text", calibration=None):
    """learn.json and start.npz for a store: what a Learner(bundle=...) starts from. `calibration`, when
    given, travels with it and is applied by the learner that loads it."""
    meta = {
        "format": FORMAT,
        "name": name,
        "modality": modality,
        "fingerprint": reader.fingerprint(),
        "questions": questions,
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "start": {q: len(store.y[q]) for q in questions},
    }
    if calibration:
        meta["calibration"] = calibration
    return json.dumps(meta, indent=1, ensure_ascii=False).encode("utf-8"), store.dump()


def build(reader, questions, rows, name=None, progress=None):
    """A starting set turned into a bundle: (learn.json bytes, start.npz bytes, report)."""
    store, report = starting_store(reader, questions, rows, progress)
    learn, start = bundle_bytes(reader, questions, store, name)
    return learn, start, report


class Learner:
    def __init__(
        self,
        questions=None,
        examples=None,
        *,
        bundle=None,
        model=None,
        device=None,
        workdir=None,
        every=50,
        auto=True,
        max_drop=0.20,
        phase_out=True,
        max_rows=5000,
        keep=3,
        cache=4096,
        follow=0,
        reader=None,
        on_event=None,
        modality="text",
        context="full",
        budget=DEFAULT_BUDGET,
        frames=4,
        asr=None,
        language=None,
    ):
        """questions: vegaml questions, {id: {"type": "choice" | "boolean" | "score", "instructions",
        "criteria"}}.
        examples: an optional starting set, [(text, {question id: answer})]. A question with enough examples
        starts
            with a head fitted on them; one with none starts with vega's own answers and learns from yours.
        bundle: instead of questions and examples, a learner bundle (a .zip or folder with vega/learn.json and
            vega/start.npz, as save_bundle() writes it) to start from.
        model / device: the vega checkpoint and device (default: vegaml's default model, cuda when present).
        workdir: where learned state is saved and resumed from (None: nothing is saved).
        every: learn after this many confirmed answers. auto: learn in a background thread.
        max_drop: a batch the current heads agree with this much less than they agree with your answers so
        far is
            held for review (a broken feed, or a real change of policy that someone should confirm).
        follow: seconds between reloads of the newest state in workdir, for a read-only copy (0: off).
            A follower never learns; one writer learns and saves, any number of followers answer.
        modality: what this learner reads. "text" takes text, conversations, audio (its transcript) and
            clips with speech; "image" takes pictures (with any words beside them) and silent clips (their
            frames). The two are read through different paths, so they are separate learners. A record is
            a string, or a dict as vegaml_learn.as_record describes (text / image / images / audio / video,
            or turns of them).
        context / budget: what is read of a long conversation ("full", "recent" or "last") within `budget`
            characters, keeping the newest turns. frames / asr / language: how clips are sampled and
            recordings transcribed."""
        self.reader = reader or Reader.load(model, device)
        if modality not in ("text", "image"):
            raise ValueError("modality is text or image, not %r" % (modality,))
        self.modality = modality
        self.read_kw = dict(context=context, budget=budget, frames=frames, asr=asr, language=language)
        self.every, self.max_drop, self.phase_out, self.max_rows, self.keep = (
            int(every),
            max_drop,
            phase_out,
            max_rows,
            keep,
        )
        self.workdir = None if workdir is None else os.path.expanduser(os.fspath(workdir))
        #: per question: {"temperature", "start_line", ...} from calibrate() or the bundle, applied by
        #: decide()
        self.calibration = {}
        self.on_event = on_event or (lambda event: None)
        self.follow = float(follow or 0)
        self.report = None
        start = None
        if bundle is not None:
            files = read_bundle(bundle)
            meta = json.loads(files["vega/learn.json"])
            if meta.get("format") != FORMAT:
                raise ValueError("this is not a vega learner bundle (format %r)" % meta.get("format"))
            if meta["fingerprint"] != self.reader.fingerprint():
                raise ValueError(
                    "this bundle was made for a different vega checkpoint (it expects %s, this one is %s)"
                    % (meta["fingerprint"], self.reader.fingerprint())
                )
            self.questions = meta["questions"]
            self.modality = meta.get("modality", "text")
            self.calibration = dict(meta.get("calibration") or {})
            source = hashlib.sha256(files["vega/start.npz"]).hexdigest()
            start = files["vega/start.npz"]
        else:
            if not questions:
                raise ValueError("give the questions to answer, or a bundle to start from")
            self.questions = json.loads(json.dumps(questions))
            examples = [(self._item(t), dict(lab)) for t, lab in (examples or [])]
            source = hashlib.sha256(
                json.dumps(
                    [(t if isinstance(t, str) else t.key, lab) for t, lab in examples],
                    sort_keys=True,
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
        self.meta = {"format": FORMAT, "fingerprint": self.reader.fingerprint(), "questions": self.questions}
        #: what saved state must match to be resumed: the questions, the checkpoint and the starting set
        #: (and the modality, for a picture learner: its features are not comparable with text ones)
        ident = {"q": self.questions, "f": self.reader.fingerprint(), "s": source}
        if self.modality != "text":
            ident["m"] = self.modality
        self.set_id = hashlib.sha256(json.dumps(ident, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        self._lock = threading.RLock()  # heads, store, version
        self._inbox = threading.Lock()  # the feedback buffer
        self._cache = (
            collections.OrderedDict()
        )  # text key -> {qid: features}, from decide(), reused by learning
        self._eng = (
            collections.OrderedDict()
        )  # text key -> {qid: vega's own label}, from decide(), reused too
        self._cache_max = int(cache)
        self._buffer = []
        self._corrections = set()  # keys of records a person corrected, not counted by the drift guard
        self._held, self._undo, self.last_error = {}, None, None
        resumed = self._resume()
        if resumed is not None:
            store, version, expected = resumed
        else:
            version, expected = 0, {}
            if start is not None:
                store = Store.load(start, list(self.questions))
            else:
                store, self.report = starting_store(self.reader, self.questions, examples)
        self.store, self.version = store, version
        #: per question, how often the heads agreed with your answers before learning from them (moving
        #: average)
        self.expected = expected
        self.heads = self._fit_all()
        self._jobs, self._thread, self._stop = queue.Queue(), None, threading.Event()
        if auto and not self.follow:
            self._thread = threading.Thread(target=self._run, name="vega-learn", daemon=True)
            self._thread.start()
        if self.follow:
            threading.Thread(target=self._follow, name="vega-learn-follow", daemon=True).start()

    def save_bundle(self, path, name=None):
        """The learned state as a bundle (.zip) another Learner can start from, on another machine."""
        with self._lock:
            learn, start = bundle_bytes(
                self.reader, self.questions, self.store, name, self.modality, calibration=self.calibration
            )
        with zipfile.ZipFile(os.path.expanduser(os.fspath(path)), "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("vega/learn.json", learn)
            z.writestr("vega/start.npz", start)
        return os.fspath(path)

    # ------------------------------------------------------------------ answering
    def record(self, x):
        """A record ready for this learner: a string, a vegaml_learn.Record, or a dict of text / image /
        images / audio / video or turns of them (transcribed and sampled here, once)."""
        rec = as_record(x, self.reader, **self.read_kw)
        if rec.kind != self.modality:
            other = "Learner(questions, modality=%r)" % rec.kind
            raise ValueError(
                "this learner reads %s; that record is %s, which %s reads"
                % (self.modality, "a picture" if rec.kind == "image" else "text", other)
            )
        return rec

    def _item(self, x):
        """What the caches and the reader take: the text itself for a text record, the Record for a
        picture."""
        if isinstance(x, str) and self.modality == "text":
            return x
        rec = self.record(x)
        return rec.text if rec.kind == "text" else rec

    def decide(self, text, questions=None):
        """Answer every installed question (or the ids in `questions`) about one record: a message, or
        what `record()` takes (a picture, a recording, a clip, a conversation)."""
        t0 = time.perf_counter()
        info = None
        if isinstance(text, str) and self.modality != "text":
            self.record(text)  # says which learner takes it
        if not isinstance(text, str):
            rec = text if isinstance(text, Record) else self.record(text)
            info = rec.describe()
            text = rec.text if rec.kind == "text" else rec
            if rec.kind != self.modality:
                self.record(rec)  # says which learner takes it
        qids = list(self.questions) if questions is None else list(questions)
        unknown = [q for q in qids if q not in self.questions]
        if unknown:
            raise KeyError(
                "not one of this learner's questions: %r; it has %r" % (unknown, sorted(self.questions))
            )
        with self._lock:
            heads, version = dict(self.heads), self.version
        by_head = [q for q in qids if heads.get(q) and heads[q]["usable"]]
        by_engine = [q for q in qids if q not in by_head]
        answers = {}
        if by_head:
            f = self._features_for(text, by_head)
            for q in by_head:
                h = heads[q]["head"]
                from vegaml import label_ids

                probs = h.predict(f[q].astype(np.float32), allowed=label_ids(self.questions[q]))
                probs = temper(probs, (self.calibration.get(q) or {}).get("temperature"))
                answers[q] = answer_from(self.questions[q], probs)
        if by_engine:
            qs = {q: self.questions[q] for q in by_engine}
            labels = {}
            for q, a in self.reader.engine_answers(text, qs).items():
                answers[q] = engine_answer(self.questions[q], a)
                labels[q] = label_of(self.questions[q], a)
            k = _key(text)
            self._eng[k] = dict(self._eng.get(k, {}), **labels)
            self._eng.move_to_end(k)
            while len(self._eng) > self._cache_max:
                self._eng.popitem(last=False)
        out = {
            "answers": {q: answers[q] for q in qids},
            "version": version,
            "latency_ms": round(1000 * (time.perf_counter() - t0), 1),
        }
        if info is not None:
            out["record"] = info
        return out

    def decide_many(self, texts, questions=None):
        return [self.decide(t, questions) for t in texts]

    def _features_for(self, text, qids):
        k = _key(text)
        got = self._cache.get(k, {})
        need = [q for q in qids if q not in got]
        if need:
            f = self.reader.features([text], {q: self.questions[q] for q in need})[0]
            got = dict(got, **{q: f[i] for i, q in enumerate(need)})
            self._cache[k] = got
            self._cache.move_to_end(k)
            while len(self._cache) > self._cache_max:
                self._cache.popitem(last=False)
        return got

    # ------------------------------------------------------------------ feedback in
    def feedback(self, text, answers, correction=False):
        """One decision whose right answer is now known. `answers`: {question id: answer}, written as decide()
        returns it in "answer" (an option name, true/false, or a level). Returns at once; learning happens
        every `every` answers in the background.

        correction: a person is fixing an answer they know was wrong (vega's, or one given earlier). It
        replaces the earlier answer for the same record and is always learned: the guard that holds a
        batch disagreeing with what was learned (a broken feed) does not count it, since a correction is
        meant to disagree."""
        if self.follow:
            raise RuntimeError(
                "this learner follows another one's saved state and does not learn; "
                "send feedback to the writer"
            )
        if not isinstance(text, str) or self.modality != "text":
            text = self._item(text)
        labels = {}
        for q, v in answers.items():
            if q not in self.questions:
                raise KeyError(
                    "question %r is not one of this learner's questions; it has %r"
                    % (q, sorted(self.questions))
                )
            labels[q] = label_of(self.questions[q], v)
        with self._inbox:
            if correction:
                self._corrections.add(_key(text))
            self._buffer.append((text, labels))
            ready = len(self._buffer) >= self.every
            if ready and self._thread is not None:
                self._jobs.put(self._buffer)
                self._buffer = []
        return {"pending": self.pending(), "learning": ready}

    def calibrate(self, examples, trust=0.95, min_rows=30):
        """Fit each head's temperature on labelled examples it was not fitted on ([(record, {question id:
        answer})]), so its confidence means what it says, and record the start line: the lowest calibrated
        confidence above which it was right `trust` of the time. Attached to this learner at once, applied
        by decide(), saved with its state and carried by save_bundle(). Returns per question: temperature,
        start line and the calibration error before and after. Calibrate again after much learning."""
        from vegaml import label_ids

        items = []
        for t, lab in examples:
            got = {q: label_of(self.questions[q], v) for q, v in lab.items() if q in self.questions}
            items.append((self._item(t), got))
        with self._lock:
            heads = dict(self.heads)
        qids = [q for q in self.questions if heads.get(q) and heads[q]["usable"]]
        if not qids or not items:
            return {}
        feats = self.reader.features([t for t, _ in items], {q: self.questions[q] for q in qids})
        report, cal = {}, dict(self.calibration)
        for qi, q in enumerate(qids):
            labels = label_ids(self.questions[q])
            rows = [
                (heads[q]["head"].predict(feats[i, qi].astype(np.float32), allowed=labels), lab[q])
                for i, (_, lab) in enumerate(items)
                if q in lab
            ]
            if len(rows) < min_rows:
                continue
            t = fit_temperature(rows)
            before = [max(p.values()) for p, _ in rows]
            after_p = [temper(p, t) for p, _ in rows]
            after = [max(p.values()) for p in after_p]
            right = [max(p, key=p.get) == y for p, (_, y) in zip(after_p, rows, strict=True)]
            cal[q] = {
                "temperature": round(t, 4),
                "start_line": start_line(after, right, trust, min_rows),
                "trust": trust,
                "rows": len(rows),
                "accuracy": round(sum(right) / len(right), 4),
                "error_before": round(calibration_error(before, right), 4),
                "error_after": round(calibration_error(after, right), 4),
                "head_version": self.version,
            }
            report[q] = cal[q]
        with self._lock:
            self.calibration = cal
            self._save()
        return report

    def pending(self):
        with self._inbox:
            return len(self._buffer)

    def step(self, min_rows=None):
        """Learn now from what is buffered (at least `min_rows`, default `every`). For use without the
        thread."""
        need = self.every if min_rows is None else max(1, int(min_rows))
        with self._inbox:
            if len(self._buffer) < need:
                return None
            batch, self._buffer = self._buffer, []
        return self._learn(batch)

    def flush(self, timeout=None):
        """Wait for queued batches, then learn from whatever is still buffered."""
        if self._thread is not None:
            end = None if timeout is None else time.time() + timeout
            while self._jobs.unfinished_tasks and (end is None or time.time() < end):
                time.sleep(0.02)
        return self.step(min_rows=1)

    def _run(self):
        while True:
            job = self._jobs.get()
            try:
                if job is None:
                    return
                self._learn(job)
            except Exception as e:  # a failed batch must never take the serving process down
                self.last_error = repr(e)
                self.on_event({"kind": "error", "error": repr(e)})
            finally:
                self._jobs.task_done()

    # ------------------------------------------------------------------ learning
    def _learn(self, batch, accept=None):
        t0 = time.time()
        qids = list(self.questions)
        texts = [t for t, _ in batch]
        # features for every question of every row, one pass a message (rows answered by decide() are cached)
        F = {}
        missing = [t for t in texts if set(qids) - set(self._cache.get(_key(t), {}))]
        if missing:
            got = self.reader.features(missing, self.questions)
            for t, f in zip(missing, got, strict=True):
                F[_key(t)] = {q: f[i] for i, q in enumerate(qids)}
        for t in texts:
            F.setdefault(_key(t), {}).update(self._cache.get(_key(t), {}))
        with self._lock:
            store = self.store.copy()
            old = dict(self.heads)
        # vega's own answer is the bar a head must beat, so it is needed where vega still answers; where a
        # head has
        # taken over, its bar was measured before and the row keeps "" (unknown) rather than paying for
        # another read
        want = [q for q in qids if not (old.get(q) and old[q]["usable"])]
        eng = {}
        for t in set(texts):
            k = _key(t)
            have = dict(self._eng.get(k, {}))
            need = {q: self.questions[q] for q in want if q not in have}
            if need:
                have.update(self.reader.engine_labels(t, need))
            eng[k] = {q: have.get(q, "") for q in qids}
        touched = []
        for q in qids:
            rows = [(t, lab[q]) for t, lab in batch if q in lab]
            if not rows:
                continue
            store.add(
                q,
                np.stack([F[_key(t)][q] for t, _ in rows]),
                [y for _, y in rows],
                [eng[_key(t)][q] for t, _ in rows],
                "yours",
                keys=[_key(t) for t, _ in rows],
            )
            touched.append(q)
        new, held, expected, refit = dict(old), {}, dict(self.expected), []
        from vegaml import label_ids

        with self._inbox:
            fixed = {_key(t) for t in texts} & self._corrections
            self._corrections -= fixed
        for q in touched:
            rows = [(t, lab[q]) for t, lab in batch if q in lab and _key(t) not in fixed]
            before = old.get(q)
            if before is not None and before["usable"] and rows:
                labels = label_ids(self.questions[q])
                agree = float(
                    np.mean(
                        [
                            max(
                                (
                                    p := before["head"].predict(
                                        F[_key(t)][q].astype(np.float32), allowed=labels
                                    )
                                ),
                                key=p.get,
                            )
                            == y
                            for t, y in rows
                        ]
                    )
                )
                exp = expected.get(q)
                if exp is not None and agree < exp - self.max_drop and not (accept and q in accept):
                    held[q] = {
                        "why": "the heads agree with this batch %.0f%% of the time, "
                        "against %.0f%% for your answers so far" % (100 * agree, 100 * exp),
                        "rows": len(rows),
                        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                    continue
                # the baseline only counts your own answers: the starting set says nothing about your data
                if self.store.counts(q)["yours"] > 0 or exp is not None:
                    expected[q] = agree if exp is None else 0.7 * exp + 0.3 * agree
                elif len(rows) >= self.every:
                    expected[q] = agree
            refit.append(q)
        new.update(self._fit(store, refit))
        if held:
            # the held questions keep their rows out of the store until accept() or discard()
            for q in held:
                store.X[q], store.y[q], store.e[q], store.src[q], store.k[q] = (
                    self.store.X[q],
                    self.store.y[q],
                    self.store.e[q],
                    self.store.src[q],
                    self.store.k[q],
                )
                held[q]["batch"] = [(t, {q: lab[q]}) for t, lab in batch if q in lab]
        with self._lock:
            self._undo = {
                "store": self.store,
                "heads": self.heads,
                "version": self.version,
                "expected": self.expected,
            }
            self.store, self.heads, self.expected = store, new, expected
            self.version += 1
            self._held.update(held)
            version = self.version
            self._save()
        out = {
            "kind": "learned",
            "version": version,
            "rows": len(batch),
            "questions": touched,
            "held": {q: h["why"] for q, h in held.items()},
            "seconds": round(time.time() - t0, 2),
            "readout": {q: ("head" if new.get(q) and new[q]["usable"] else "engine") for q in qids},
        }
        self.on_event(out)
        return out

    def _fit(self, store, qids):
        """Refit the heads of `qids`, one thread each: the heavy part is numpy, which runs outside the GIL."""

        def one(q):
            idx = store.rows(q, self.phase_out, self.max_rows)
            if not idx:
                return q, None
            return q, fit_head(
                self.questions[q], store.X[q][idx], [store.y[q][i] for i in idx], [store.e[q][i] for i in idx]
            )

        if len(qids) <= 1:
            return dict(one(q) for q in qids)
        with cf.ThreadPoolExecutor(max_workers=len(qids)) as pool:
            return dict(pool.map(one, qids))

    def _fit_all(self, store=None):
        return self._fit(store or self.store, list(self.questions))

    # ------------------------------------------------------------------ held batches and undo
    def held(self):
        """Batches that would have made a question worse, kept out until you decide: {question: why, rows,
        at}."""
        with self._lock:
            return {q: {k: v for k, v in h.items() if k != "batch"} for q, h in self._held.items()}

    def accept(self, qid):
        """The policy really did change: learn the held batch anyway."""
        with self._lock:
            h = self._held.pop(qid)
        return self._learn(h["batch"], accept={qid})

    def discard(self, qid):
        """The held batch was wrong (a broken feed): drop it."""
        with self._lock:
            self._held.pop(qid)

    def undo(self):
        """Back to the heads before the last learned batch (one step)."""
        with self._lock:
            if self._undo is None:
                raise RuntimeError("nothing to undo")
            u, self._undo = self._undo, None
            self.store, self.heads, self.expected = u["store"], u["heads"], u["expected"]
            self.version += 1
            self._save()
            v = self.version
        self.on_event({"kind": "undone", "back_to": u["version"], "version": v})
        return v

    # ------------------------------------------------------------------ files
    def _save(self):
        """The current state under workdir/v<version>/. The last `keep` versions are kept."""
        if self.workdir is None:
            return
        d = os.path.join(self.workdir, "v%06d" % self.version)
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, "state.npz.tmp")
        with open(tmp, "wb") as f:
            f.write(self.store.dump())
        os.replace(tmp, os.path.join(d, "state.npz"))
        meta = {
            "version": self.version,
            "set": self.set_id,
            "expected": self.expected,
            "calibration": self.calibration,
            "saved_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        with open(os.path.join(d, "meta.json.tmp"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        os.replace(os.path.join(d, "meta.json.tmp"), os.path.join(d, "meta.json"))
        done = sorted(
            x
            for x in os.listdir(self.workdir)
            if x.startswith("v") and os.path.isfile(os.path.join(self.workdir, x, "meta.json"))
        )
        for old in done[: -self.keep]:
            for name in ("state.npz", "meta.json"):
                p = os.path.join(self.workdir, old, name)
                if os.path.exists(p):
                    os.remove(p)
            try:
                os.rmdir(os.path.join(self.workdir, old))
            except OSError:
                pass

    def _newest(self):
        if self.workdir is None or not os.path.isdir(self.workdir):
            return None
        for d in sorted((x for x in os.listdir(self.workdir) if x.startswith("v")), reverse=True):
            m, s = os.path.join(self.workdir, d, "meta.json"), os.path.join(self.workdir, d, "state.npz")
            if not (os.path.isfile(m) and os.path.isfile(s)):
                continue
            with open(m, encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("set") == self.set_id:
                if meta.get("calibration"):
                    self.calibration = dict(meta["calibration"])
                return meta["version"], s, meta.get("expected") or {}
        return None

    def _resume(self):
        got = self._newest()
        if got is None:
            return None
        with open(got[1], "rb") as f:
            return Store.load(f.read(), list(self.questions)), got[0], got[2]

    def refresh(self):
        """Load the newest saved state if it is newer than this one (what a follower does on its timer)."""
        got = self._newest()
        if got is None or got[0] <= self.version:
            return False
        with open(got[1], "rb") as f:
            store = Store.load(f.read(), list(self.questions))
        heads = self._fit_all(store)
        with self._lock:
            self.store, self.heads, self.version, self.expected = store, heads, got[0], got[2]
        return True

    def _follow(self):
        while not self._stop.wait(self.follow):
            try:
                self.refresh()
            except Exception as e:
                self.last_error = repr(e)

    def status(self):
        with self._lock:
            qs = {}
            for q in self.questions:
                h = self.heads.get(q)
                qs[q] = dict(
                    self.store.counts(q),
                    readout="head" if h and h["usable"] else "engine",
                    learner=None if not h else h.get("kind", "ttt"),  # which learner fitted it: ttt or wells
                    cross_validated_accuracy=None if not h or h["cv"] is None else round(h["cv"], 3),
                    vega_alone_accuracy=None if not h else round(h["engine"], 3),
                )
            return {
                "version": self.version,
                "set": self.set_id,
                "questions": qs,
                "pending": self.pending(),
                "held": {q: h["why"] for q, h in self._held.items()},
                "agreement": {q: round(v, 3) for q, v in self.expected.items()},
                "every": self.every,
                "workdir": self.workdir,
                "follower": bool(self.follow),
                "background": self._thread is not None,
                "checkpoint": self.reader.fingerprint(),
                "last_error": self.last_error,
            }

    def close(self):
        """Stop the background thread. Answers buffered but not learned yet are dropped; call flush() first
        to keep them."""
        self._stop.set()
        if self._thread is not None:
            self._jobs.put(None)
            self._thread.join(timeout=120)
            self._thread = None
