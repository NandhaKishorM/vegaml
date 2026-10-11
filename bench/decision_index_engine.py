"""Vega as an engine for the Decision Index reproduction kit.

The kit is https://github.com/apolinario/decision-index, which rebuilds the public part of the
Decision Index suite and drives any engine through it:

    pip install "decision-index[transformers,rebuild] @ git+https://github.com/apolinario/decision-index"
    pip install vegaml
    python -m decision_index pipeline \
        --engine bench.decision_index_engine:VegaEngine \
        --option model=0.8b --out runs/vega-08b

One fixed call per request, the published readout (``mode="engine"``, ``views=1``), and no
per-benchmark prompting. The adapter never truncates: a request that does not fit Vega's native
budgets is refused as ``Unsupported``, which the suite counts as wrong. That is the kit's rule and
it is the honest reading of what the model can do.

``--option mode=ttt`` adds the second readout: a head fitted here, on labelled examples you supply,
over the same frozen features. It is fitted per answer space, before the run, from
``--option examples=<path>``, and never from the suite:

    python -m decision_index pipeline \
        --engine bench.decision_index_engine:VegaEngine \
        --option model=0.8b --option mode=ttt --option examples=train-rows.jsonl.gz \
        --out runs/vega-08b-ttt

The examples file holds rows in the suite's own row format (``state``, ``questions``, ``expected``)
drawn from a benchmark's **train** split. Evaluation rows are not training data: the suite is
evaluation data under 43 upstream licences, and a head fitted on it would be measuring itself. An
answer space with no head, or one whose cross-validated accuracy does not beat chance, is answered
by the engine instead, and every answer records which readout produced it.

Two names for one thing: the kit calls the yes/no primitive ``noul``, which is what this answer
type was called here before 0.5.0 renamed it to ``boolean``. The mapping lives in this file only, in
the three places that cross the boundary: ``to_vega`` on the way in, ``to_kit_answers`` on the way
out, and ``_ttt_answer`` for booleans a fitted head answers.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import time

from decision_index.engines import Engine, Unsupported

from vegaml import label_ids
from vegaml._common import QUERY_FIRST, TEMPLATES, answer_space_hash, option_texts, serialize_state

# build_prompt's own defaults for the two budgets the checkpoint config does not carry. _view does
# not override ins_max_tokens, so 256 is the real limit on a question's instructions.
INS_MAX_TOKENS = 256
HEADROOM = 16          # build_prompt starts shrinking options at max_len - 16
QUERY_FIRST_CAP = 128  # build_prompt caps the query-first prefix at 128 tokens

MODES = ("engine", "ttt")
MIN_EXAMPLES = 12      # below this a head's accuracy cannot be cross-validated honestly
CV_FOLDS = 5


def to_vega(key: str, question: dict) -> dict:
    """One kit question as a vegaml question. Raises Unsupported for a type Vega does not have."""
    kind = question["type"]
    if kind == "choice":
        return {"type": "choice", "instructions": question["instructions"],
                "criteria": question["criteria"]}
    if kind == "noul":
        return {"type": "boolean", "instructions": question["instructions"]}
    raise Unsupported("%s: question type %r is outside the typed interface" % (key, kind))


def internal(question: dict) -> dict:
    """The shape vegaml's prompt builder reads, without going through a private helper.

    A choice question may carry its options as a list rather than a mapping. The package's own
    loader normalises that (a bare option name has no description), and this adapter must match it:
    without this, a list-shaped choice question raises inside the prompt builder's `crit.items()`
    instead of working, so the adapter is narrower than the API it wraps.
    """
    ins = question["instructions"]
    crit = question.get("criteria")
    if question["type"] == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    return {"t": question["type"],
            "ins": ins if isinstance(ins, str) else json.dumps(ins),
            "crit": crit}


def to_kit_answers(questions: dict, answers: dict) -> dict:
    """Vega's answers in the kit's response shape, with a full distribution per choice question."""
    out = {}
    for key, question in questions.items():
        answer = answers[key]
        if question["type"] == "noul":
            out[key] = {"type": "noul", "noul": float(answer["boolean"])}
            continue
        probs = {k: float(answer["probs"][str(k)]) for k in question["criteria"]}
        total = sum(probs.values())
        if total > 0:                      # take out the float error, never reorder the options
            probs = {k: v / total for k, v in probs.items()}
        choice = answer["choice"]
        if choice not in question["criteria"]:
            choice = max(probs, key=probs.get)
        out[key] = {"type": "choice", "choice": choice, "probabilities": probs}
    return out


def space_key(question: dict) -> str:
    """Answer spaces, not question keys, are what a head can be fitted on.

    One BANKING77 row asks one question, but every row in it shares the same 77 options, so a head
    fitted on that space serves them all. The hash covers the type and the option texts, which is
    exactly what has to match for a label to mean the same thing.
    """
    return "%s:%x" % (question["t"], answer_space_hash(question))


def read_rows(path: str):
    """Rows from a .jsonl or .jsonl.gz file in the suite's own row format."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def label_text(question: dict, gold) -> str:
    """One gold answer as the label string the head is fitted on.

    A boolean gold arrives from the kit as a number or as the string "true"/"false". `bool("false")`
    is True, so a plain truthiness test fitted every string-false example as a positive one --
    inverting the class the head is being asked to predict. Parse the string forms explicitly.
    """
    if question["type"] == "boolean":
        if isinstance(gold, str):
            return "true" if gold.strip().lower() in ("true", "1", "yes") else "false"
        return "true" if float(gold) != 0.0 else "false"
    return str(gold)


class VegaEngine(Engine):
    name = "vega"
    latency = ("In-process request wall time including prompt construction and the frozen backbone "
               "read; excludes model loading.")

    def __init__(self, **options):
        super().__init__(**options)
        import vegaml

        self.repo = options.get("model") or "0.8b"
        self.views = int(options.get("views", 1))
        self.alpha = str(options.get("alpha", "0.1"))
        self.mode = str(options.get("mode", "engine"))
        if self.mode not in MODES:
            raise ValueError("mode must be one of %s, got %r" % (", ".join(MODES), self.mode))
        self.examples_path = options.get("examples")
        if self.mode == "ttt" and not self.examples_path:
            raise ValueError("mode=ttt needs --option examples=<path>: a head is fitted on labelled "
                             "examples you supply, never on the suite")
        self.vega = vegaml.load(self.repo, device=options.get("device"),
                               adapters=options.get("adapters", "auto"))
        engine = self.vega.engine
        self.cfg = engine.cfg
        self.tok = engine.perc.tok
        self.max_len = int(self.cfg["max_len"])
        self.opt_max = int(self.cfg["opt_max_tokens"])
        self.query_first = int(self.cfg.get("query_first_max_state") or 0)
        self.provenance = {
            "package": "vegaml==" + vegaml.__version__,
            "repo": self.repo,
            "model_dir": str(engine.model_dir),
            "backbone": self.cfg.get("backbone_name"),
            # The mode actually in force. This said "engine" unconditionally, so a mode=ttt run's
            # environment.json contradicted the runtime() block reporting "ttt" for the same run.
            "mode": self.mode,
            "views": self.views,
            "budgets": {"max_len": self.max_len, "opt_max_tokens": self.opt_max,
                        "ins_max_tokens": INS_MAX_TOKENS},
            "abstain_flag": ("carried in raw_output, not acted on: every accepted question gets a "
                             "full distribution, so there is nothing to abstain from here"),
            "readout": self.mode,
        }
        self.heads = {}
        if self.mode == "ttt":
            self.fit_report = self._fit_heads(self.examples_path)
            self.provenance["ttt"] = {
                "examples": str(self.examples_path),
                "examples_sha256": file_sha256(self.examples_path),
                "min_examples": MIN_EXAMPLES,
                "cv_folds": CV_FOLDS,
                "spaces": self.fit_report,
                "note": ("one head per answer space, fitted before the run from the file above; a "
                         "space with no head or one at or below chance is answered by the engine"),
            }

    # ------------------------------------------------------------------ fitting
    def _fit_heads(self, path):
        """One head per answer space in the examples file. Fitted here, before anything is timed."""
        from vegaml import TTTHead

        grouped = {}
        for row in read_rows(path):
            gold = row.get("expected") or {}
            for key, question in (row.get("questions") or {}).items():
                want = gold.get(key)
                if want is None:
                    continue
                asked = to_vega(key, question)
                bucket = grouped.setdefault(space_key(internal(asked)),
                                            {"question": asked, "rows": []})
                # Each example keeps its OWN question, not the space's first one. An answer space is
                # the type plus the option texts, so rows in it share their options and nothing
                # else: BANKING77 carries the utterance in the instructions over an empty state, so
                # reusing one question here would hand every example an identical prompt, identical
                # features, and a head fitted on noise.
                bucket["rows"].append((row.get("state"), asked, label_text(asked, want)))

        report = {}
        for space, bucket in sorted(grouped.items()):
            question, rows = bucket["question"], bucket["rows"]
            labels = label_ids(question)
            if len(rows) < MIN_EXAMPLES:
                report[space] = {"fitted": False, "examples": len(rows),
                                 "reason": "needs at least %d labelled examples" % MIN_EXAMPLES}
                continue
            features = [self.vega.features(state, asked) for state, asked, _ in rows]
            y = [label for _, _, label in rows]
            head = TTTHead(labels).fit(features, y)
            cv = head.cross_val_accuracy(features, y, folds=CV_FOLDS)
            chance = 1.0 / max(len(labels), 2)
            # The package's own warning, enforced rather than printed: a head at or below chance has
            # learned nothing, and using it would dress noise up as a sharper answer.
            keep = cv is not None and cv > chance
            if keep:
                self.heads[space] = head
            report[space] = {"fitted": bool(keep), "examples": len(rows), "options": len(labels),
                             "cv_accuracy": cv, "chance": round(chance, 4)}
            if not keep:
                report[space]["reason"] = ("cross-validated accuracy is not above chance"
                                           if cv is not None else
                                           "too few examples per option to cross-validate")
        return report

    def _ttt_answer(self, question: dict, asked: dict, state) -> dict:
        """One answer from the fitted head for this answer space, in the kit's shape."""
        head = self.heads[space_key(internal(asked))]
        probs = head.predict(self.vega.features(state, asked), allowed=label_ids(asked))
        if question["type"] == "noul":
            return {"type": "noul", "noul": float(probs.get("true", 0.0))}
        out = {k: float(probs.get(str(k), 0.0)) for k in question["criteria"]}
        total = sum(out.values())
        if total > 0:
            out = {k: v / total for k, v in out.items()}
        return {"type": "choice", "choice": max(out, key=out.get), "probabilities": out}

    # ------------------------------------------------------------------ budgets
    def _enc(self, text: str):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def _fixed_tokens(self, key: str, question: dict) -> int:
        """Tokens the question itself costs, refusing anything build_prompt would silently cut."""
        template = TEMPLATES[0]
        texts = option_texts(question)
        ins = self._enc(template["q"].format(t=question["t"], ins=str(question["ins"])))
        if len(ins) > INS_MAX_TOKENS:
            raise Unsupported("%s: instructions longer than its native budget of %d tokens "
                              "(no truncation allowed)" % (key, INS_MAX_TOKENS))
        options = [self._enc(template["opt"].format(o=text)) for text in texts]
        if max((len(o) for o in options), default=0) > self.opt_max:
            raise Unsupported("%s: answer options longer than its per-option budget of %d tokens "
                              "(no truncation allowed)" % (key, self.opt_max))
        bos = 1 if getattr(self.tok, "bos_token_id", None) is not None else 0
        fixed = (bos + len(self._enc(template["head"])) + len(ins)
                 + sum(len(o) for o in options) + len(self._enc(template["end"])))
        if fixed > self.max_len - HEADROOM:
            raise Unsupported("%s: %d answer options do not fit its context window of %d tokens "
                              "(no truncation allowed)" % (key, len(texts), self.max_len))
        return fixed

    def _check_state(self, key: str, question: dict, fixed: int, state_tokens: int) -> None:
        extra = 0
        if self.query_first and state_tokens < self.query_first:
            prefix = self._enc(QUERY_FIRST.format(t=question["t"], ins=str(question["ins"])))
            extra = min(len(prefix), QUERY_FIRST_CAP)
        room = max(0, self.max_len - fixed - extra)
        if state_tokens > room:
            raise Unsupported("%s: prompt longer than its context window of %d tokens (state %d "
                              "tokens, room %d; no truncation allowed)"
                              % (key, self.max_len, state_tokens, room))

    # ------------------------------------------------------------------ deciding
    def __call__(self, state, questions):
        asked, shapes, fixed = {}, {}, {}
        for key, question in questions.items():
            asked[key] = to_vega(key, question)
            shapes[key] = internal(asked[key])
            fixed[key] = self._fixed_tokens(key, shapes[key])
        # One state is read for every question in the request, so it has to fit the tightest of them.
        state_tokens = len(self._enc(serialize_state(state)))
        for key, shape in shapes.items():
            self._check_state(key, shape, fixed[key], state_tokens)

        # Which readout answers which question. A head only exists for an answer space that was
        # fitted and beat chance, so this is a per-space split, never a per-row one.
        by_head = [key for key in questions if space_key(shapes[key]) in self.heads]
        head_set = set(by_head)          # by_head is scanned once per question further down
        by_engine = {key: asked[key] for key in questions if key not in head_set}

        # A head still reads the frozen backbone: Vega.features() is one pooled forward pass over the
        # same prompt. So the whole request is timed here and head questions are counted as tokens
        # too. Before this, a request answered wholly by heads reported latency_ms: None, views: None
        # and usage.input_tokens == 0, which is not a measurement of anything.
        started = time.perf_counter()
        answers, result = {}, {}
        if by_engine:
            result = self.vega.decide(state, by_engine, mode="engine", alpha=self.alpha,
                                      views=self.views)
            answers.update(to_kit_answers({k: questions[k] for k in by_engine}, result["answers"]))
        for key in by_head:
            answers[key] = self._ttt_answer(questions[key], asked[key], state)
        elapsed_ms = round(1000 * (time.perf_counter() - started), 1)
        n_tokens = int(result.get("n_tokens") or 0) if by_engine else 0
        if by_head:
            # Each head question is its own single-question read of the state, so the state is
            # re-read once per head question rather than shared between them.
            n_tokens += len(by_head) * (state_tokens + sum(fixed[k] for k in by_head) // len(by_head))

        response = {"model": "vega-" + self.repo, "answers": answers,
                    "usage": {"input_tokens": n_tokens}}
        reported = ("confidence", "abstain", "unbound", "settle_steps", "energy", "adapter", "gate")
        raw = {"latency_ms": elapsed_ms, "views": self.views,
               "readout": {key: ("ttt" if key in head_set else "engine") for key in questions},
               "engine": {key: {f: answer[f] for f in reported if f in answer}
                          for key, answer in (result.get("answers") or {}).items()}}
        return response, raw

    def runtime(self):
        out = {"vegaml": self.provenance["package"], "backbone": self.cfg.get("backbone_name"),
               "device": str(getattr(self.vega.engine, "device", "cpu")),
               "views": self.views, "mode": self.mode}
        if self.mode == "ttt":
            out["ttt_heads"] = len(self.heads)
            out["ttt_examples"] = str(self.examples_path)
        return out
