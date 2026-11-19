"""vega, read the way the learner needs it: every question of a message in one padded pass, vega's own
answers in
label form, and a fingerprint of the checkpoint so saved state is only used with the vega it was made for."""

import threading

import numpy as np


def label_of(question, answer):
    """A vegaml answer, or a value a person wrote, as the label string a head is fitted on."""
    from vegaml import label_ids

    t = question["type"]
    if isinstance(answer, dict):
        if t == "boolean":
            p = answer.get("boolean", (answer.get("probs") or {}).get("true"))
            return "true" if float(p) >= 0.5 else "false"
        if t == "score":
            return str(int(answer["score"]))
        return str(answer["choice"])
    if t == "boolean":
        if isinstance(answer, bool):
            return "true" if answer else "false"
        low = str(answer).strip().lower()
        if low in ("true", "yes", "1", "y"):
            return "true"
        if low in ("false", "no", "0", "n"):
            return "false"
        raise ValueError("not a yes or a no: %r" % (answer,))
    s = str(int(answer)) if t == "score" and not isinstance(answer, str) else str(answer).strip()
    if s not in label_ids(question):
        raise ValueError(
            "%r is not an answer to this question; its answers are %s" % (answer, label_ids(question))
        )
    return s


class Reader:
    """Wraps a loaded `vegaml.Vega`. Thread-safe: one forward pass at a time."""

    def __init__(self, vega):
        self.vega = vega
        self.engine = vega.engine
        self._lock = threading.Lock()
        self._sha = None

    @classmethod
    def load(cls, model=None, device=None, **kw):
        import vegaml

        return cls(vegaml.load(model, device=device, **kw) if model else vegaml.load(device=device, **kw))

    def fingerprint(self):
        if self._sha is None:
            core = getattr(self.engine, "engine", self.engine)
            self._sha = (core.base_sha256() if hasattr(core, "base_sha256") else self.engine.base_sha256())[
                :16
            ]
        return self._sha

    def _items(self, text, questions):
        from vegaml._common import build_prompt, option_texts

        E = self.engine
        out = []
        for q in questions.values():
            qi = E._internal(q)
            k = len(option_texts(qi))
            ids, spans = build_prompt(
                E.perc.tok,
                text,
                qi,
                E.cfg["max_len"],
                option_order=list(range(k)),
                opt_max_tokens=E.cfg["opt_max_tokens"],
                query_first_max_state=E.cfg.get("query_first_max_state", 0),
            )
            out.append({"ids": np.asarray(ids, dtype=np.int32), "spans": spans})
        return out

    def features(self, texts, questions, batch_size=8):
        """[len(texts), len(questions), d] float16: vega's pooled features, all of a message's questions in
        one pass."""
        import torch

        qn = len(questions)
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            items = [it for t in chunk for it in self._items(t, questions)]
            with self._lock, torch.no_grad():
                f, _ = self.engine.perc.pooled(
                    items, prefix_min=self.engine.cfg.get("prefix_min_tokens", 4096)
                )
            out.append(f.float().cpu().numpy().reshape(len(chunk), qn, -1).astype(np.float16))
        return np.concatenate(out) if out else np.zeros((0, qn, 0), np.float16)

    def engine_answers(self, text, questions):
        """vega's own answers (no head), as vegaml returns them."""
        with self._lock:
            return self.engine.decide(text, questions)["answers"]

    def engine_labels(self, text, questions):
        a = self.engine_answers(text, questions)
        return {q: label_of(questions[q], a[q]) for q in questions}
