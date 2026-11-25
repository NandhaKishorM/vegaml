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
        self._vision = None  # the vision path, loaded on the first image
        self._asr = {}  # transcribers by checkpoint, loaded on the first recording

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

    def features(self, records, questions, batch_size=8):
        """[len(records), len(questions), d] float16: vega's pooled features, all of a record's questions in
        one pass. A record is a string or a vegaml_learn.Record; pictures go through the vision path."""
        import torch

        qn = len(questions)
        recs = [r if isinstance(r, str) else (r.text if r.kind == "text" else r) for r in records]
        texts = [i for i, r in enumerate(recs) if isinstance(r, str)]
        out = [None] * len(recs)
        for i in range(0, len(texts), batch_size):
            idx = texts[i : i + batch_size]
            items = [it for j in idx for it in self._items(recs[j], questions)]
            with self._lock, torch.no_grad():
                f, _ = self.engine.perc.pooled(
                    items, prefix_min=self.engine.cfg.get("prefix_min_tokens", 4096)
                )
            f = f.float().cpu().numpy().reshape(len(idx), qn, -1).astype(np.float16)
            for r, j in enumerate(idx):
                out[j] = f[r]
        for j, r in enumerate(recs):
            if out[j] is None:
                with self._lock:
                    out[j] = self.vision().features(r.text, questions, r.image).astype(np.float16)
        return np.stack(out) if out else np.zeros((0, qn, 0), np.float16)

    def engine_answers(self, record, questions):
        """vega's own answers (no head), as vegaml returns them."""
        if not isinstance(record, str) and record.kind == "text":
            record = record.text
        with self._lock:
            if isinstance(record, str):
                return self.engine.decide(record, questions)["answers"]
            return self.vision().decide_image_text(record.text, questions, record.image)["answers"]

    # ------------------------------------------------------------------ media
    def vision(self):
        """vega's vision path, loaded on first use (a text-only learner never pays for it)."""
        if self._vision is None:
            from vegaml._vision import VegaVision

            e = self.engine
            self._vision = VegaVision(
                e.model_dir, backbone=e.backbone, device=str(getattr(e, "device", "cpu"))
            )
        return self._vision

    def transcribe(self, audio, sampling_rate=None, asr=None, language=None):
        """What a recording says, through a small local transcriber (vegaml's default: distil-whisper)."""
        from vegaml._audio import DEFAULT_ASR, Transcriber, to_waveform

        want = asr or DEFAULT_ASR
        with self._lock:
            if want not in self._asr:
                self._asr[want] = Transcriber(want, device=str(getattr(self.engine, "device", "cpu")))
            words = self._asr[want].transcribe(to_waveform(audio, sampling_rate), language=language)
        if not words:
            raise ValueError("the transcriber heard no speech in that audio; nothing to decide on")
        return words

    def video(self, clip, frames=4, asr=None, language=None, read="auto"):
        """("speech", transcript) when the clip's sound track carries words, else ("frames", [images]).
        read: "auto", or "frames" / "speech" to choose (see vegaml_learn.as_record)."""
        from vegaml import _video

        if read not in ("auto", "frames", "speech"):
            raise ValueError("read a clip as auto, frames or speech, not %r" % (read,))
        src = _video.as_source(clip)
        info = _video.probe(src)
        wave = _video.extract_audio(src) if read != "frames" else None
        if read == "speech":
            return "speech", self.transcribe(wave, sampling_rate=_video.TARGET_SR, asr=asr, language=language)
        if read == "auto" and _video.has_speech(wave):
            try:
                words = self.transcribe(wave, sampling_rate=_video.TARGET_SR, asr=asr, language=language)
            except ValueError:
                words = ""
            if words and _video.has_speech(wave, words):
                return "speech", words
        if not info.get("has_video", True):
            raise ValueError("the clip has no picture and no speech; there is nothing to decide on")
        sampled, _ = _video.sample_frames(src, frames, info["duration_s"])
        return "frames", [f.convert("RGB") for f in sampled]

    def engine_labels(self, text, questions):
        a = self.engine_answers(text, questions)
        return {q: label_of(questions[q], a[q]) for q in questions}
