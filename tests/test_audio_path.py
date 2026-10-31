"""The audio path: a small transcriber in front of the text engine, with nothing hidden.

The design claim worth testing is the split of responsibilities. The transcriber turns speech into
text and the text rides the ordinary :meth:`Vega.decide`, so fitted heads, modes and the band all
apply, and the transcript travels in the response so the caller can audit what was decided about.
These tests drive `decide_audio` against stubs, so they need no checkpoint, no ASR download and no
soundfile install; what needs a real model is the transcription quality, which is the checkpoint's
to prove, not ours.
"""
from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

import vegaml
from vegaml import Vega
from vegaml._audio import DEFAULT_ASR, TARGET_SR, to_waveform

PKG = pathlib.Path(vegaml.__file__).parent


class _Engine:
    cfg = {"max_len": 73728}
    model_dir = "/nowhere"
    backbone = "dummy"
    device = "cpu"


class _Transcriber:
    """Stands in for the ASR pipeline: fixed text out, records what came in."""

    def __init__(self, text: str):
        self.model = DEFAULT_ASR
        self.text = text
        self.heard = None

    def transcribe(self, wave, language=None, chunk_s=30):
        self.heard = wave
        return self.text


# ------------------------------------------------------------------ waveform intake
def test_a_numpy_waveform_must_say_its_sampling_rate():
    """A silent default rate would feed a 44.1 kHz recording into a 16 kHz feature extractor and
    degrade the transcript with no error anywhere. The only safe failure is a loud one."""
    with pytest.raises(ValueError, match="sampling_rate"):
        to_waveform(np.zeros(1600, dtype=np.float32))


def test_resampling_lands_on_the_extractor_rate():
    one_second_at_8k = np.sin(np.linspace(0, 440 * 2 * np.pi, 8000)).astype(np.float32)
    wave = to_waveform(one_second_at_8k, sampling_rate=8000)
    assert abs(len(wave) - TARGET_SR) <= 2, "one second in is one second out, at 16 kHz"
    assert wave.dtype == np.float32


def test_already_at_rate_passes_through_untouched():
    wave = np.random.default_rng(0).standard_normal(TARGET_SR).astype(np.float32)
    assert to_waveform(wave, sampling_rate=TARGET_SR) is wave


def test_int16_pcm_is_scaled_to_unit_floats():
    loud = np.full(100, np.iinfo(np.int16).max, dtype=np.int16)
    wave = to_waveform(loud, sampling_rate=TARGET_SR)
    assert wave.dtype == np.float32 and abs(float(wave.max()) - 1.0) < 1e-4


def test_stereo_is_averaged_not_truncated():
    left, right = np.ones(100, dtype=np.float32), np.zeros(100, dtype=np.float32)
    wave = to_waveform(np.stack([left, right], axis=1), sampling_rate=TARGET_SR)
    assert wave.ndim == 1 and abs(float(wave.mean()) - 0.5) < 1e-6


def test_an_unreadable_type_is_named_in_the_error():
    with pytest.raises(TypeError, match="dict"):
        to_waveform({"not": "audio"}, sampling_rate=TARGET_SR)


# ------------------------------------------------------------------ decide_audio
def test_text_only_callers_never_pay_for_the_transcriber():
    assert Vega(_Engine())._audio is None


def test_the_transcript_rides_the_text_path_and_the_response():
    """The whole point of the two-stage design: the engine sees the words, the caller sees the
    words, and the response says which checkpoint produced them."""
    v = Vega(_Engine())
    v._audio = _Transcriber("refund the customer and close the ticket")
    seen = {}

    def fake_decide(state, questions, **kw):
        seen["state"], seen["kw"] = state, kw
        return {"answers": {"act": {"choice": "refund"}}, "mode": "auto"}

    v.decide = fake_decide
    out = v.decide_audio(np.zeros(TARGET_SR, dtype=np.float32),
                         {"act": {"type": "choice"}}, sampling_rate=TARGET_SR,
                         mode="auto", review_below=0.8)
    assert seen["state"] == "refund the customer and close the ticket"
    assert seen["kw"] == {"mode": "auto", "review_below": 0.8}, \
        "decide_audio forwards the text path's knobs, because the text path is doing the deciding"
    assert out["transcript"] == "refund the customer and close the ticket"
    assert out["asr"] == DEFAULT_ASR
    assert out["answers"]["act"]["choice"] == "refund"


def test_silence_is_an_error_not_a_confident_answer_about_nothing():
    v = Vega(_Engine())
    v._audio = _Transcriber("")
    v.decide = lambda *a, **k: pytest.fail("decide must not run on an empty transcript")
    with pytest.raises(ValueError, match="no speech"):
        v.decide_audio(np.zeros(TARGET_SR, dtype=np.float32), {"q": {}}, sampling_rate=TARGET_SR)


def test_a_loaded_transcriber_is_reused_until_a_different_checkpoint_is_named():
    """Reuse is what makes the second call cheap; the swap is what makes `asr=` honest. The source
    is checked rather than loading two real checkpoints in a unit test."""
    src = (PKG / "__init__.py").read_text(encoding="utf-8")
    fn = ast.unparse(next(n for n in ast.walk(ast.parse(src))
                          if isinstance(n, ast.FunctionDef) and n.name == "decide_audio"))
    assert "self._audio is None or self._audio.model != want" in fn


def test_the_audio_modules_relative_imports_stay_inside_the_package():
    """The vision path shipped a bare `vega_common` import once; the audio path gets the same pin."""
    for node in ast.walk(ast.parse((PKG / "_audio.py").read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("vega"):
            assert node.level >= 1, "`from %s import ...` escapes the package" % node.module


def test_soundfile_is_an_extra_and_the_error_says_how_to_get_it():
    """A waveform caller needs no decoder, so soundfile must not be imported at module import time,
    and a file caller without it must be told the install line, not shown a bare ImportError."""
    tree = ast.parse((PKG / "_audio.py").read_text(encoding="utf-8"))
    top_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    named = {a.name for n in top_level for a in n.names if isinstance(n, ast.Import)}
    assert "soundfile" not in named and "transformers" not in named
    assert 'vegaml[audio]' in (PKG / "_audio.py").read_text(encoding="utf-8")
