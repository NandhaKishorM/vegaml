"""Speech in, the same typed decision out: a small local transcriber writes the state.

No open model under a billion parameters both hears audio and reads a question sheet, so the
audio path does not pretend to be one model: it is two honest stages. A small speech-recognition
checkpoint (distil-whisper by default; any Hugging Face ASR id by name, so `openai/whisper-tiny`
at 39M for multilingual traffic or `UsefulSensors/moonshine-tiny` at 27M both work) turns the
recording into text, and that text is a state :meth:`Vega.decide` already knows how to answer.
Because the engine sees text, everything the text path has applies unchanged: fitted heads,
calibration, conformal sets, the band. That is the difference from the image path, where pixels
bypass the heads.

The transcript rides along in the response. A decision made on words the caller never saw cannot
be audited, and the split of blame has to stay clean: a wrong transcript is the transcriber's
error, a wrong answer about a correct transcript is ours.

Everything runs on the machine. The checkpoint downloads once from the Hub, inference is local,
and no audio leaves the process, which is the same privacy posture as the rest of the library.
"""
from __future__ import annotations

import io
import os
from typing import Optional

import numpy as np

#: English-only and 166M parameters: small enough to ride along, strong enough to be useful.
#: Multilingual callers should name `openai/whisper-tiny` (39M) or `openai/whisper-small` (244M);
#: `UsefulSensors/moonshine-tiny` (27M, English) is the smallest that still earns its place.
DEFAULT_ASR = "distil-whisper/distil-small.en"

#: What every whisper-family feature extractor expects.
TARGET_SR = 16_000


def _resample_fft(wave: np.ndarray, sr: int, target: int = TARGET_SR) -> np.ndarray:
    """Rate conversion through the frequency domain, so no extra dependency rides in for it.

    FFT resampling is not what a DSP engineer would ship for music, but for speech headed into a
    16 kHz feature extractor it is clean enough, and the alternative was a hard dependency on
    torchaudio or scipy for every caller who hands in a waveform at the file's native rate.
    """
    wave = wave.astype(np.float32, copy=False)
    if sr == target:
        return wave
    n = max(1, int(round(len(wave) * target / sr)))
    spectrum = np.fft.rfft(wave)
    bins = n // 2 + 1
    resized = np.zeros(bins, dtype=spectrum.dtype)
    keep = min(len(spectrum), bins)
    resized[:keep] = spectrum[:keep]
    return (np.fft.irfft(resized, n=n) * (n / max(1, len(wave)))).astype(np.float32)


def to_waveform(audio, sampling_rate: Optional[int] = None) -> np.ndarray:
    """Whatever the caller has, as float32 mono at 16 kHz.

    A numpy waveform must say its `sampling_rate` out loud: a silent default would make a 44.1 kHz
    recording play at the wrong speed into the feature extractor and degrade the transcript with
    no error anywhere. A path or raw file bytes decode through soundfile
    (`pip install "vegaml[audio]"`; libsndfile 1.1+ reads mp3 as well as wav, flac and ogg).
    """
    if isinstance(audio, np.ndarray):
        if sampling_rate is None:
            raise ValueError("a numpy waveform needs sampling_rate=, because a wrong guess here "
                             "degrades the transcript silently")
        wave = audio
    elif isinstance(audio, (bytes, bytearray, str, os.PathLike)):
        # The type check comes before the import, so a caller who passed something unreadable is
        # told so even on a machine without the audio extra installed.
        try:
            import soundfile as sf
        except ImportError as e:
            raise ImportError('decoding an audio file needs soundfile: pip install "vegaml[audio]". '
                              "A raw numpy waveform with sampling_rate= works without it.") from e
        if isinstance(audio, (bytes, bytearray)):
            wave, sampling_rate = sf.read(io.BytesIO(bytes(audio)), always_2d=False)
        else:
            wave, sampling_rate = sf.read(os.fspath(audio), always_2d=False)
    else:
        raise TypeError("audio must be a path, raw audio-file bytes, or a 1-D numpy waveform, "
                        "got %r" % type(audio).__name__)
    wave = np.asarray(wave)
    if wave.ndim == 2:          # stereo: average, do not pick a side
        wave = wave.mean(axis=1)
    if wave.ndim != 1:
        raise ValueError("a waveform is one-dimensional; got shape %r" % (wave.shape,))
    if np.issubdtype(wave.dtype, np.integer):
        wave = wave.astype(np.float32) / float(np.iinfo(wave.dtype).max)
    return _resample_fft(wave.astype(np.float32, copy=False), int(sampling_rate))


class Transcriber:
    """One lazily built ASR pipeline, reused across calls, swapped only when a new id is named."""

    def __init__(self, model: str = DEFAULT_ASR, device: str = "cpu"):
        from transformers import pipeline as hf_pipeline

        self.model = model
        self._pipe = hf_pipeline("automatic-speech-recognition", model=model, device=device)

    def transcribe(self, wave: np.ndarray, language: Optional[str] = None,
                   chunk_s: Optional[int] = 30) -> str:
        kw = {}
        if chunk_s:
            # Long-form audio in 30 s windows; whisper-family models were trained on exactly that.
            kw["chunk_length_s"] = chunk_s
        if language:
            # Multilingual whisper checkpoints take a language; English-only and moonshine ones
            # refuse it, and transformers says so more precisely than a guard here could.
            kw["generate_kwargs"] = {"language": language}
        out = self._pipe({"array": wave, "sampling_rate": TARGET_SR}, **kw)
        return (out.get("text") or "").strip()
