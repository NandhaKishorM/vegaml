"""Video in, the same typed decision out: the sound track if it carries speech, frames if not.

No open model under a billion parameters watches a clip and reads a question sheet, so the
video path does not pretend to be one. It splits the clip into the two things the engine can
already read, and chooses between them by what the clip contains:

* **Speech.** The sound track is pulled out at 16 kHz mono and transcribed. If the transcript
  holds real words, the decision is made on it through the text path, so fitted heads, the
  calibration, the conformal set and the route all apply, exactly as for audio. The transcript
  rides along, because a decision made on words the caller never saw cannot be audited.
* **Frames.** A silent clip (or one whose sound track is music, noise, or nothing) has a few
  evenly spaced frames sampled and tiled into one contact sheet, which goes through the image
  path. That is the vision tower's zero-shot readout: heads and calibration do not apply to
  pixels, and a policy says so with ``calibrated: False``.

Decoding goes through PyAV (``pip install "vegaml[video]"``), which links ffmpeg's own
libraries: no binary on the PATH and no subprocess, which this package never starts. A machine
without it is told exactly that in one sentence rather than failing somewhere inside a codec.
Everything runs locally: the clip never leaves the process.

    from vegaml._video import probe, extract_audio, sample_frames, contact_sheet
"""
from __future__ import annotations

import io
import os
import re
from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["available", "probe", "extract_audio", "sample_frames", "contact_sheet",
           "has_speech", "as_source", "TARGET_SR"]

#: What whisper-family transcribers expect.
TARGET_SR = 16_000

#: Below this root-mean-square level the sound track is treated as silent and never sent to the
#: transcriber, which would otherwise invent words for it.
SILENCE_RMS = 1e-3

#: A transcript with fewer words than this is noise the transcriber dressed up, not speech.
MIN_WORDS = 3

_NO_AV = ('video needs PyAV: pip install "vegaml[video]" (it links ffmpeg\'s libraries; no '
          "binary on the PATH is needed). Images and audio work without it.")


def available() -> bool:
    try:
        import av  # noqa: F401
    except ImportError:
        return False
    return True


def _av():
    try:
        import av
    except ImportError as e:
        raise ImportError(_NO_AV) from e
    return av


def as_source(video):
    """Something PyAV opens: a path as given, or raw file bytes wrapped in a seekable buffer."""
    if isinstance(video, (bytes, bytearray)):
        return io.BytesIO(bytes(video))
    if isinstance(video, (str, os.PathLike)):
        path = os.fspath(video)
        if not os.path.isfile(path):
            raise FileNotFoundError("no video file at %r" % path)
        return path
    raise TypeError("video must be a path or raw video-file bytes, got %r" % type(video).__name__)


def _rewind(src):
    if hasattr(src, "seek"):
        src.seek(0)
    return src


def probe(src) -> dict:
    """Duration in seconds (0.0 when the container states none) and which streams it has."""
    av = _av()
    with av.open(_rewind(src)) as c:
        dur = float(c.duration) / 1_000_000 if c.duration else 0.0
        if not dur and c.streams.video:
            s = c.streams.video[0]
            if s.duration and s.time_base:
                dur = float(s.duration * s.time_base)
        return {"duration_s": dur, "has_audio": bool(c.streams.audio),
                "has_video": bool(c.streams.video)}


def extract_audio(src) -> Optional[np.ndarray]:
    """The sound track as a 1-D float32 waveform at 16 kHz mono, or None when there is none."""
    av = _av()
    with av.open(_rewind(src)) as c:
        if not c.streams.audio:
            return None
        resampler = av.AudioResampler(format="flt", layout="mono", rate=TARGET_SR)
        parts = []
        for frame in c.decode(c.streams.audio[0]):
            for out in resampler.resample(frame):
                parts.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            parts.append(out.to_ndarray().reshape(-1))
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32)


def has_speech(wave: Optional[np.ndarray], transcript: Optional[str] = None) -> bool:
    """The gate that chooses the path: a sound track quiet enough to be silence never reaches
    the transcriber, and a transcript shorter than a few words is not treated as speech."""
    if wave is None or wave.size == 0:
        return False
    if float(np.sqrt(np.mean(wave.astype(np.float64) ** 2))) < SILENCE_RMS:
        return False
    if transcript is None:
        return True            # loud enough to be worth transcribing; the caller decides after
    # Words with letters or digits in them. A tone or music comes back from whisper as rows of
    # dots or dashes, which split() counts as words and this does not.
    return len(re.findall(r"[^\W_]+", transcript)) >= MIN_WORDS


def sample_frames(src, n: int = 4, duration_s: Optional[float] = None) -> Tuple[List, List[float]]:
    """``n`` frames at evenly spaced times, as PIL RGB images, with the times they came from.

    Times sit at the centres of ``n`` equal slices, so a two-frame sample of a ten-second clip
    reads 2.5 s and 7.5 s rather than the first and last frame, which are often black. Each
    target takes the nearest frame; the times returned are the frames' own.
    """
    av = _av()
    try:
        from PIL import Image  # noqa: F401
    except ImportError as e:
        raise ImportError("video frames need Pillow: pip install pillow") from e
    if duration_s is None:
        duration_s = probe(src)["duration_s"]
    n = max(1, int(n))
    targets = [0.0] if duration_s <= 0 else [duration_s * (i + 0.5) / n for i in range(n)]
    frames, got = [], []
    with av.open(_rewind(src)) as c:
        if not c.streams.video:
            raise ValueError("the clip has no video stream")
        stream = c.streams.video[0]
        want = 0
        prev = None             # (frame, time) just before the current one
        for frame in c.decode(stream):
            t = float(frame.time) if frame.time is not None else 0.0
            while want < len(targets) and t >= targets[want] - 1e-6:
                # the nearer of this frame and the one before it
                pick = prev if prev is not None and targets[want] - prev[1] < t - targets[want] \
                    else (frame, t)
                frames.append(pick[0].to_image().convert("RGB"))
                got.append(pick[1])
                want += 1
            if want >= len(targets):
                break
            prev = (frame, t)
        while prev is not None and want < len(targets):
            # targets past the last frame: the clip's final frame is the nearest
            frames.append(prev[0].to_image().convert("RGB"))
            got.append(prev[1])
            want += 1
    if not frames:
        raise RuntimeError("no frame could be read from the clip")
    return frames, got


def contact_sheet(frames: Sequence, cell: int = 512):
    """The frames tiled left to right, top to bottom, into one image the vision tower reads
    at once. Each frame is fitted into a ``cell`` square, keeping its aspect ratio."""
    from PIL import Image
    if not frames:
        raise ValueError("no frames to tile")
    n = len(frames)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    sheet = Image.new("RGB", (cols * cell, rows * cell), (0, 0, 0))
    for i, fr in enumerate(frames):
        fr = fr.copy()
        fr.thumbnail((cell, cell))
        x = (i % cols) * cell + (cell - fr.width) // 2
        y = (i // cols) * cell + (cell - fr.height) // 2
        sheet.paste(fr, (x, y))
    return sheet
