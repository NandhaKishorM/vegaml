"""What the learner reads: text, a picture (with any words beside it), audio, a video clip, or a conversation
whose turns mix them. Each is turned, once, into one of the two things vega reads:

* text: a message, a conversation written out turn by turn, an audio recording's transcript, or a clip's
  speech;
* a picture with text beside it: one image, several images tiled into one sheet, or a silent clip's frames.

    from vegaml.learn import Learner
    words = Learner(questions)                       # text, audio and clips with speech
    pictures = Learner(questions, modality="image")  # images and silent clips

    words.decide({"audio": "call.wav"})
    pictures.decide({"text": "Is this damage covered?", "image": "parcel.jpg"})
    words.decide({"turns": [{"role": "user", "text": "My card was declined"},
                            {"role": "assistant", "text": "Where were you paying?"},
                            {"role": "user", "text": "In Spain, and now it says pending"}]})

A picture and a text are read through different paths, so their features are not comparable: a learner
reads one modality, and a record of the other kind is refused with a message saying which learner takes it.

Long conversations: `context` picks what vega reads of the turns. "full" reads them all (cut from the
start, so the newest turns survive, past `budget` characters); "recent" reads the opening message and as
many of the newest turns as fit in `budget`; "last" reads the last turn alone.
"""

import base64
import hashlib
import io

import numpy as np

CONTEXTS = ("full", "recent", "last")
DEFAULT_BUDGET = 12000
MEDIA = ("image", "images", "audio", "video")


class Record:
    """One record, ready to read. kind: "text" or "image". text: what vega reads (for an image, the words
    beside it). images: the pictures as given (a clip's frames), up to eight, for a teacher that looks at
    them; image: the one picture vega reads (several are tiled). transcript: what was heard, if anything,
    so a decision made on words nobody saw can be audited."""

    __slots__ = ("kind", "text", "image", "images", "transcript", "sources", "key")

    def __init__(self, kind, text="", image=None, images=(), transcript=None, sources=("text",)):
        if kind not in ("text", "image"):
            raise ValueError("a record is text or an image, not %r" % (kind,))
        self.kind, self.text, self.image = kind, str(text or ""), image
        self.images, self.transcript, self.sources = list(images), transcript, tuple(sources)
        h = hashlib.sha1(kind.encode())
        h.update(self.text.encode("utf-8"))
        if image is not None:
            a = np.asarray(image)
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
        self.key = h.hexdigest()

    def __repr__(self):
        return "Record(%s, %r%s)" % (self.kind, self.text[:40], ", image" if self.image is not None else "")

    def describe(self):
        out = {"kind": self.kind, "sources": list(self.sources)}
        if self.transcript:
            out["transcript"] = self.transcript
        return out


def _rgb(image):
    from vegaml._vision import _as_rgb_image

    if isinstance(image, (bytes, bytearray)):
        from PIL import Image

        with Image.open(io.BytesIO(image)) as im:
            return im.convert("RGB")
    return _as_rgb_image(image)


def decode_data_url(url):
    """bytes of a data: URL (base64), or None for anything else (an http URL is not fetched here)."""
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    head, _, body = url.partition(",")
    if ";base64" not in head:
        return None
    return base64.b64decode(body)


def window(turns, context="full", budget=DEFAULT_BUDGET):
    """The conversation as vega reads it: [(role, text)] written out turn by turn, within `budget`
    characters, keeping the newest turns."""
    if context not in CONTEXTS:
        raise ValueError("context is one of %s, not %r" % (CONTEXTS, context))
    if not turns:
        return ""
    lines = ["%s: %s" % (r, t) if r else t for r, t in turns]
    if len(lines) == 1:
        return lines[0][-budget:]
    if context == "last":
        return turns[-1][1][-budget:]
    if context == "full":
        text = "\n".join(lines)
        return text if len(text) <= budget else "[earlier turns cut]\n" + text[-budget:]
    # recent: the opening message (it usually says what the problem is) and the newest turns that fit
    first = lines[0][: budget // 4]
    room = budget - len(first)
    kept = []
    for line in reversed(lines[1:]):
        if len(line) + 1 > room:
            break
        kept.append(line)
        room -= len(line) + 1
    kept.reverse()
    if not kept:  # one huge newest turn: keep its end
        kept = [lines[-1][-max(room, 200) :]]
    gap = len(lines) - 1 - len(kept)
    mid = ["[%d earlier turns omitted]" % gap] if gap > 0 else []
    return "\n".join([first] + mid + kept)


def as_record(
    x, reader=None, *, context="full", budget=DEFAULT_BUDGET, frames=4, asr=None, language=None, video="auto"
):
    """A Record from what a caller has: a string, a Record, a dict with any of text / image / images /
    audio (with sampling_rate for a raw waveform) / video, or {"turns": [...]} whose turns carry a role
    and any of those. Audio and video need `reader` (a vegaml.learn.Reader) to transcribe and sample.

    video: how a clip is read. "auto" reads its speech when the sound track holds words, its frames
    otherwise; "frames" always looks (a transcriber can hear words in music, and an activity is in the
    picture); "speech" always listens."""
    if isinstance(x, Record):
        return x
    if isinstance(x, str):
        return Record("text", x)
    if not isinstance(x, dict):
        raise TypeError("a record is a string, a Record or a dict, not %s" % type(x).__name__)
    context = x.get("context", context)
    budget = int(x.get("budget", budget))
    video = x.get("video_read", video)
    turns = x["turns"] if "turns" in x else [dict(x, role=None)]
    lines, images, heard, sources = [], [], [], []
    for turn in turns:
        if not isinstance(turn, dict):
            raise TypeError("each turn is a dict with a role and text / image / audio / video")
        parts = []
        if turn.get("text"):
            parts.append(str(turn["text"]))
            sources.append("text")
        pics = list(turn.get("images") or []) + ([turn["image"]] if turn.get("image") is not None else [])
        for p in pics:
            images.append(_rgb(p))
            parts.append("[image %d]" % len(images))
            sources.append("image")
        if turn.get("audio") is not None:
            if reader is None:
                raise ValueError("audio needs a reader to transcribe it")
            words = reader.transcribe(
                turn["audio"], sampling_rate=turn.get("sampling_rate"), asr=asr, language=language
            )
            heard.append(words)
            parts.append("[audio] %s" % words)
            sources.append("audio")
        if turn.get("video") is not None:
            if reader is None:
                raise ValueError("video needs a reader to decode it")
            how, got = reader.video(
                turn["video"], frames=frames, asr=asr, language=language, read=turn.get("video_read", video)
            )
            if how == "speech":
                heard.append(got)
                parts.append("[video, speech] %s" % got)
                sources.append("video:speech")
            else:
                start = len(images) + 1
                images.extend(got)
                parts.append("[video: frames %d-%d]" % (start, len(images)))
                sources.append("video:frames")
        lines.append((turn.get("role"), " ".join(parts)))
    text = window(lines, context, budget)
    transcript = "\n".join(heard) or None
    if not images:
        return Record("text", text, transcript=transcript, sources=sources)
    if len(images) == 1:
        sheet = images[0]
    else:
        from vegaml._video import contact_sheet

        sheet = contact_sheet(images[:16])
    return Record("image", text, image=sheet, images=images[:8], transcript=transcript, sources=sources)


def image_bytes(image, max_side=1568, quality=88):
    """JPEG bytes of a picture, its long side at most `max_side` (what hosted models read in full)."""
    im = image.copy()
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
