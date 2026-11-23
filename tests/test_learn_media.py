"""Continuous learning on pictures, recordings, clips and long conversations: records are read once, a
picture learner and a text learner stay apart, teachers are shown the pictures, and the proxy routes each
request to the learner that can read it. A stand-in reader (colour decides the team for pictures, words for
text, a recording's bytes are its words), so nothing downloads."""

import base64
import io
import json
import random

import httpx
import numpy as np
import pytest
from test_learn_teacher import WORDS, FlaggingReader, stream, truth

from vegaml.learn import Escalator, Learner, Record, as_record, claude_teacher, window

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

COLOURS = {"billing": (220, 30, 30), "technical": (30, 200, 30), "sales": (30, 30, 220)}
Q = {
    "team": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "money", "technical": "faults", "sales": "buying"},
    }
}


def picture(team, rng):
    c = np.array(COLOURS[team]) + rng.integers(-25, 25, 3)
    a = np.clip(np.ones((24, 24, 3)) * c + rng.integers(-10, 10, (24, 24, 3)), 0, 255).astype("uint8")
    return Image.fromarray(a)


def colour_team(image):
    m = np.asarray(image).reshape(-1, 3).mean(0)
    return list(COLOURS)[int(np.argmax(m))]


class MediaReader(FlaggingReader):
    """FlaggingReader for text, plus: a picture's features carry its colour's team, a recording's bytes are
    its words, a clip is b"speech:<words>" or b"frames:<team>"."""

    def __init__(self):
        self.heard = 0

    def features(self, records, questions, batch_size=8):
        out = np.zeros((len(records), len(questions), 16), np.float32)
        for i, r in enumerate(records):
            if isinstance(r, Record) and r.kind == "image":
                rng = np.random.default_rng(int(r.key[:8], 16))
                v = rng.normal(0, 0.3, 16)
                v[8 + list(WORDS).index(colour_team(r.image))] += 2.5  # picture features live elsewhere
                out[i, :] = v
            else:
                t = r if isinstance(r, str) else r.text
                out[i] = super().features([t], questions)[0]
        return out.astype(np.float16)

    def engine_answers(self, record, questions):
        return super().engine_answers("", questions)

    def transcribe(self, audio, sampling_rate=None, asr=None, language=None):
        self.heard += 1
        return audio.decode() if isinstance(audio, bytes) else str(audio)

    def video(self, clip, frames=4, asr=None, language=None, read="auto"):
        how, _, what = clip.decode().partition(":")
        if read == "frames":
            how, what = "frames", "sales"
        if how == "speech":
            return "speech", self.transcribe(what.encode())
        rng = np.random.default_rng(0)
        return "frames", [picture(what, rng) for _ in range(frames)]


def test_window_keeps_the_newest_turns():
    turns = [("user", "my card was declined")] + [
        ("assistant" if i % 2 else "user", "x" * 300) for i in range(40)
    ]
    turns.append(("user", "now it says pending"))
    full = window(turns, "full", 2000)
    assert (
        full.endswith("now it says pending") and full.startswith("[earlier turns cut]") and len(full) < 2100
    )
    recent = window(turns, "recent", 2000)
    assert recent.startswith("user: my card was declined") and "earlier turns omitted" in recent
    assert recent.endswith("user: now it says pending") and len(recent) <= 2000
    assert window(turns, "last", 2000) == "now it says pending"
    assert window([(None, "hello")]) == "hello"
    with pytest.raises(ValueError):
        window(turns, "middle")


def test_records_from_every_kind_of_input():
    r = MediaReader()
    rng = np.random.default_rng(1)
    assert as_record("hi").kind == "text"
    a = as_record({"audio": b"refund my invoice"}, r)
    assert a.kind == "text" and a.text == "[audio] refund my invoice" and a.transcript == "refund my invoice"
    p = as_record({"text": "is this covered?", "image": picture("sales", rng)}, r)
    assert p.kind == "image" and p.text == "is this covered? [image 1]" and len(p.images) == 1
    many = as_record({"images": [picture("sales", rng), picture("sales", rng)]}, r)
    assert many.kind == "image" and many.image.size[0] > 24  # tiled into one sheet
    speech = as_record({"video": b"speech:the app crashes"}, r)
    assert speech.kind == "text" and "the app crashes" in speech.text and speech.sources == ("video:speech",)
    silent = as_record({"video": b"frames:billing"}, r)
    assert silent.kind == "image" and len(silent.images) == 4
    looked = as_record({"video": b"speech:music heard as words", "video_read": "frames"}, r)
    assert looked.kind == "image" and looked.sources == ("video:frames",)
    convo = as_record(
        {
            "turns": [
                {"role": "user", "text": "hello"},
                {"role": "assistant", "text": "how can I help?"},
                {"role": "user", "text": "this arrived broken", "image": picture("technical", rng)},
            ]
        },
        r,
    )
    assert convo.kind == "image" and convo.text.splitlines()[-1] == "user: this arrived broken [image 1]"
    same = picture("sales", rng)
    assert as_record({"text": "x", "image": same}).key == as_record({"text": "x", "image": same.copy()}).key
    assert as_record({"text": "x", "image": same}).key != as_record({"text": "y", "image": same}).key
    with pytest.raises(ValueError):
        as_record({"audio": b"x"})  # no reader to transcribe with


def test_a_picture_learner_learns_and_keeps_to_pictures():
    r = MediaReader()
    rng = np.random.default_rng(2)
    L = Learner(Q, reader=r, auto=False, every=30, modality="image")
    for _ in range(4):
        for _ in range(30):
            team = random.Random(int(rng.integers(1e9))).choice(list(COLOURS))
            L.feedback({"text": "please route", "image": picture(team, rng)}, {"team": team})
        L.step()
    assert L.status()["questions"]["team"]["readout"] == "head"
    right = 0
    for _ in range(40):
        team = list(COLOURS)[int(rng.integers(3))]
        out = L.decide({"text": "please route", "image": picture(team, rng)})
        right += out["answers"]["team"]["answer"] == team
        assert out["record"]["kind"] == "image"
    assert right >= 36
    with pytest.raises(ValueError, match="modality='text'"):
        L.decide("refund my invoice")
    words = Learner(Q, reader=r, auto=False)
    with pytest.raises(ValueError, match="modality='image'"):
        words.decide({"image": picture("sales", rng)})


def test_audio_learns_on_its_transcript_and_is_transcribed_once():
    r = MediaReader()
    L = Learner(Q, reader=r, auto=False, every=25)
    calls = []

    def teacher(text, questions):
        calls.append(text)
        return {"team": truth(text.replace("[audio] ", ""))}

    desk = Escalator(L, teacher, audit=0.0)
    for text, _ in stream(150, 3):
        desk.decide({"audio": text.encode()})
    assert r.heard == 150  # once per record, though the learner, the teacher and the feedback all read it
    assert all(c.startswith("[audio] ") for c in calls)
    assert desk.report()["teacher_share_recent"] < 0.5


def test_the_teacher_is_shown_the_pictures():
    rng = np.random.default_rng(4)
    seen = {}

    class Msgs:
        def create(self, **kw):
            seen.update(kw)
            return type(
                "R",
                (),
                {
                    "stop_reason": "end_turn",
                    "content": [type("B", (), {"type": "text", "text": json.dumps({"team": "sales"})})()],
                    "usage": type("U", (), {"input_tokens": 1, "output_tokens": 1})(),
                },
            )()

    client = type("C", (), {"messages": Msgs()})()
    teach = claude_teacher(client=client, fallbacks=False)
    rec = as_record({"text": "which team? [see picture]", "images": [picture("sales", rng)] * 3})
    assert teach(rec, Q) == {"team": "sales"}
    content = seen["messages"][0]["content"]
    assert [b["type"] for b in content] == ["image", "image", "image", "text"]
    Image.open(io.BytesIO(base64.b64decode(content[0]["source"]["data"])))  # a real JPEG
    assert "3 pictures are attached" in content[-1]["text"]


# ---------------------------------------------------------------------- through the proxy
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from vegaml.learn.proxy import create_app  # noqa: E402

SCHEMA = {
    "type": "object",
    "properties": {"team": {"type": "string", "enum": list(WORDS)}},
    "required": ["team"],
}
H = {"x-vega-key": "k", "x-api-key": "sk-ant", "anthropic-version": "2023-06-01"}


def b64(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class LLM:
    def __init__(self):
        self.calls = 0

    def __call__(self, request):
        self.calls += 1
        body = json.loads(request.content)
        last = body["messages"][-1]["content"]
        team = None
        if isinstance(last, list):
            for b in last:
                if b.get("type") == "image":
                    team = colour_team(Image.open(io.BytesIO(base64.b64decode(b["source"]["data"]))))
                if b.get("type") == "input_audio":
                    team = truth(base64.b64decode(b["input_audio"]["data"]).decode())
            text = " ".join(b.get("text", "") for b in last if b.get("type") == "text")
        else:
            text = last
        if team is None:
            try:
                team = truth(text)
            except StopIteration:  # nothing to go on: any answer will do for this stand-in
                team = "billing"
        ans = json.dumps({"team": team})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json={"choices": [{"message": {"content": ans}}], "usage": {}})
        return httpx.Response(
            200,
            json={
                "type": "message",
                "content": [{"type": "text", "text": ans}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 2},
            },
        )


def app(tmp_path, llm):
    return create_app(
        MediaReader(),
        state=str(tmp_path / "s"),
        keys={"k": "acme"},
        transport=httpx.MockTransport(llm),
        every=25,
    )


def test_pictures_get_their_own_learner_and_learn(tmp_path):
    llm = LLM()
    a = app(tmp_path, llm)
    c = TestClient(a)
    rng = np.random.default_rng(5)
    srcs = []
    for i in range(240):
        team = list(COLOURS)[i % 3]
        body = {
            "model": "claude-haiku-5-5",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "hi, I have a problem"},
                {"role": "assistant", "content": "Send me a photo."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64(picture(team, rng)),
                            },
                        },
                        {"type": "text", "text": "here"},
                    ],
                },
            ],
            "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}},
        }
        r = c.post("/v1/messages", json=body, headers=H)
        assert r.status_code == 200
        srcs.append((r.headers["x-vega-source"], json.loads(r.json()["content"][0]["text"])["team"] == team))
        if i % 60 == 59:
            for d in a.state.desks.values():
                d.learner.flush(timeout=30)
    assert r.headers["x-vega-read"] == "image,text"
    tail = srcs[-60:]
    assert sum(s == "vega" for s, _ in tail) > 20 and all(ok for _, ok in tail)
    rep = c.get("/v1/vega/stats", headers=H).json()["decisions"]
    assert [d["modality"] for d in rep] == ["image"]
    # a text request with the same schema is a different learner
    c.post(
        "/v1/messages", json={**body, "messages": [{"role": "user", "content": "refund invoice"}]}, headers=H
    )
    assert sorted(d["modality"] for d in c.get("/v1/vega/stats", headers=H).json()["decisions"]) == [
        "image",
        "text",
    ]


def test_audio_in_the_openai_format_and_unread_media(tmp_path):
    llm = LLM()
    c = TestClient(app(tmp_path, llm))
    h = {"x-vega-key": "k", "authorization": "Bearer sk"}
    rf = {"type": "json_schema", "json_schema": {"name": "t", "schema": SCHEMA}}
    audio = {
        "type": "input_audio",
        "input_audio": {"data": base64.b64encode(b"refund my invoice").decode(), "format": "wav"},
    }
    r = c.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": [audio]}], "response_format": rf},
        headers=h,
    )
    assert (
        r.status_code == 200 and r.headers["x-vega-read"] == "audio" and r.headers["x-vega-source"] == "llm"
    )
    url = {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
    r = c.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": [url]}], "response_format": rf},
        headers=h,
    )
    assert r.headers["x-vega-source"] == "passthrough"  # never fetched by the proxy


def test_context_modes_are_separate_learners(tmp_path):
    c = TestClient(app(tmp_path, LLM()))
    msgs = [
        {"role": "user", "content": "my invoice is wrong"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "please fix the charge"},
    ]
    body = {
        "model": "claude-haiku-5-5",
        "max_tokens": 50,
        "messages": msgs,
        "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}},
    }
    keys = set()
    for how in ("full", "recent", "last"):
        r = c.post("/v1/messages", json=body, headers=dict(H, **{"x-vega-context": how}))
        assert r.status_code == 200
        keys.add(r.headers["x-vega-decision"])
    assert len(keys) == 3
    assert (
        c.post("/v1/messages", json=body, headers=dict(H, **{"x-vega-context": "middle"})).status_code == 400
    )
