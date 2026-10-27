"""The HTTP server, run against a fake checkpoint: routing, validation, batching, error mapping."""
from __future__ import annotations

import pathlib
import sys

import pytest

pytest.importorskip("fastapi", reason="pip install 'vegaml[serve]'")

from fastapi.testclient import TestClient  # noqa: E402

from vegaml.serve import create_app  # noqa: E402

Q = {"churn": {"type": "boolean", "instructions": "Does the customer threaten to leave?"}}


class _FakeVega:
    mode = "auto"
    heads: dict = {}

    def __init__(self):
        self.calls = []
        self.engine = type("E", (), {"cfg": {"backbone_name": "test/backbone", "max_len": 73000}})()

    def decide(self, state, questions, mode=None, alpha="0.1", views=1, review_below=None):
        self.calls.append({"state": state, "questions": questions, "mode": mode,
                           "alpha": alpha, "views": views, "review_below": review_below})
        if questions.get("boom"):
            raise ValueError("unknown question type 'boom'")
        return {"answers": {q: {"probs": {"false": 0.1, "true": 0.9}, "confidence": 0.9,
                                "boolean": 0.9, "prediction_set": ["true"], "abstain": False}
                            for q in questions},
                "latency_ms": 1.0, "n_tokens": 7, "views": views}


@pytest.fixture
def served():
    v = _FakeVega()
    return v, TestClient(create_app(model=v, model_name="fake"))


def test_health_says_whether_the_model_is_up(served):
    _, client = served
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_decide_round_trip(served):
    v, client = served
    r = client.post("/v1/decide", json={"state": "the state", "questions": Q,
                                        "mode": "engine", "alpha": "0.05", "views": 2})
    assert r.status_code == 200
    assert r.json()["answers"]["churn"]["confidence"] == 0.9
    assert v.calls[0] == {"state": "the state", "questions": Q, "mode": "engine",
                          "alpha": "0.05", "views": 2, "review_below": None}


def test_dict_states_pass_through_unflattened(served):
    v, client = served
    state = {"subject": "double billing", "turns": ["a", "b"]}
    assert client.post("/v1/decide", json={"state": state, "questions": Q}).status_code == 200
    assert v.calls[0]["state"] == state


def test_batch_answers_in_order_on_one_model(served):
    v, client = served
    r = client.post("/v1/decide/batch", json={"items": [
        {"state": "one", "questions": Q}, {"state": "two", "questions": Q}]})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2
    assert [c["state"] for c in v.calls] == ["one", "two"]


def test_lazy_load_happens_once(monkeypatch):
    import vegaml
    v = _FakeVega()
    loads = []
    monkeypatch.setattr(vegaml, "load", lambda name, **kw: loads.append((name, kw)) or v)
    client = TestClient(create_app(model_name="4b", load_kw={"device": "cpu"}))
    assert client.get("/health").json()["model_loaded"] is False
    for _ in range(2):
        assert client.post("/v1/decide", json={"state": "s", "questions": Q}).status_code == 200
    assert loads == [("4b", {"device": "cpu"})]
    assert client.get("/health").json()["model_loaded"] is True


def test_model_info_reports_the_checkpoint(served):
    _, client = served
    info = client.get("/v1/model").json()
    assert info["model"] == "fake" and info["backbone"] == "test/backbone"
    assert info["max_len"] == 73000 and info["mode"] == "auto" and info["heads_fitted"] == 0
    assert info["calibration"] is None and info["band"] is None
    # The audit block is what a deployment records beside each decision, so it is part of the
    # server's description of itself rather than something a caller has to assemble.
    assert info["audit"]["reader"] == "test/backbone"
    assert "engine_sha256" in info["audit"] and "vegaml" in info["audit"]


def test_every_answer_carries_the_audit_block(served):
    _, client = served
    audit = client.post("/v1/decide", json={"state": "s", "questions": Q}).json()["audit"]
    assert audit["reader"] == "test/backbone" and audit["heads_fitted"] == []
    assert audit["calibration_version"] is None


def test_review_below_reaches_the_model_and_routes_the_answer():
    v = _FakeVega()
    client = TestClient(create_app(model=v, model_name="fake"))
    client.post("/v1/decide", json={"state": "s", "questions": Q, "review_below": 0.95})
    assert v.calls[-1]["review_below"] == 0.95


def test_a_server_band_routes_when_the_caller_asks_for_nothing():
    from vegaml.band import Band

    v = _FakeVega()
    client = TestClient(create_app(model=v, model_name="fake", band=Band(default=0.95)))
    answer = client.post("/v1/decide", json={"state": "s", "questions": Q}).json()
    # The fake answers 0.9, which is under the line.
    assert answer["answers"]["churn"]["route"] == "review"
    assert answer["band"]["review"] == 1
    assert client.get("/v1/model").json()["band"] == {"default": 0.95, "questions": {}}


def test_a_loaded_calibration_rescales_the_answer_and_stamps_its_version(tmp_path):
    from vegaml.band import Band
    from vegaml.calibrate import Calibration

    cal = Calibration(0.1, {"churn": {"type": "boolean", "labels": ["false", "true"],
                                      "temperature": 3.0, "qhat": 0.4, "examples": 40}},
                      {}, Band(default=0.95), {}, {"reader": "test/backbone"})
    v = _FakeVega()
    client = TestClient(create_app(model=v, model_name="fake", calibration=cal))
    out = client.post("/v1/decide", json={"state": "s", "questions": Q}).json()
    a = out["answers"]["churn"]
    assert a["calibrated"] is True
    assert a["confidence"] < 0.9                       # a temperature above one softens it
    assert a["route"] == "review"                      # and the band still fires
    assert out["audit"]["calibration_version"] == cal.version()
    assert client.get("/v1/model").json()["calibration"]["version"] == cal.version()


def test_a_calibration_fitted_for_another_model_is_reported_not_refused(capsys):
    from vegaml.calibrate import Calibration

    cal = Calibration(0.1, {}, {}, None, {}, {"reader": "someone/else"})
    client = TestClient(create_app(model=_FakeVega(), model_name="fake", calibration=cal))
    assert client.post("/v1/decide", json={"state": "s", "questions": Q}).status_code == 200
    assert "does not match the loaded model" in capsys.readouterr().out


@pytest.mark.parametrize("body", [
    {"state": "s"},                                             # no questions
    {"questions": Q},                                           # no state
    {"state": "s", "questions": Q, "views": 5},                 # views out of range
    {"state": "s", "questions": Q, "mode": "chatbot"},          # not a mode
])
def test_bad_requests_are_422_not_500(served, body):
    _, client = served
    assert client.post("/v1/decide", json=body).status_code == 422


def test_empty_batch_is_422(served):
    _, client = served
    assert client.post("/v1/decide/batch", json={"items": []}).status_code == 422


def test_model_errors_map_to_400_with_the_message(served):
    _, client = served
    r = client.post("/v1/decide", json={"state": "s", "questions": {"boom": {"type": "boom",
                                                                             "instructions": "x"}}})
    assert r.status_code == 400
    assert "unknown question type" in r.json()["detail"]


def test_entry_point_and_extra_are_declared():
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib
    root = pathlib.Path(__file__).resolve().parent.parent
    py = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["scripts"]["vegaml-serve"] == "vegaml.serve:main"
    assert any(d.startswith("fastapi") for d in py["project"]["optional-dependencies"]["serve"])
