"""vega Proxy: decisions go to the LLM until vega has learned them, everything else passes through, and
the app's LLM key is forwarded but never stored. A stand-in reader and a stand-in LLM (an httpx mock
transport), so nothing downloads and nothing is sent anywhere."""

import json

import pytest

httpx = pytest.importorskip("httpx")  # the proxy extra: vegaml[proxy]

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402
from test_learn_teacher import WORDS, FlaggingReader, stream, truth  # noqa: E402

from vegaml.learn.proxy import (  # noqa: E402
    anthropic_spec,
    cost_of,
    create_app,
    openai_spec,
    questions_from_schema,
)

SCHEMA = {
    "type": "object",
    "properties": {"team": {"type": "string", "enum": list(WORDS), "description": "Which team?"}},
    "required": ["team"],
    "additionalProperties": False,
}
SYSTEM = "Route the support ticket to a team."


def ant_body(text, model="claude-haiku-5-5"):
    return {
        "model": model,
        "max_tokens": 256,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": text}],
        "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}},
    }


def oai_body(text, model="gpt-x"):
    return {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "t", "schema": SCHEMA}},
    }


class LLM:
    """Answers every ticket correctly, in whichever API format it was asked in."""

    def __init__(self, status=200, refuse=False):
        self.calls, self.status, self.refuse, self.seen = 0, status, refuse, []

    def __call__(self, request):
        self.calls += 1
        self.seen.append(dict(request.headers))
        body = json.loads(request.content) if request.content else {}
        if self.status != 200:
            return httpx.Response(
                self.status, json={"type": "error", "error": {"type": "authentication_error"}}
            )
        if request.url.path == "/v1/messages":
            if "output_config" not in body:
                return httpx.Response(
                    200, json={"type": "message", "content": [{"type": "text", "text": "hi"}], "usage": {}}
                )
            text = body["messages"][-1]["content"]
            usage = {"input_tokens": 1000, "output_tokens": 20}
            if self.refuse:
                return httpx.Response(
                    200, json={"type": "message", "stop_reason": "refusal", "content": [], "usage": usage}
                )
            ans = json.dumps({"team": truth(text)})
            return httpx.Response(
                200,
                json={
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": body["model"],
                    "content": [{"type": "text", "text": ans}],
                    "stop_reason": "end_turn",
                    "usage": usage,
                },
            )
        if request.url.path == "/v1/chat/completions":
            text = body["messages"][-1]["content"]
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"role": "assistant", "content": json.dumps({"team": truth(text)})}}
                    ],
                    "usage": {"prompt_tokens": 900, "completion_tokens": 10},
                },
            )
        return httpx.Response(200, json={"data": [{"id": "m"}]})


def app_with(tmp_path, llm, **kw):
    kw.setdefault("keys", {"key-a": "acme", "key-b": "globex"})
    return create_app(
        FlaggingReader(),
        state=str(tmp_path / "state"),
        transport=httpx.MockTransport(llm),
        every=25,
        **kw,
    )


H = {"x-vega-key": "key-a", "x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"}


def run_stream(client, n, seed, headers=H, path="/v1/messages", body=ant_body):
    out = []
    for text, gold in stream(n, seed):
        r = client.post(path, json=body(text), headers=headers)
        assert r.status_code == 200, r.text
        if path == "/v1/messages":
            got = json.loads(r.json()["content"][0]["text"])
        else:
            got = json.loads(r.json()["choices"][0]["message"]["content"])
        out.append((r.headers["x-vega-source"], got["team"] == gold))
    return out


def flush(app):
    for d in app.state.desks.values():
        d.learner.flush(timeout=30)


def test_the_llm_teaches_and_then_is_needed_less(tmp_path):
    llm = LLM()
    app = app_with(tmp_path, llm)
    c = TestClient(app)
    first = run_stream(c, 100, 1)
    assert all(src == "llm" for src, _ in first[:20])  # vega starts out unbound: everything is the LLM's
    for _ in range(3):
        flush(app)
        run_stream(c, 100, 2)
    flush(app)
    last = run_stream(c, 200, 3)
    by_vega = [ok for src, ok in last if src == "vega"]
    assert len(by_vega) > 120  # the LLM now sees only a small share
    assert sum(by_vega) / len(by_vega) > 0.95  # and what vega answers alone is right
    rep = c.get("/v1/vega/stats", headers=H).json()["decisions"][0]
    assert (
        rep["served_by_vega"] + rep["llm_calls"] == rep["requests"] == 600 and rep["llm_calls"] == llm.calls
    )
    assert rep["saved_usd"] > 0 and rep["llm_usd"] > 0
    # the app's LLM key went upstream; the vega key did not
    assert all(s.get("x-api-key") == "sk-ant-test" and "x-vega-key" not in s for s in llm.seen)


def test_a_vega_answer_is_a_well_formed_response(tmp_path):
    app = app_with(tmp_path, LLM())
    c = TestClient(app)
    for _ in range(4):
        run_stream(c, 100, 5)
        flush(app)
    r = next(
        r
        for r in (c.post("/v1/messages", json=ant_body(t), headers=H) for t, _ in stream(50, 6))
        if r.headers["x-vega-source"] == "vega"
    )
    m = r.json()
    assert m["type"] == "message" and m["role"] == "assistant" and m["stop_reason"] == "end_turn"
    assert m["model"] == "claude-haiku-5-5" and m["usage"] == {"input_tokens": 0, "output_tokens": 0}
    assert set(json.loads(m["content"][0]["text"])) == {"team"}
    assert r.headers["x-vega-id"].startswith("vg_") and float(r.headers["x-vega-confidence"]) > 0


def test_anything_that_is_not_a_decision_passes_through(tmp_path):
    llm = LLM()
    c = TestClient(app_with(tmp_path, llm))
    chat = {"model": "claude-haiku-5-5", "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}]}
    r = c.post("/v1/messages", json=chat, headers=H)
    assert r.status_code == 200 and r.headers["x-vega-source"] == "passthrough"
    assert r.json()["content"][0]["text"] == "hi"
    tools = dict(ant_body("refund please"), tools=[{"name": "x", "input_schema": {"type": "object"}}])
    assert c.post("/v1/messages", json=tools, headers=H).headers["x-vega-source"] == "passthrough"
    assert c.get("/v1/models", headers=H).headers["x-vega-source"] == "passthrough"
    assert c.get("/v1/vega/stats", headers=H).json()["decisions"] == []


def test_shadow_asks_the_llm_every_time_and_measures_vega(tmp_path):
    llm = LLM()
    app = app_with(tmp_path, llm, mode="shadow")
    c = TestClient(app)
    for _ in range(4):
        out = run_stream(c, 100, 7)
        flush(app)
    assert all(src == "llm" for src, _ in out) and llm.calls == 400
    rep = c.get("/v1/vega/stats", headers=H).json()["decisions"][0]
    assert rep["mode"] == "shadow" and rep["served_by_vega"] == 0
    assert rep["shadow_vega_would_answer"] > 50
    assert rep["questions"]["team"]["agreement_with_llm"] > 0.9
    # switching the decision to live lets vega answer what it is sure of
    c.post("/v1/vega/mode", json={"decision": rep["decision"], "mode": "live"}, headers=H)
    assert any(src == "vega" for src, _ in run_stream(c, 100, 8))


def test_vega_mode_never_calls_the_llm_and_off_never_asks_vega(tmp_path):
    llm = LLM()
    c = TestClient(app_with(tmp_path, llm))
    r = c.post(
        "/v1/messages", json=ant_body("refund the invoice"), headers=dict(H, **{"x-vega-mode": "vega"})
    )
    assert r.headers["x-vega-source"] == "vega" and llm.calls == 0
    r = c.post("/v1/messages", json=ant_body("refund the invoice"), headers=dict(H, **{"x-vega-mode": "off"}))
    assert r.headers["x-vega-source"] == "passthrough" and llm.calls == 1
    assert (
        c.post("/v1/messages", json=ant_body("x"), headers=dict(H, **{"x-vega-mode": "nope"})).status_code
        == 400
    )


def test_llm_errors_and_refusals_are_returned_as_they_are(tmp_path):
    c = TestClient(app_with(tmp_path, LLM(status=401)))
    r = c.post("/v1/messages", json=ant_body("refund the invoice"), headers=H)
    assert r.status_code == 401 and r.headers["x-vega-source"] == "llm"
    app = app_with(tmp_path / "b", LLM(refuse=True))
    c = TestClient(app)
    r = c.post("/v1/messages", json=ant_body("refund the invoice"), headers=H)
    assert r.json()["stop_reason"] == "refusal"
    rep = c.get("/v1/vega/stats", headers=H).json()["decisions"][0]
    assert rep["llm_refused"] == 1 and list(app.state.desks.values())[0].learner.pending() == 0


def test_keys_and_tenants(tmp_path):
    app = app_with(tmp_path, LLM())
    c = TestClient(app)
    assert c.post("/v1/messages", json=ant_body("x"), headers={"x-api-key": "k"}).status_code == 401
    assert (
        c.post("/v1/messages", json=ant_body("x"), headers=dict(H, **{"x-vega-key": "bad"})).status_code
        == 401
    )
    r = c.post("/v1/messages", json=ant_body("refund invoice"), headers=H)
    rid = r.headers["x-vega-id"]
    other = dict(H, **{"x-vega-key": "key-b"})
    assert c.get("/v1/vega/stats", headers=other).json()["decisions"] == []
    assert (
        c.post(
            "/v1/vega/feedback", json={"id": rid, "answers": {"team": "billing"}}, headers=other
        ).status_code
        == 404
    )
    ok = c.post("/v1/vega/feedback", json={"id": rid, "answers": {"team": "billing"}}, headers=H)
    assert ok.status_code == 200 and ok.json()["pending"] >= 1
    bad = c.post("/v1/vega/feedback", json={"id": rid, "answers": {"team": "nobody"}}, headers=H)
    assert bad.status_code == 400


def test_a_restart_resumes_every_decision(tmp_path):
    llm = LLM()
    app = app_with(tmp_path, llm, mode="shadow")
    c = TestClient(app)
    run_stream(c, 60, 9)
    flush(app)
    for d in app.state.desks.values():
        d.save()
    before = c.get("/v1/vega/stats", headers=H).json()["decisions"][0]
    app2 = app_with(tmp_path, llm)
    after = TestClient(app2).get("/v1/vega/stats", headers=H).json()["decisions"][0]
    assert after["decision"] == before["decision"] and after["requests"] == 60
    assert after["mode"] == "shadow"  # the decision keeps its own mode across restarts
    assert after["learned_version"] == before["learned_version"] >= 1


def test_the_openai_format(tmp_path):
    llm = LLM()
    app = app_with(tmp_path, llm)
    c = TestClient(app)
    h = {"x-vega-key": "key-a", "authorization": "Bearer sk-openai"}
    for _ in range(4):
        run_stream(c, 100, 11, headers=h, path="/v1/chat/completions", body=oai_body)
        flush(app)
    out = run_stream(c, 100, 12, headers=h, path="/v1/chat/completions", body=oai_body)
    assert sum(src == "vega" for src, _ in out) > 50
    assert all(s.get("authorization") == "Bearer sk-openai" for s in llm.seen)


def test_which_requests_are_decisions():
    assert questions_from_schema(SCHEMA)[1] == {"team": "str"}
    mixed = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}, "n": {"enum": [1, 2, 3]}},
        "required": ["ok", "n"],
    }
    qs, kinds = questions_from_schema(mixed, "ctx")
    assert kinds == {"ok": "bool", "n": "int"} and qs["ok"]["type"] == "boolean"
    assert qs["n"]["instructions"].startswith("ctx")
    free = {"type": "object", "properties": {"why": {"type": "string"}}, "required": ["why"]}
    assert questions_from_schema(free) is None  # free text is the LLM's job
    optional = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": []}
    assert questions_from_schema(optional) is None
    assert anthropic_spec(ant_body("hello"))["text"] == "hello"
    assert anthropic_spec(dict(ant_body("x"), stream=True)) is None
    img = ant_body("x")
    img["messages"][0]["content"] = [{"type": "image", "source": {}}]
    assert anthropic_spec(img) is None
    prefill = ant_body("x")
    prefill["messages"].append({"role": "assistant", "content": "{"})
    assert anthropic_spec(prefill) is None
    assert openai_spec(oai_body("hi"))["system"] == SYSTEM
    assert openai_spec(dict(oai_body("hi"), n=3)) is None


def test_cost():
    assert cost_of("claude-haiku-5-5", {"input_tokens": 1_000_000, "output_tokens": 0}) == pytest.approx(0.10)
    assert cost_of("claude-opus-5-5-20261101", {"input_tokens": 0, "output_tokens": 1_000_000}) == 20.0
    assert cost_of("unknown-model", {"input_tokens": 5}) is None


def test_trust_per_field_by_header_and_endpoint(tmp_path):
    app = app_with(tmp_path, LLM())
    c = TestClient(app)
    r = c.post(
        "/v1/messages", json=ant_body("refund invoice"), headers=dict(H, **{"x-vega-trust": "team=0.8"})
    )
    assert r.status_code == 200
    rep = c.get("/v1/vega/stats", headers=H).json()["decisions"][0]
    assert rep["questions"]["team"]["trust"] == 0.8
    for bad in ("team=1.5", "nobody=0.9", "team=high"):
        h = dict(H, **{"x-vega-trust": bad})
        assert c.post("/v1/messages", json=ant_body("x"), headers=h).status_code == 400
    ok = c.post("/v1/vega/trust", json={"decision": rep["decision"], "trust": {"team": 0.9}}, headers=H)
    assert ok.status_code == 200 and ok.json()["trust"] == {"team": 0.9}
    for d in app.state.desks.values():
        d.save()
    again = TestClient(app_with(tmp_path, LLM())).get("/v1/vega/stats", headers=H).json()["decisions"][0]
    assert again["questions"]["team"]["trust"] == 0.9  # kept across a restart


def test_keys_from_the_admin_api_usage_logs_and_the_admin_token(tmp_path):
    llm = LLM()
    app = app_with(tmp_path, llm, keys={}, admin_token="adm-secret")
    c = TestClient(app)
    A = {"authorization": "Bearer adm-secret"}
    # the admin routes need the token; without one configured they do not exist
    assert c.post("/v1/vega/admin/keys", json={"owner": "user_1"}).status_code == 401
    assert c.get("/v1/vega/admin/usage", headers={"authorization": "Bearer wrong"}).status_code == 401
    assert c.post("/v1/vega/admin/keys", json={"owner": "bad owner!"}, headers=A).status_code == 400
    made = c.post("/v1/vega/admin/keys", json={"owner": "user_1", "name": "prod"}, headers=A).json()
    key = made["key"]
    assert key.startswith("vg_") and made["owner"] == "user_1" and made["prefix"] == key[:10]
    # once a key exists, a request without one is refused (no more anonymous "default" tenant)
    assert c.post("/v1/messages", json=ant_body("refund my invoice")).status_code == 401
    h = {"x-vega-key": key, "x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"}
    run_stream(c, 160, 3, headers=h)
    plain = dict(ant_body("hello"))
    plain.pop("output_config")
    assert c.post("/v1/messages", json=plain, headers=h).status_code == 200  # a passthrough
    use = c.get("/v1/vega/admin/usage", params={"owner": "user_1"}, headers=A).json()
    t = use["totals"]
    assert t["requests"] == 161 and t["passthrough"] == 1 and t["vega"] + t["llm"] == 160
    assert t["vega"] > 0 and t["cost_usd"] > 0 and t["saved_usd"] > 0
    assert use["decisions"] and use["daily"] and use["models"][0]["model"] == "claude-haiku-5-5"
    # the key holder sees the same numbers for themselves, and nobody else's
    mine = c.get("/v1/vega/usage", headers=h).json()
    assert mine["totals"]["requests"] == 161
    logs = c.get("/v1/vega/admin/logs", params={"owner": "user_1", "limit": 5}, headers=A).json()["events"]
    assert len(logs) == 5 and logs[0]["id"] > logs[-1]["id"] and logs[0]["tenant"] == "user_1"
    assert {e["source"] for e in c.get("/v1/vega/logs", headers=h).json()["events"]} <= {
        "vega",
        "llm",
        "passthrough",
    }
    allu = c.get("/v1/vega/admin/usage", headers=A).json()
    assert [u["tenant"] for u in allu["users"]] == ["user_1"]
    assert c.get("/v1/vega/admin/decisions", params={"owner": "user_1"}, headers=A).json()["decisions"]
    # no message content is ever stored
    row = logs[0]
    assert not any(isinstance(v, str) and "invoice" in v for v in row.values())
    # a revoked key stops working at once
    assert c.post("/v1/vega/admin/keys/%s/revoke" % made["id"], headers=A).json()["ok"]
    assert c.post("/v1/messages", json=ant_body("refund"), headers=h).status_code == 401
    keys = c.get("/v1/vega/admin/keys", params={"owner": "user_1"}, headers=A).json()["keys"]
    assert keys[0]["revoked"] is not None and "hash" not in keys[0]


def test_without_an_admin_token_the_admin_routes_do_not_exist(tmp_path, monkeypatch):
    monkeypatch.delenv("VEGA_ADMIN_TOKEN", raising=False)
    c = TestClient(app_with(tmp_path, LLM()))
    assert c.get("/v1/vega/admin/usage", headers={"authorization": "Bearer x"}).status_code == 404


def test_a_limit_stops_vega_answering_and_never_refuses(tmp_path):
    llm = LLM()
    app = app_with(tmp_path, llm, keys={}, admin_token="adm-secret")
    c = TestClient(app)
    A = {"authorization": "Bearer adm-secret"}
    key = c.post("/v1/vega/admin/keys", json={"owner": "user_1"}, headers=A).json()["key"]
    h = {"x-vega-key": key, "x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"}
    for seed in range(4):  # the LLM teaches until vega answers most of them
        run_stream(c, 100, seed, headers=h)
        flush(app)
    assert c.get("/v1/vega/usage", headers=h).json()["limit"] is None  # no limit: nothing to report
    before = c.get("/v1/vega/admin/usage", params={"owner": "user_1"}, headers=A).json()["totals"]["vega"]
    assert before > 50
    # bad requests are refused, a good one sets the cap three answers above what is already used
    assert c.post("/v1/vega/admin/limits", json={"owner": "user_1"}, headers=A).status_code == 400
    assert (
        c.post("/v1/vega/admin/limits", json={"owner": "u u", "vega_limit": 1}, headers=A).status_code == 400
    )
    assert (
        c.post("/v1/vega/admin/limits", json={"owner": "user_1", "vega_limit": -1}, headers=A).status_code
        == 400
    )
    assert c.post("/v1/vega/admin/limits", json={"owner": "user_1", "vega_limit": 5}).status_code == 401
    got = c.post(
        "/v1/vega/admin/limits", json={"owner": "user_1", "vega_limit": before + 3}, headers=A
    ).json()
    assert got["limit"]["used"] == before and got["limit"]["left"] == 3 and not got["limit"]["reached"]
    calls = llm.calls
    out = run_stream(c, 60, 9, headers=h)  # every request still gets a right answer
    assert sum(src == "vega" for src, _ in out) == 3  # exactly the three that were left
    assert llm.calls - calls == 57 and all(ok for src, ok in out if src == "llm")
    r = c.post("/v1/messages", json=ant_body("refund my invoice"), headers=h)
    assert r.status_code == 200 and r.headers["x-vega-source"] == "llm"
    assert r.headers["x-vega-limit"] == "reached" and r.headers["x-vega-escalated"] == "limit"
    assert r.headers["x-vega-mode"] == "live"  # the mode the caller is in, not how the cap was applied
    lim = c.get("/v1/vega/usage", headers=h).json()["limit"]
    assert lim["reached"] and lim["used"] == before + 3 and lim["left"] == 0 and lim["resets"] > lim["since"]
    assert c.get("/v1/vega/admin/limits", headers=A).json()["limits"][0]["owner"] == "user_1"
    act = c.get("/v1/vega/admin/activity", params={"days": 30}, headers=A).json()
    assert (
        act["tenants"][0]["tenant"] == "user_1"
        and act["tenants"][0]["keys"] == 1
        and act["rows"][0]["vega"] > 0
    )
    assert c.get("/v1/vega/admin/activity").status_code == 401
    logs = c.get("/v1/vega/admin/logs", params={"owner": "user_1", "limit": 1}, headers=A).json()["events"]
    assert logs[0]["escalated"] == "limit" and logs[0]["source"] == "llm"
    # vega mode never calls the LLM, but past the cap it has to
    r = c.post("/v1/messages", json=ant_body("refund my invoice"), headers=h | {"x-vega-mode": "vega"})
    assert r.headers["x-vega-source"] == "llm" and r.headers["x-vega-limit"] == "reached"
    # another owner is not capped by this one's limit
    other = c.post("/v1/vega/admin/keys", json={"owner": "user_2"}, headers=A).json()["key"]
    assert (
        "x-vega-limit"
        not in c.post("/v1/messages", json=ant_body("refund"), headers=h | {"x-vega-key": other}).headers
    )
    # lifting the cap: vega answers again at once, with what it learned while capped
    assert (
        c.post("/v1/vega/admin/limits", json={"owner": "user_1", "vega_limit": None}, headers=A).json()[
            "limit"
        ]
        is None
    )
    assert sum(src == "vega" for src, _ in run_stream(c, 60, 11, headers=h)) > 30
    # the dashboard acts for a user without their key; a wrong token or a missing user is refused, and neither
    # header (nor the admin token) ever reaches the LLM
    as_user = {
        "x-vega-admin": "adm-secret",
        "x-vega-as": "user_1",
        "x-api-key": "sk-ant-test",
        "anthropic-version": "2023-06-01",
    }
    r = c.post(
        "/v1/messages", json=ant_body("refund my invoice"), headers=as_user | {"x-vega-mode": "shadow"}
    )
    assert r.status_code == 200 and r.headers["x-vega-source"] == "llm"
    assert "x-vega-admin" not in llm.seen[-1] and "x-vega-as" not in llm.seen[-1]
    assert "adm-secret" not in json.dumps(llm.seen[-1])
    assert (
        c.post("/v1/messages", json=ant_body("x"), headers=as_user | {"x-vega-admin": "wrong"}).status_code
        == 401
    )
    assert (
        c.post("/v1/messages", json=ant_body("x"), headers={"x-vega-admin": "adm-secret"}).status_code == 401
    )
    assert c.post("/v1/messages", json=ant_body("x"), headers={"x-vega-as": "user_1"}).status_code == 401
    assert c.get("/v1/vega/usage", headers=as_user).json()["owner"] == "user_1"


def test_ledger_limits_and_exact_windows(tmp_path):
    import time

    from vegaml.learn.ledger import Ledger, month_start, next_month_start

    led = Ledger(str(tmp_path / "l.db"))
    now = time.time()
    assert month_start(now) <= now < next_month_start(now)
    assert time.gmtime(month_start(now)).tm_mday == 1 and time.gmtime(next_month_start(now)).tm_mday == 1
    assert next_month_start(1798761599) == 1798761600  # 31 Dec 2026 23:59:59 UTC -> 1 Jan 2027
    assert not led.over_limit("a") and led.get_limit("a") is None and led.limits() == []
    last_month = month_start(now) - 5 * 86400
    for ts, src in (
        (last_month, "vega"),
        (last_month, "vega"),
        (now - 30, "vega"),
        (now - 20, "llm"),
        (now - 10, "vega"),
    ):
        led.log(tenant="a", source=src, ts=ts)
    led.set_limit("a", 3)
    assert led.get_limit("a")["used"] == 2 and not led.over_limit("a")  # last month does not count
    led.log(tenant="a", source="vega")
    led.log(tenant="b", source="vega")  # someone else's answers do not count either
    assert led.over_limit("a") and led.get_limit("a")["left"] == 0 and not led.over_limit("b")
    led.set_limit("a", 10, since=last_month - 1)  # counted from a fixed moment: everything since then
    lim = led.get_limit("a")
    assert lim["used"] == 5 and lim["resets"] is None and not lim["reached"]
    led.set_limit("a", 0)
    assert led.over_limit("a")
    # a limit survives a restart
    again = Ledger(str(tmp_path / "l.db"))
    assert again.over_limit("a") and again.get_limit("a")["used"] == 3
    led.set_limit("a", None)
    assert not led.over_limit("a") and led.limits() == []
    # usage for an exact window: a billing period
    u = led.usage(owner="a", since=month_start(now), until=now - 15)
    assert u["totals"]["vega"] == 1 and u["totals"]["llm"] == 1 and u["since"] == month_start(now)
    assert led.usage(owner="a", since=last_month - 1, until=month_start(now))["totals"]["vega"] == 2
    assert led.usage(owner="a", days=30)["until"] is None
    assert led.vega_count("a", month_start(now)) == 3 and led.vega_count("a", 0, until=month_start(now)) == 2
    # activity: a row per tenant per day, a line per tenant, and the people who made a key and never used it
    led.create_key("c", "unused")
    led.create_key("a", "prod")
    led.log(
        tenant="a", source="vega", decision="d1", model="m", read="text,image", mode="shadow", saved_usd=0.5
    )
    act = led.activity(days=90)
    by = {t["tenant"]: t for t in act["tenants"]}
    assert (
        set(by) == {"a", "b", "c"}
        and by["c"]["requests"] == 0
        and by["c"]["keys"] == 1
        and by["c"]["first_seen"] is None
    )
    assert (
        by["a"]["requests"] == 7
        and by["a"]["vega"] == 6
        and by["a"]["active_days"] == 2
        and by["a"]["keys"] == 1
    )
    assert by["a"]["first_vega"] == last_month and by["a"]["images"] == 1 and by["a"]["shadow"] == 1
    assert by["a"]["first_seen"] == last_month and by["a"]["saved_usd"] == 0.5 and by["b"]["keys"] == 0
    days_a = [r for r in act["rows"] if r["tenant"] == "a"]
    assert (
        len(days_a) == 2 and sum(r["requests"] for r in days_a) == 7 and days_a[0]["day"] < days_a[1]["day"]
    )
    assert led.activity(days=1)["rows"][0]["day"] == days_a[1]["day"]  # a short window drops last month's row


def test_two_proxies_share_one_ledger(tmp_path):
    import time

    from vegaml.learn.ledger import Ledger

    path = str(tmp_path / "shared.db")
    a, b = Ledger(path, shared=0.05), Ledger(path, shared=0.05)
    key, row = a.create_key("user_1", "prod")
    assert b.tenant_for(key) == (row["id"], "user_1")  # a key made through one works on the other at once
    a.set_limit("user_1", 3)
    time.sleep(0.08)
    assert b.get_limit("user_1")["vega_limit"] == 3 and not b.over_limit("user_1")
    a.log(tenant="user_1", source="vega")
    b.log(tenant="user_1", source="vega")
    a.log(tenant="user_1", source="vega")
    time.sleep(0.08)  # each sees the other's answers once what it remembered is old
    assert a.over_limit("user_1") and b.over_limit("user_1") and b.get_limit("user_1")["used"] == 3
    assert (
        a.usage(owner="user_1", days=1)["totals"]["vega"]
        == 3
        == b.usage(owner="user_1", days=1)["totals"]["vega"]
    )
    assert a.revoke_key(row["id"])
    assert a.tenant_for(key) is None
    time.sleep(0.08)
    assert b.tenant_for(key) is None  # revoked through one: refused by the other
    # a ledger that is not shared never re-reads (one proxy: nothing else can change it)
    solo = Ledger(str(tmp_path / "solo.db"))
    k2, _ = solo.create_key("u", "x")
    assert solo.tenant_for(k2) and solo.shared == 0.0


def test_the_proxy_says_which_model_answered(tmp_path):
    app = app_with(tmp_path, LLM(), model_name="4b")
    c = TestClient(app)
    assert c.get("/health").json()["model"] == "4b"
    r = c.post("/v1/messages", json=ant_body("refund my invoice"), headers=H)
    assert r.headers["x-vega-model"] == "4b"


def test_a_removed_state_folder_does_not_fail_requests(tmp_path):
    import shutil

    app = app_with(tmp_path, LLM())
    c = TestClient(app)
    run_stream(c, 12, 1)
    shutil.rmtree(tmp_path / "state" / "acme")  # someone tidied the disk under a running proxy
    assert all(src in ("llm", "vega") for src, _ in run_stream(c, 30, 2))  # saves every 10 requests: none fails
