"""vega Proxy: vega in front of an LLM API. An app changes its base URL and nothing else.

    python -m vegaml.learn proxy --host 0.0.0.0 --port 8890 --state ./vega-proxy

    client = anthropic.Anthropic(base_url="http://localhost:8890", default_headers={"x-vega-key": KEY})
    client = openai.OpenAI(base_url="http://localhost:8890/v1", default_headers={"x-vega-key": KEY})

A request that asks for a fixed set of answers (structured output whose fields are all enums or
booleans) is a decision. vega answers it when it is sure; otherwise the request goes to the LLM exactly
as the app sent it, the LLM's response is returned unchanged, and vega learns from the answer. Every
other request (chat, writing, tools, streaming) is passed through untouched.

The app's own LLM key travels with each request and is forwarded, never stored. Each decision (an app's
system prompt plus its schema, or a name in the x-vega-decision header) gets its own learner.

Modes, per request (x-vega-mode header) or per decision (POST /v1/vega/mode):
    live    vega answers what it is sure of, the LLM the rest (the default)
    shadow  the LLM answers everything; vega answers alongside, learns, and is measured
    vega    vega answers everything and the LLM is never called
    off     plain passthrough

Limits: POST /v1/vega/admin/limits {"owner": id, "vega_limit": n} caps how many decisions vega answers for an
owner per calendar month. Past the cap nothing is refused: the LLM answers every decision (x-vega-limit:
reached) and vega keeps reading alongside, so what it learned is ready when the cap lifts.

How sure vega must be, per field: x-vega-trust: topic=0.85,route=0.9 (or POST /v1/vega/trust) sets the
share of vega's answers that must agree with the LLM before vega answers that field alone; fields left
out keep the server default (0.95). One unsure field still sends the whole request to the LLM, so give a
low-stakes field a lower bar and keep the strict one where a mistake costs. Saved per decision.

Response headers: x-vega-source (vega | llm | passthrough), x-vega-decision, x-vega-id (for
POST /v1/vega/feedback), x-vega-confidence, x-vega-escalated.
"""

import base64
import collections
import contextlib
import contextvars
import hashlib
import hmac
import json
import os
import re
import threading
import time
import uuid

from .reader import label_of
from .records import CONTEXTS, DEFAULT_BUDGET, as_record, decode_data_url, window

#: list prices, USD per million tokens (input, output); extend with VEGA_PROXY_PRICES='{"model": [in, out]}'
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
}
MODES = ("live", "shadow", "vega", "off")
#: a conversation longer than this is the LLM's alone; below it, vega reads it within its context window
MAX_CHARS = 400000
_CALL = contextvars.ContextVar("vega_proxy_call", default=None)
_DROP = {
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "accept-encoding",
    "content-encoding",
}
_NAME = re.compile(r"[A-Za-z0-9_.\-]{1,64}")


# ---------------------------------------------------------------------- what counts as a decision
def questions_from_schema(schema, system=""):
    """vegaml questions for a JSON schema whose fields are all required enums or booleans, with the
    kind of each field (str, int or bool) so vega's answer is written back the way the schema wants it.
    None for any other schema: that request is not a decision and goes to the LLM untouched."""
    if not isinstance(schema, dict) or schema.get("type") != "object":
        return None
    props = schema.get("properties")
    if not isinstance(props, dict) or not 1 <= len(props) <= 16:
        return None
    if set(props) - set(schema.get("required") or []):
        return None
    context = (system or "").strip()[:2000]
    qs, kinds = {}, {}
    for name, p in props.items():
        if not isinstance(p, dict) or not _NAME.fullmatch(name):
            return None
        ask = str(p.get("description") or "").strip() or "Which %s?" % name.replace("_", " ")
        instructions = context + "\n\n" + ask if context else ask
        if "enum" in p:
            vals = p["enum"]
            if not isinstance(vals, list) or not 2 <= len(vals) <= 500:
                return None
            if all(isinstance(v, str) for v in vals):
                kinds[name] = "str"
            elif all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
                kinds[name] = "int"
            else:
                return None
            ids = [str(v) for v in vals]
            if len(set(ids)) != len(ids) or any(not i.strip() for i in ids):
                return None
            qs[name] = {"type": "choice", "instructions": instructions, "criteria": {i: None for i in ids}}
        elif p.get("type") == "boolean":
            kinds[name] = "bool"
            qs[name] = {
                "type": "boolean",
                "instructions": instructions,
                "criteria": {"false": "no", "true": "yes"},
            }
        else:
            return None
    return qs, kinds


def _text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for b in content:
            if not isinstance(b, dict) or b.get("type") != "text":
                return None
            out.append(str(b.get("text", "")))
        return "\n".join(out)
    return None


def _anthropic_turn(role, content):
    """One Anthropic message as a record turn: text, and pictures sent inline (base64). A picture by URL,
    a document or a tool block is not read here: None, and the request goes to the LLM untouched."""
    if isinstance(content, str):
        return {"role": role, "text": content}
    if not isinstance(content, list):
        return None
    texts, images = [], []
    for b in content:
        if not isinstance(b, dict):
            return None
        t = b.get("type")
        if t == "text":
            texts.append(str(b.get("text", "")))
        elif t == "image" and (b.get("source") or {}).get("type") == "base64":
            images.append(base64.b64decode(b["source"].get("data") or ""))
        else:
            return None
    return {"role": role, "text": "\n".join(texts), "images": images}


def _openai_turn(role, content):
    """One OpenAI message as a record turn: text, pictures as data URLs, input_audio, and video_url data
    URLs (the form vLLM and other OpenAI-compatible servers take for clips). URLs to fetch are not read."""
    if isinstance(content, str):
        return {"role": role, "text": content}
    if not isinstance(content, list):
        return None
    turn = {"role": role, "text": "", "images": []}
    texts = []
    for p in content:
        if not isinstance(p, dict):
            return None
        t = p.get("type")
        if t == "text":
            texts.append(str(p.get("text", "")))
        elif t == "image_url":
            raw = decode_data_url((p.get("image_url") or {}).get("url"))
            if raw is None:
                return None
            turn["images"].append(raw)
        elif t == "input_audio" and "audio" not in turn:
            data = (p.get("input_audio") or {}).get("data")
            if not data:
                return None
            turn["audio"] = base64.b64decode(data)
        elif t == "video_url" and "video" not in turn:
            raw = decode_data_url((p.get("video_url") or {}).get("url"))
            if raw is None:
                return None
            turn["video"] = raw
        else:
            return None
    turn["text"] = "\n".join(texts)
    return turn


def _spec(questions_kinds, system, turns, model):
    if questions_kinds is None or not turns or any(t is None for t in turns):
        return None
    if sum(len(t.get("text") or "") for t in turns) > MAX_CHARS:
        return None  # beyond what any context window here would read: the LLM's job
    media = any(t.get("images") or t.get("audio") or t.get("video") for t in turns)
    if not media and not any((t.get("text") or "").strip() for t in turns):
        return None
    qs, kinds = questions_kinds
    text = (
        None
        if media
        else window(
            [(t["role"], t["text"]) for t in turns] if len(turns) > 1 else [(None, turns[0]["text"])],
            "full",
            MAX_CHARS,
        )
    )
    return {
        "questions": qs,
        "kinds": kinds,
        "system": system,
        "turns": turns,
        "media": media,
        "text": text,
        "model": str(model or ""),
    }


def anthropic_spec(body):
    """The decision in an Anthropic Messages request, or None when it is not one."""
    if not isinstance(body, dict) or body.get("stream") or body.get("tools"):
        return None
    fmt = (body.get("output_config") or {}).get("format") or {}
    if not isinstance(fmt, dict) or fmt.get("type") != "json_schema":
        return None
    system = body.get("system") or ""
    system = _text_of(system) if isinstance(system, list) else str(system)
    msgs = body.get("messages")
    if system is None or not isinstance(msgs, list) or not msgs or not isinstance(msgs[-1], dict):
        return None
    if msgs[-1].get("role") != "user":
        return None  # an assistant prefill continues the LLM's own text
    if not all(isinstance(m, dict) for m in msgs):
        return None
    turns = [_anthropic_turn(m.get("role"), m.get("content")) for m in msgs]
    return _spec(questions_from_schema(fmt.get("schema"), system), system, turns, body.get("model"))


def openai_spec(body):
    """The decision in an OpenAI chat completions request, or None when it is not one."""
    if not isinstance(body, dict) or body.get("stream") or body.get("tools") or body.get("functions"):
        return None
    if body.get("n") not in (None, 1):
        return None
    rf = body.get("response_format") or {}
    if not isinstance(rf, dict) or rf.get("type") != "json_schema":
        return None
    schema = (rf.get("json_schema") or {}).get("schema")
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs or not isinstance(msgs[-1], dict):
        return None
    if msgs[-1].get("role") != "user":
        return None
    system, turns = [], []
    for m in msgs:
        if not isinstance(m, dict):
            return None
        if m.get("role") in ("system", "developer"):
            t = _text_of(m.get("content"))
            if t is None:
                return None
            system.append(t)
        else:
            turns.append(_openai_turn(m.get("role"), m.get("content")))
    system = "\n\n".join(system)
    return _spec(questions_from_schema(schema, system), system, turns, body.get("model"))


def anthropic_answers(content):
    """The JSON answers in an Anthropic response body; None for a refusal or anything unreadable."""
    try:
        r = json.loads(content)
    except ValueError:
        return None, {}
    usage = r.get("usage") or {}
    if r.get("stop_reason") == "refusal":
        return None, usage
    text = next((b.get("text") for b in r.get("content") or [] if b.get("type") == "text"), None)
    try:
        got = json.loads(text) if text else None
    except ValueError:
        got = None
    return (got if isinstance(got, dict) else None), usage


def openai_answers(content):
    try:
        r = json.loads(content)
        msg = r["choices"][0]["message"]
    except (ValueError, KeyError, IndexError, TypeError):
        return None, {}
    usage = r.get("usage") or {}
    if msg.get("refusal"):
        return None, usage
    try:
        got = json.loads(msg.get("content") or "")
    except ValueError:
        got = None
    return (got if isinstance(got, dict) else None), usage


def price_of(model, prices=None):
    prices = prices or PRICES
    if model in prices:
        return prices[model]
    for k in sorted(prices, key=len, reverse=True):  # a dated id, e.g. claude-haiku-5-5-20261001
        if model.startswith(k):
            return prices[k]
    return None


def cost_of(model, usage, prices=None):
    """USD for one LLM call, from its usage block (Anthropic or OpenAI), or None for an unpriced model."""
    p = price_of(model, prices)
    if p is None:
        return None
    if "input_tokens" in usage:
        tin = (
            int(usage.get("input_tokens") or 0)
            + 1.25 * int(usage.get("cache_creation_input_tokens") or 0)
            + 0.1 * int(usage.get("cache_read_input_tokens") or 0)
        )
        tout = int(usage.get("output_tokens") or 0)
    else:
        tin, tout = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
    return (tin * p[0] + tout * p[1]) / 1e6


def _teacher(text, questions):
    """The Escalator's teacher: the app's own request, sent to its LLM as it was written."""
    ctx = _CALL.get()
    if ctx is None:
        return None
    ctx["resp"] = ctx["send"]()
    if ctx["resp"] is None or ctx["resp"][0] != 200:
        return None
    got, ctx["usage"] = ctx["parse"](ctx["resp"][2])
    if got is None:
        ctx["refused"] = True
        return None
    return {q: got[q] for q in questions if q in got}


# ---------------------------------------------------------------------- one learner per decision
class Desk:
    def __init__(
        self, key, tenant, api, spec, reader, root, every, audit, mode, prices, trust=0.95, modality="text"
    ):
        from .learner import Learner
        from .teacher import Escalator

        self.key, self.tenant, self.api = key, tenant, api
        self.questions, self.kinds = spec["questions"], spec["kinds"]
        self.order = list(self.questions)
        self.prices = prices
        self.dir = os.path.join(root, tenant, key)
        os.makedirs(self.dir, exist_ok=True)
        self.modality = modality
        self.learner = Learner(
            self.questions,
            reader=reader,
            workdir=os.path.join(self.dir, "learner"),
            every=every,
            modality=modality,
        )
        self.esc = Escalator(self.learner, _teacher, audit=audit, trust=trust)
        self.mode, self.stats, self.cost = mode, collections.Counter(), {}
        self.created = time.time()
        self._lock = threading.Lock()
        self._since_save = 0
        path = os.path.join(self.dir, "desk.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                saved = json.load(f)
            self.mode = saved.get("mode") or mode
            self.stats.update(saved.get("stats", {}))
            self.cost = {m: list(v) for m, v in saved.get("cost", {}).items()}
            self.created = saved.get("created", self.created)
            self.esc.load_state(saved.get("escalator", {}))
        else:
            with open(os.path.join(self.dir, "questions.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {"api": api, "questions": self.questions, "kinds": self.kinds, "modality": modality},
                    f,
                    indent=1,
                )

    def handle(self, text, ctx, mode):
        tok = _CALL.set(ctx)
        try:
            if mode == "vega":
                out = self.learner.decide(text)
                for a in out["answers"].values():
                    a["source"] = "vega"
            else:
                out = self.esc.decide(text, shadow=mode == "shadow")
        finally:
            _CALL.reset(tok)
        model = ctx.get("model", "")
        by_llm = ctx.get("resp") is not None
        with self._lock:
            s = self.stats
            s["requests"] += 1
            s["mode_" + mode] += 1
            if ctx.get("error"):
                s["llm_errors"] += 1
            elif by_llm:
                s["llm_calls"] += 1
                if ctx["resp"][0] != 200:
                    s["llm_status_%d" % ctx["resp"][0]] += 1
                if ctx.get("refused"):
                    s["llm_refused"] += 1
                usd = cost_of(model, ctx.get("usage") or {}, self.prices)
                if usd is not None and ctx["resp"][0] == 200:
                    c = self.cost.setdefault(model, [0, 0.0])
                    c[0] += 1
                    c[1] += usd
                    s["llm_usd_micro"] += round(usd * 1e6)
                if mode == "shadow" and all(a.get("escalated") == "shadow" for a in out["answers"].values()):
                    s["shadow_vega_would_answer"] += 1
            else:
                s["served_by_vega"] += 1
                c = self.cost.get(model)
                if c and c[0]:
                    s["saved_usd_micro"] += round(1e6 * c[1] / c[0])
            self._since_save += 1
            save = self._since_save >= 10
        if save:
            self.save()
        return out

    def written(self, out):
        """vega's answers as the JSON object the app's schema asks for, in the schema's order."""
        obj = {}
        for q in self.order:
            lab = label_of(self.questions[q], out["answers"][q]["answer"])
            k = self.kinds[q]
            obj[q] = lab == "true" if k == "bool" else int(lab) if k == "int" else lab
        return obj

    def save(self):
        with self._lock:
            data = {
                "key": self.key,
                "api": self.api,
                "mode": self.mode,
                "created": self.created,
                "stats": dict(self.stats),
                "cost": self.cost,
                "escalator": self.esc.state(),
            }
            self._since_save = 0
        tmp = os.path.join(self.dir, "desk.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, os.path.join(self.dir, "desk.json"))

    def report(self):
        with self._lock:
            s = dict(self.stats)
        rep = self.esc.report()
        st = self.learner.status()
        n = max(s.get("requests", 0), 1)
        return {
            "decision": self.key,
            "api": self.api,
            "modality": self.modality,
            "mode": self.mode,
            "questions": {
                q: {
                    "options": len(self.questions[q].get("criteria") or {}),
                    "readout": st["questions"][q]["readout"],
                    "cross_validated_accuracy": st["questions"][q]["cross_validated_accuracy"],
                    "line": round(rep["line"][q], 4),
                    "trust": rep["trust"][q],
                    "agreement_with_llm": rep["audit_agreement"][q],
                    "checked": rep["audits"][q],
                }
                for q in self.order
            },
            "requests": s.get("requests", 0),
            "served_by_vega": s.get("served_by_vega", 0),
            "llm_calls": s.get("llm_calls", 0),
            "vega_share": s.get("served_by_vega", 0) / n,
            "llm_share_last_100": rep["teacher_share_recent"],
            "shadow_vega_would_answer": s.get("shadow_vega_would_answer", 0),
            "llm_errors": s.get("llm_errors", 0),
            "llm_refused": s.get("llm_refused", 0),
            "llm_usd": s.get("llm_usd_micro", 0) / 1e6,
            "saved_usd": s.get("saved_usd_micro", 0) / 1e6,
            "learned_version": st["version"],
            "learner_error": st["last_error"],
        }


# ---------------------------------------------------------------------- the server
def create_app(
    reader,
    state="./vega-proxy",
    keys=None,
    mode="live",
    every=25,
    audit=0.05,
    anthropic_url="https://api.anthropic.com",
    openai_url="https://api.openai.com",
    max_desks=200,
    timeout=600.0,
    transport=None,
    prices=None,
    trust=0.95,
    context="full",
    budget=DEFAULT_BUDGET,
    ledger=None,
    admin_token=None,
    model_name=None,
    shared=0.0,
):
    """keys: {vega key: tenant}; None or {} (and no key in the ledger) lets anyone in as tenant "default"
    (local use only). transport: an httpx transport for the upstream calls (tests).
    ledger: a vegaml.learn.ledger.Ledger (or a path) recording every request and holding keys made through
    the admin API; default <state>/ledger.db. admin_token: the bearer token for /v1/vega/admin/* (default
    VEGA_ADMIN_TOKEN; unset, those routes answer 404)."""
    try:
        import httpx
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse, Response, StreamingResponse
        from starlette.background import BackgroundTask
        from starlette.concurrency import run_in_threadpool
    except ImportError as e:
        raise SystemExit("the proxy needs its extras: pip install 'vegaml[proxy]' (%s)" % e) from e
    if mode not in MODES:
        raise ValueError("mode is one of %s" % (MODES,))
    from .ledger import Ledger

    os.makedirs(state, exist_ok=True)
    if ledger is None or isinstance(ledger, str):
        ledger = Ledger(ledger or os.path.join(state, "ledger.db"), shared=shared)
    admin_token = admin_token if admin_token is not None else os.environ.get("VEGA_ADMIN_TOKEN", "")
    prices = dict(PRICES, **(prices or {}))
    hashed = {hashlib.sha256(k.encode()).hexdigest(): t for k, t in (keys or {}).items()}
    upstream = {"anthropic": anthropic_url.rstrip("/"), "openai": openai_url.rstrip("/")}
    sync = httpx.Client(timeout=timeout, transport=transport)
    async_ = httpx.AsyncClient(timeout=timeout, transport=transport)
    desks, desks_lock = {}, threading.Lock()
    seen = collections.OrderedDict()  # x-vega-id -> (tenant, desk key, text), for feedback
    seen_media = collections.OrderedDict()
    seen_lock = threading.Lock()

    @contextlib.asynccontextmanager
    async def lifespan(app):
        yield
        for d in list(desks.values()):  # learned state is on disk before the process goes
            d.save()
            d.learner.close()

    app = FastAPI(title="vega Proxy", docs_url=None, redoc_url=None, lifespan=lifespan)

    def tenant_of(request):
        """The tenant a request belongs to: a key from VEGA_PROXY_KEYS, or one made through the admin API.
        A dashboard's server can also act for one of its users without holding their key: x-vega-admin (the
        admin token) with x-vega-as (the user). Both are x-vega-* headers, so neither is ever sent on."""
        boss, who = request.headers.get("x-vega-admin") or "", request.headers.get("x-vega-as") or ""
        if boss or who:
            if not (admin_token and boss and hmac.compare_digest(boss, admin_token) and _NAME.fullmatch(who)):
                raise HTTPException(401, "x-vega-admin and x-vega-as need the admin token and a user id")
            request.state.key_id = None
            return who
        k = request.headers.get("x-vega-key") or ""
        if k:
            t = hashed.get(hashlib.sha256(k.encode()).hexdigest())
            if t is not None:
                request.state.key_id = None
                return t
            got = ledger.tenant_for(k)
            if got is not None:
                request.state.key_id = got[0]
                return got[1]
        if not hashed and not ledger.has_keys():
            request.state.key_id = None
            return "default"
        raise HTTPException(401, "send your vega key in the x-vega-key header")

    def usage_of(api, content):
        """(input tokens, output tokens, usage block) from a non-streamed upstream reply, or Nones."""
        try:
            u = json.loads(content).get("usage") or {}
        except (ValueError, AttributeError, TypeError):
            return None, None, {}
        if "input_tokens" in u:
            return u.get("input_tokens"), u.get("output_tokens"), u
        return u.get("prompt_tokens"), u.get("completion_tokens"), u

    def note(request, tenant, **e):
        """One ledger row; never lets a logging failure reach the caller."""
        try:
            ledger.log(tenant=tenant, key_id=getattr(request.state, "key_id", None), **e)
        except Exception:  # a full disk must not take requests down
            pass

    def forward_headers(request):
        return {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in _DROP and not k.lower().startswith("x-vega-")
        } | {"accept-encoding": "identity"}

    def back_headers(headers):
        return {k: v for k, v in headers.items() if k.lower() not in _DROP}

    def which(request):
        h = request.headers
        return "anthropic" if ("x-api-key" in h or "anthropic-version" in h) else "openai"

    async def passthrough(request, api, path, raw, stream=False):
        url = upstream[api] + path
        if request.url.query:
            url += "?" + request.url.query
        req = async_.build_request(request.method, url, headers=forward_headers(request), content=raw)
        try:
            r = await async_.send(req, stream=stream)
        except httpx.HTTPError as e:
            return JSONResponse(
                {"error": {"type": "upstream_unreachable", "message": repr(e)}},
                502,
                headers={"x-vega-source": "passthrough"},
            )
        h = back_headers(r.headers) | {"x-vega-source": "passthrough"}
        if stream:
            return StreamingResponse(
                r.aiter_raw(), status_code=r.status_code, headers=h, background=BackgroundTask(r.aclose)
            )
        return Response(r.content, r.status_code, headers=h)

    def parse_trust(text):
        out = {}
        for part in filter(None, (p.strip() for p in text.split(","))):
            q, _, v = part.partition("=")
            try:
                out[q.strip()] = float(v)
            except ValueError:
                raise HTTPException(400, "x-vega-trust is field=number,field=number") from None
        return out

    def apply_trust(desk, bars):
        changed = {q: v for q, v in bars.items() if desk.esc.trust.get(q) != v}
        if not changed:
            return
        try:
            desk.esc.set_trust(changed)
        except (KeyError, ValueError) as e:
            raise HTTPException(400, "trust: %s" % e) from None
        desk.save()

    def desk_for(tenant, api, spec, name, kind="text", how=("full", None)):
        # one learner per app decision, per modality (pictures and text are not comparable) and per way of
        # reading a long conversation (features read differently are not comparable either)
        ident = [api, spec["questions"]]
        if kind != "text":
            ident.append(kind)
        if how[0] != "full" or how[1] is not None:
            ident.append(list(how))
        sig = hashlib.sha256(
            json.dumps(ident, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:12]
        if name and _NAME.fullmatch(name):
            key = "%s-%s" % (name, sig)
        else:
            key = "d-" + sig
        with desks_lock:
            d = desks.get((tenant, key))
            if d is None:
                if sum(1 for t, _ in desks if t == tenant) >= max_desks:
                    raise HTTPException(429, "too many decisions for one key (%d)" % max_desks)
                d = desks[(tenant, key)] = Desk(
                    key, tenant, api, spec, reader, state, every, audit, mode, prices, trust, kind
                )
        return d

    def remember(tenant, key, rec):
        """Kept for feedback by x-vega-id: the last 200,000 text records and the last 2,000 pictures."""
        rid = "vg_" + uuid.uuid4().hex
        with seen_lock:
            box = seen if rec.kind == "text" else seen_media
            box[rid] = (tenant, key, rec)
            while len(box) > (200000 if box is seen else 2000):
                box.popitem(last=False)
        return rid

    def synthesized(api, model, obj):
        text = json.dumps(obj, ensure_ascii=False)
        if api == "anthropic":
            return {
                "id": "msg_vega_" + uuid.uuid4().hex[:24],
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        return {
            "id": "chatcmpl-vega-" + uuid.uuid4().hex[:24],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text, "refusal": None},
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    async def decision(request, api, path):
        tenant = tenant_of(request)
        raw = await request.body()
        try:
            body = json.loads(raw)
        except ValueError:
            body = None
        want = request.headers.get("x-vega-mode")
        if want and want not in MODES:
            raise HTTPException(400, "x-vega-mode is one of %s" % (MODES,))
        spec = None
        if want != "off" and body is not None:
            spec = anthropic_spec(body) if api == "anthropic" else openai_spec(body)
        model = (body or {}).get("model") if isinstance(body, dict) else None
        if spec is None:
            t0 = time.perf_counter()
            streamed = bool((body or {}).get("stream")) if isinstance(body, dict) else False
            resp = await passthrough(request, api, path, raw, stream=streamed)
            tin, tout, u = (None, None, {}) if streamed else usage_of(api, resp.body)
            note(request, tenant, api=api, model=model, source="passthrough", status=resp.status_code,
                 latency_ms=1000 * (time.perf_counter() - t0), in_tokens=tin, out_tokens=tout,
                 cost_usd=cost_of(model or "", u, prices) if u else None)
            return resp
        how = request.headers.get("x-vega-context") or context
        if how not in CONTEXTS:
            raise HTTPException(400, "x-vega-context is one of %s" % (CONTEXTS,))
        try:
            chars = int(request.headers.get("x-vega-context-chars") or budget)
        except ValueError:
            raise HTTPException(400, "x-vega-context-chars is a number of characters") from None
        chars = max(500, min(chars, MAX_CHARS))
        turns = [dict(t, role=None) for t in spec["turns"]] if len(spec["turns"]) == 1 else spec["turns"]
        clip = request.headers.get("x-vega-video") or "auto"
        if clip not in ("auto", "frames", "speech"):
            raise HTTPException(400, "x-vega-video is auto, frames or speech")
        try:  # transcribing, decoding a clip, opening a picture: once, here
            rec = await run_in_threadpool(
                as_record, {"turns": turns, "context": how, "budget": chars, "video_read": clip}, reader
            )
        except Exception as e:  # vega cannot read it (no speech, a codec it lacks): the LLM's alone
            t0 = time.perf_counter()
            resp = await passthrough(request, api, path, raw)
            resp.headers["x-vega-unread"] = repr(e)[:200].replace("\n", " ")
            tin, tout, u = usage_of(api, resp.body)
            note(request, tenant, api=api, model=model, source="passthrough", status=resp.status_code,
                 latency_ms=1000 * (time.perf_counter() - t0), in_tokens=tin, out_tokens=tout,
                 cost_usd=cost_of(model or "", u, prices) if u else None, escalated="unread")
            return resp
        named = (how, None if chars == budget else chars)
        desk = await run_in_threadpool(
            desk_for, tenant, api, spec, request.headers.get("x-vega-decision"), rec.kind, named
        )
        bar = request.headers.get("x-vega-trust")
        if bar:
            apply_trust(desk, parse_trust(bar))
        use = want or desk.mode
        asked = use
        # the owner's vega answers for the period are used up: the LLM answers every decision and vega only
        # reads alongside, so nothing is refused and what it has learned is kept for when the limit lifts
        capped = use in ("live", "vega") and ledger.over_limit(tenant)
        if capped:
            use = "shadow"
        if use == "off":
            t0 = time.perf_counter()
            resp = await passthrough(request, api, path, raw)
            tin, tout, u = usage_of(api, resp.body)
            note(request, tenant, api=api, model=model, decision=desk.key, mode="off", source="passthrough",
                 status=resp.status_code, latency_ms=1000 * (time.perf_counter() - t0), in_tokens=tin,
                 out_tokens=tout, cost_usd=cost_of(model or "", u, prices) if u else None)
            return resp
        headers = forward_headers(request)
        ctx = {"model": spec["model"], "parse": anthropic_answers if api == "anthropic" else openai_answers}

        def send():
            try:
                r = sync.post(upstream[api] + path, headers=headers, content=raw)
                return (r.status_code, back_headers(r.headers), r.content)
            except httpx.HTTPError as e:
                ctx["error"] = repr(e)
                return None

        ctx["send"] = send
        t0 = time.perf_counter()
        out = await run_in_threadpool(desk.handle, rec, ctx, use)
        rid = remember(tenant, desk.key, rec)
        confs = [float(a.get("confidence", 0.0)) for a in out["answers"].values()]
        why = sorted({a["escalated"] for a in out["answers"].values() if a.get("escalated")})
        vh = {
            "x-vega-decision": desk.key,
            "x-vega-id": rid,
            "x-vega-mode": asked,
            "x-vega-confidence": "%.4f" % (min(confs) if confs else 0.0),
            "x-vega-latency-ms": "%.1f" % (1000 * (time.perf_counter() - t0)),
            "x-vega-read": ",".join(sorted(set(rec.sources))) or rec.kind,
        }
        if model_name:
            vh["x-vega-model"] = model_name
        if capped:
            why = ["limit"]
            vh["x-vega-limit"] = "reached"
        if why:
            vh["x-vega-escalated"] = ",".join(why)
        ms = 1000 * (time.perf_counter() - t0)
        row = {"api": api, "model": spec["model"], "decision": desk.key, "mode": asked, "latency_ms": ms,
               "confidence": min(confs) if confs else None, "escalated": ",".join(why) or None,
               "read": vh["x-vega-read"]}
        if ctx.get("resp") is not None:
            status, h, content = ctx["resp"]
            tin, tout, u = usage_of(api, content)
            note(request, tenant, source="llm", status=status, in_tokens=tin, out_tokens=tout,
                 cost_usd=cost_of(spec["model"] or "", u, prices) if u else None, **row)
            return Response(content, status, headers=h | vh | {"x-vega-source": "llm"})
        if ctx.get("error"):
            note(request, tenant, source="error", status=502, **row)
            return JSONResponse(
                {"error": {"type": "upstream_unreachable", "message": ctx["error"]}},
                502,
                headers=vh | {"x-vega-source": "llm"},
            )
        obj = desk.written(out)
        c = desk.cost.get(spec["model"])
        note(request, tenant, source="vega", status=200, cost_usd=0.0,
             saved_usd=(c[1] / c[0]) if c and c[0] else None, **row)
        return JSONResponse(synthesized(api, spec["model"], obj), headers=vh | {"x-vega-source": "vega"})

    @app.get("/health")
    def health():
        return {"ok": True, "decisions": len(desks), "checkpoint": reader.fingerprint(), "model": model_name}

    @app.post("/v1/messages")
    async def messages(request: Request):
        return await decision(request, "anthropic", "/v1/messages")

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await decision(request, "openai", "/v1/chat/completions")

    @app.post("/v1/vega/feedback")
    async def feedback(request: Request):
        tenant = tenant_of(request)
        try:
            body = json.loads(await request.body())
            rid, answers = body["id"], body["answers"]
            assert isinstance(answers, dict) and answers
        except (ValueError, KeyError, TypeError, AssertionError):
            raise HTTPException(400, 'send {"id": <x-vega-id>, "answers": {field: answer}}') from None
        with seen_lock:
            got = seen.get(rid) or seen_media.get(rid)
        if got is None or got[0] != tenant:
            raise HTTPException(404, "no decision with that id (ids are kept for the last 200,000)")
        d = desks.get((tenant, got[1]))
        try:
            r = d.learner.feedback(got[2], answers)
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e)) from None
        return {"ok": True, "decision": d.key, **r}

    @app.get("/v1/vega/stats")
    def stats(request: Request):
        tenant = tenant_of(request)
        mine = [d for (t, _), d in list(desks.items()) if t == tenant]
        return {"decisions": [d.report() for d in mine]}

    @app.post("/v1/vega/mode")
    async def set_mode(request: Request):
        tenant = tenant_of(request)
        try:
            body = json.loads(await request.body())
            key, m = body["decision"], body["mode"]
        except (ValueError, KeyError, TypeError):
            raise HTTPException(400, 'send {"decision": <x-vega-decision>, "mode": "live"}') from None
        if m not in MODES:
            raise HTTPException(400, "mode is one of %s" % (MODES,))
        d = desks.get((tenant, key))
        if d is None:
            raise HTTPException(404, "no decision %r for this key" % key)
        d.mode = m
        d.save()
        return {"ok": True, "decision": key, "mode": m}

    @app.post("/v1/vega/trust")
    async def set_trust(request: Request):
        tenant = tenant_of(request)
        try:
            body = json.loads(await request.body())
            key, bars = body["decision"], body["trust"]
            assert isinstance(bars, dict) and bars
        except (ValueError, KeyError, TypeError, AssertionError):
            raise HTTPException(400, 'send {"decision": <x-vega-decision>, "trust": {field: 0.9}}') from None
        d = desks.get((tenant, key))
        if d is None:
            raise HTTPException(404, "no decision %r for this key" % key)
        apply_trust(d, bars)
        return {"ok": True, "decision": key, "trust": dict(d.esc.trust), "line": dict(d.esc.line)}

    @app.get("/v1/vega/usage")
    def my_usage(request: Request, days: int = 30):
        """Your own numbers: totals, a daily series, per decision and per model, and your limit if you
        have one."""
        tenant = tenant_of(request)
        return dict(ledger.usage(owner=tenant, days=days), limit=ledger.get_limit(tenant))

    @app.get("/v1/vega/logs")
    def my_logs(request: Request, limit: int = 100, before: int | None = None):
        return {"events": ledger.logs(owner=tenant_of(request), limit=limit, before=before)}

    # ---------------------------------------------------------------- admin (a dashboard's server side)
    def admin(request):
        given = (request.headers.get("authorization") or "").removeprefix("Bearer ").strip()
        if not admin_token:
            raise HTTPException(404, "not found")
        if not given or not hmac.compare_digest(given, admin_token):
            raise HTTPException(401, "admin token required")

    @app.post("/v1/vega/admin/keys")
    async def admin_create_key(request: Request):
        admin(request)
        try:
            body = json.loads(await request.body())
            owner = str(body["owner"])
        except (ValueError, KeyError, TypeError):
            raise HTTPException(400, 'send {"owner": <user id>, "name": <label>}') from None
        if not _NAME.fullmatch(owner):
            raise HTTPException(400, "owner is letters, digits, - _ . only")
        key, row = ledger.create_key(owner, body.get("name") or "")
        return {"key": key, **row}

    @app.get("/v1/vega/admin/keys")
    def admin_keys(request: Request, owner: str | None = None):
        admin(request)
        return {"keys": ledger.list_keys(owner)}

    @app.post("/v1/vega/admin/keys/{key_id}/revoke")
    def admin_revoke(key_id: str, request: Request, owner: str | None = None):
        admin(request)
        if not ledger.revoke_key(key_id, owner):
            raise HTTPException(404, "no live key with that id")
        return {"ok": True, "id": key_id}

    @app.get("/v1/vega/admin/usage")
    def admin_usage(request: Request, owner: str | None = None, days: int = 30, since: float | None = None,
                    until: float | None = None):
        """The last `days`, or exactly [since, until) in unix seconds: a billing period."""
        admin(request)
        return ledger.usage(owner=owner, days=days, since=since, until=until)

    @app.post("/v1/vega/admin/limits")
    async def admin_set_limit(request: Request):
        """{"owner": id, "vega_limit": n} caps vega's answers per calendar month (UTC); add "since" (unix
        seconds) to count from a fixed moment instead; "vega_limit": null removes the cap. Past the cap the
        LLM answers every decision: nothing is refused."""
        admin(request)
        try:
            body = json.loads(await request.body())
            owner = str(body["owner"])
            cap = body["vega_limit"]
            cap = None if cap is None else int(cap)
            since = body.get("since")
            since = None if since is None else float(since)
        except (ValueError, KeyError, TypeError):
            raise HTTPException(400, 'send {"owner": <user id>, "vega_limit": <number or null>}') from None
        if not _NAME.fullmatch(owner):
            raise HTTPException(400, "owner is letters, digits, - _ . only")
        if cap is not None and cap < 0:
            raise HTTPException(400, "vega_limit is zero or more, or null for no limit")
        ledger.set_limit(owner, cap, since)
        return {"ok": True, "owner": owner, "limit": ledger.get_limit(owner)}

    @app.get("/v1/vega/admin/limits")
    def admin_limits(request: Request, owner: str | None = None):
        admin(request)
        return {"limits": [ledger.get_limit(owner)] if owner else ledger.limits()}

    @app.get("/v1/vega/admin/activity")
    def admin_activity(request: Request, days: int = 90):
        """Per tenant per day, and every tenant's history in one line: what retention, active-user and funnel
        charts are drawn from. No content, as everywhere in the ledger."""
        admin(request)
        return ledger.activity(days=days)

    @app.get("/v1/vega/admin/logs")
    def admin_logs(request: Request, owner: str | None = None, limit: int = 100, before: int | None = None,
                   decision: str | None = None, source: str | None = None):
        admin(request)
        got = ledger.logs(owner=owner, limit=limit, before=before, decision=decision, source=source)
        return {"events": got}

    @app.get("/v1/vega/admin/decisions")
    def admin_decisions(request: Request, owner: str | None = None):
        admin(request)
        mine = [(t, d) for (t, _), d in list(desks.items()) if owner is None or t == owner]
        return {"decisions": [dict(d.report(), tenant=t) for t, d in mine]}

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    async def other(path: str, request: Request):
        tenant_of(request)
        raw = await request.body()
        stream = False
        if raw:
            try:
                stream = bool(json.loads(raw).get("stream"))
            except (ValueError, AttributeError):
                pass
        return await passthrough(request, which(request), "/" + path, raw, stream=stream)

    def _resume():  # decisions this server has seen before come back as they were
        if not os.path.isdir(state):
            return
        for tenant in os.listdir(state):
            if not os.path.isdir(os.path.join(state, tenant)):  # the ledger lives beside the tenants
                continue
            for key in os.listdir(os.path.join(state, tenant)):
                p = os.path.join(state, tenant, key, "questions.json")
                if not os.path.exists(p):
                    continue
                with open(p, encoding="utf-8") as f:
                    q = json.load(f)
                spec = {"questions": q["questions"], "kinds": q["kinds"]}
                desks[(tenant, key)] = Desk(
                    key,
                    tenant,
                    q["api"],
                    spec,
                    reader,
                    state,
                    every,
                    audit,
                    mode,
                    prices,
                    trust,
                    q.get("modality", "text"),
                )

    _resume()
    app.state.desks = desks
    app.state.ledger = ledger
    return app


def load_keys(path=None):
    """{vega key: tenant} from a JSON file, or from VEGA_PROXY_KEYS as key:tenant,key:tenant."""
    if path:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    env = os.environ.get("VEGA_PROXY_KEYS", "")
    out = {}
    for part in filter(None, (p.strip() for p in env.split(","))):
        k, _, t = part.partition(":")
        if not k or not t or not _NAME.fullmatch(t):
            raise SystemExit("VEGA_PROXY_KEYS is key:tenant,key:tenant (tenant: letters, digits, - _ .)")
        out[k] = t
    return out


def run(a):
    from .reader import Reader

    if a.gpu_memory:  # a shared GPU: long reads leave cached blocks behind, which other services then lack
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(float(a.gpu_memory))

    keys = load_keys(a.keys_file)
    if not keys and not os.environ.get("VEGA_ADMIN_TOKEN") and a.host not in ("127.0.0.1", "localhost"):
        raise SystemExit("set VEGA_PROXY_KEYS, --keys-file or VEGA_ADMIN_TOKEN (keys made through the admin "
                         "API); without keys the proxy is for 127.0.0.1 only")
    prices = json.loads(os.environ.get("VEGA_PROXY_PRICES", "{}"))
    app = create_app(
        Reader.load(a.model, a.device),
        state=a.state,
        keys=keys,
        mode=a.mode,
        every=a.every,
        audit=a.audit,
        anthropic_url=a.anthropic_url,
        openai_url=a.openai_url,
        trust=a.trust,
        context=a.context,
        budget=a.context_chars,
        prices={k: tuple(v) for k, v in prices.items()},
        ledger=a.ledger,
        shared=a.ledger_shared,
        model_name=a.model or "default",
    )
    import uvicorn

    uvicorn.run(app, host=a.host, port=a.port, workers=1)
