"""A REST API around a Learner: FastAPI, every route but /healthz behind a JWT.

    python -m vegaml.learn serve --questions questions.json --examples examples.csv --workdir ./vega-state

Tokens are checked with a shared secret (HS256, env VEGAML_LEARN_JWT_SECRET) or a public key (RS256, ES256
or EdDSA,
--jwt-public-key key.pem, for tokens your own identity provider issues). A token must carry `exp`, and its
scopes in
`scope` (space separated) or `scopes` (a list):

    decide     POST /v1/decide, POST /v1/decide/batch, GET /v1/questions
    feedback   POST /v1/feedback
    admin      GET /v1/status, POST /v1/learn, GET /v1/held, POST /v1/held/{q}/accept|discard, POST /v1/undo
"""

import os
from typing import Any, Dict, List, Optional

SCOPES = ("decide", "feedback", "admin")


def _scopes(claims):
    s = claims.get("scope", claims.get("scopes", []))
    return set(s.split()) if isinstance(s, str) else set(s or [])


def create_app(
    learner,
    secret=None,
    public_key=None,
    algorithms=None,
    audience=None,
    issuer=None,
    no_auth=False,
    max_batch=64,
    max_chars=20000,
):
    try:
        import jwt
        from fastapi import Depends, FastAPI, Header, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as e:
        raise SystemExit("the REST server needs its extras: pip install 'vegaml[learn]' (%s)" % e) from e
    if not (secret or public_key or no_auth):
        raise SystemExit(
            "no way to check tokens: set VEGAML_LEARN_JWT_SECRET, pass --jwt-public-key, "
            "or --no-auth on 127.0.0.1 for local testing"
        )
    key = public_key or secret
    algorithms = algorithms or (["RS256", "ES256", "EdDSA"] if public_key else ["HS256"])

    def caller(scope):
        def check(authorization: Optional[str] = Header(default=None)) -> Dict:
            if no_auth:
                return {"sub": "local", "scope": " ".join(SCOPES)}
            if not authorization or not authorization.lower().startswith("bearer "):
                raise HTTPException(
                    401, "send Authorization: Bearer <token>", headers={"WWW-Authenticate": "Bearer"}
                )
            try:
                claims = jwt.decode(
                    authorization[7:].strip(),
                    key,
                    algorithms=algorithms,
                    audience=audience,
                    issuer=issuer,
                    leeway=30,
                    options={"require": ["exp"], "verify_aud": audience is not None},
                )
            except jwt.ExpiredSignatureError:
                raise HTTPException(
                    401, "the token has expired", headers={"WWW-Authenticate": "Bearer"}
                ) from None
            except jwt.InvalidTokenError as e:
                raise HTTPException(
                    401, "the token is not valid: %s" % e, headers={"WWW-Authenticate": "Bearer"}
                ) from e
            if scope not in _scopes(claims) and "admin" not in _scopes(claims):
                raise HTTPException(403, "this token does not carry the %r scope" % scope)
            return claims

        return check

    class DecideBody(BaseModel):
        text: str = Field(..., min_length=1, max_length=max_chars)
        questions: Optional[List[str]] = None

    class BatchBody(BaseModel):
        texts: List[str] = Field(..., min_length=1, max_length=max_batch)
        questions: Optional[List[str]] = None

    class FeedbackBody(BaseModel):
        text: str = Field(..., min_length=1, max_length=max_chars)
        answers: Dict[str, Any]

    app = FastAPI(
        title="vega learner",
        description="Decisions from a vega model that keeps learning from the answers you confirm.",
    )

    @app.get("/healthz")
    def healthz() -> Dict:
        return {"ok": True, "version": learner.version}

    @app.get("/v1/questions")
    def questions(who=Depends(caller("decide"))) -> Dict:
        return {"questions": learner.questions}

    @app.post("/v1/decide")
    def decide(body: DecideBody, who=Depends(caller("decide"))) -> Dict:
        try:
            return learner.decide(body.text, body.questions)
        except KeyError as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/v1/decide/batch")
    def decide_batch(body: BatchBody, who=Depends(caller("decide"))) -> Dict:
        if any(not t or len(t) > max_chars for t in body.texts):
            raise HTTPException(422, "every text must be 1 to %d characters" % max_chars)
        try:
            return {"results": learner.decide_many(body.texts, body.questions)}
        except KeyError as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/v1/feedback")
    def feedback(body: FeedbackBody, who=Depends(caller("feedback"))) -> Dict:
        try:
            return learner.feedback(body.text, body.answers)
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e)) from e
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e

    @app.get("/v1/status")
    def status(who=Depends(caller("admin"))) -> Dict:
        return learner.status()

    @app.post("/v1/learn")
    def learn(who=Depends(caller("admin"))) -> Dict:
        return {"learned": learner.flush(timeout=600)}

    @app.get("/v1/held")
    def held(who=Depends(caller("admin"))) -> Dict:
        return {"held": learner.held()}

    @app.post("/v1/held/{qid}/accept")
    def accept(qid: str, who=Depends(caller("admin"))) -> Dict:
        try:
            return learner.accept(qid)
        except KeyError:
            raise HTTPException(404, "nothing is held for %r" % qid) from None

    @app.post("/v1/held/{qid}/discard")
    def discard(qid: str, who=Depends(caller("admin"))) -> Dict:
        try:
            learner.discard(qid)
        except KeyError:
            raise HTTPException(404, "nothing is held for %r" % qid) from None
        return {"discarded": qid}

    @app.post("/v1/undo")
    def undo(who=Depends(caller("admin"))) -> Dict:
        try:
            return {"version": learner.undo()}
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e

    return app


def mint(secret, subject, scopes, days=30, audience=None, issuer=None):
    """A token for one caller, signed with the shared secret."""
    import datetime as dt

    import jwt

    now = dt.datetime.now(dt.timezone.utc)
    claims = {"sub": subject, "scope": " ".join(scopes), "iat": now, "exp": now + dt.timedelta(days=days)}
    if audience:
        claims["aud"] = audience
    if issuer:
        claims["iss"] = issuer
    return jwt.encode(claims, secret, algorithm="HS256")


def _rows(path, text_column, questions):
    """A CSV of labelled examples: one text column, one column per question (blank: not labelled)."""
    import csv

    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            labels = {q: r[q] for q in questions if (r.get(q) or "").strip()}
            if (r.get(text_column) or "").strip() and labels:
                out.append((r[text_column], labels))
    return out


def _learner(a, auto=True):
    import json

    from .learner import Learner

    kw = dict(model=a.model, device=a.device, workdir=a.workdir)
    if a.bundle:
        return Learner(
            bundle=a.bundle, auto=auto, **kw, **({"every": a.every, "follow": a.follow} if auto else {})
        )
    if not a.questions:
        raise SystemExit("give --questions questions.json (or --bundle)")
    with open(a.questions, encoding="utf-8") as f:
        questions = json.load(f)
    rows = _rows(a.examples, a.text_column, questions) if a.examples else []
    return Learner(
        questions, rows, auto=auto, **kw, **({"every": a.every, "follow": a.follow} if auto else {})
    )


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(prog="python -m vegaml.learn")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def source(p):
        p.add_argument("--questions", default=None, help="JSON file: the questions, as vegaml takes them")
        p.add_argument("--examples", default=None, help="CSV of labelled examples to start from (optional)")
        p.add_argument(
            "--text-column", default="text", help="the CSV column holding the text (default: text)"
        )
        p.add_argument("--bundle", default=None, help="start from a learner bundle instead (.zip or folder)")
        p.add_argument("--workdir", default="./vega-state")
        p.add_argument("--model", default=None, help="vega checkpoint (default: vegaml's)")
        p.add_argument("--device", default=None, help="cuda, cpu or mps (default: cuda when present)")

    s = sub.add_parser("serve", help="run the REST API")
    source(s)
    s.add_argument("--every", type=int, default=50)
    s.add_argument(
        "--follow", type=float, default=0, help="read-only copy: reload the writer's state every N seconds"
    )
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument(
        "--jwt-public-key", default=None, help="PEM file: verify tokens your identity provider signs"
    )
    s.add_argument("--jwt-audience", default=None)
    s.add_argument("--jwt-issuer", default=None)
    s.add_argument(
        "--no-auth", action="store_true", help="local testing only; refused unless --host is 127.0.0.1"
    )
    t = sub.add_parser("token", help="mint a token signed with VEGAML_LEARN_JWT_SECRET")
    t.add_argument("--subject", required=True, help="who the token is for, e.g. helpdesk-webhook")
    t.add_argument("--scope", nargs="+", default=["decide"], choices=SCOPES)
    t.add_argument("--days", type=int, default=30)
    t.add_argument("--jwt-audience", default=None)
    t.add_argument("--jwt-issuer", default=None)
    st = sub.add_parser("status", help="print what the learner knows, without serving")
    source(st)
    px = sub.add_parser("proxy", help="run vega in front of an LLM API (Anthropic or OpenAI format)")
    px.add_argument("--host", default="127.0.0.1")
    px.add_argument("--port", type=int, default=8890)
    px.add_argument("--state", default="./vega-proxy", help="where each decision's learner is saved")
    px.add_argument("--model", default=None, help="vega checkpoint (default: vegaml's)")
    px.add_argument(
        "--ledger",
        default=None,
        help="the keys and usage file (default: <state>/ledger.db); two proxies may share one",
    )
    px.add_argument(
        "--ledger-shared",
        type=float,
        default=0.0,
        help="seconds after which keys and limits are read again; set a few when two proxies share a ledger",
    )
    px.add_argument("--device", default=None, help="cuda, cpu or mps (default: cuda when present)")
    px.add_argument("--every", type=int, default=25, help="learn after this many LLM answers")
    px.add_argument("--audit", type=float, default=0.05, help="share of sure answers still checked")
    px.add_argument("--mode", default="live", choices=("live", "shadow", "vega", "off"))
    px.add_argument(
        "--trust", type=float, default=0.95, help="default share of answers that must agree with the LLM"
    )
    px.add_argument(
        "--context",
        default="full",
        choices=("full", "recent", "last"),
        help="what vega reads of a long conversation (x-vega-context per request)",
    )
    px.add_argument("--context-chars", type=int, default=12000, help="how much of it, in characters")
    px.add_argument(
        "--gpu-memory",
        type=float,
        default=None,
        help="cap this process at this share of the GPU (0-1); the cache is freed instead of growing past it",
    )
    px.add_argument("--anthropic-url", default="https://api.anthropic.com")
    px.add_argument("--openai-url", default="https://api.openai.com")
    px.add_argument(
        "--keys-file", default=None, help="JSON {vega key: tenant}; or set VEGA_PROXY_KEYS=key:tenant,..."
    )
    a = ap.parse_args(argv)

    if a.cmd == "proxy":
        from .proxy import run

        return run(a)

    if a.cmd == "token":
        secret = os.environ.get("VEGAML_LEARN_JWT_SECRET")
        if not secret:
            raise SystemExit("set VEGAML_LEARN_JWT_SECRET (the same secret the server uses)")
        print(mint(secret, a.subject, a.scope, a.days, a.jwt_audience, a.jwt_issuer))
        return
    if a.cmd == "status":
        import json

        print(json.dumps(_learner(a, auto=False).status(), indent=1, default=str))
        return
    if a.no_auth and a.host not in ("127.0.0.1", "localhost"):
        raise SystemExit("--no-auth is for 127.0.0.1 only; anyone who can reach the port could teach it")
    public_key = open(a.jwt_public_key, encoding="utf-8").read() if a.jwt_public_key else None
    learner = _learner(a)
    app = create_app(
        learner,
        secret=os.environ.get("VEGAML_LEARN_JWT_SECRET"),
        public_key=public_key,
        audience=a.jwt_audience,
        issuer=a.jwt_issuer,
        no_auth=a.no_auth,
    )
    import uvicorn

    uvicorn.run(app, host=a.host, port=a.port, workers=1)


if __name__ == "__main__":
    main()
