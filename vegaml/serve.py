"""vegaml-serve: the decide() call over HTTP, one process, one loaded checkpoint.

    pip install "vegaml[serve]"
    vegaml-serve --model 4b --port 8000
    vegaml-serve --calibration calibration.json --threads 1      # fitted probabilities and a band

    curl -s localhost:8000/v1/decide -H 'content-type: application/json' -d '{
        "state": "I was billed twice, refund me or I cancel.",
        "questions": {"churn": {"type": "boolean", "instructions": "Does the customer threaten to leave?"}}}'

POST /v1/decide answers one state; POST /v1/decide/batch answers a list on the same loaded
checkpoint, in order. GET /health says whether the model is up; GET /v1/model says what is
loaded. Responses are exactly what decide() returns, so a client written against the Python
API reads the same fields over HTTP.

One checkpoint, one process: requests are serialized around the model with a lock, because a
decision is a single forward pass measured in tens of milliseconds, and two interleaved ones
would share mutable per-call state. Scale-out is more processes behind any HTTP balancer,
not threads in this one.

--calibration points at the file `vegaml-calibrate fit` writes, and then every answer is served
through it: the temperature, the conformal threshold and the review band that were fitted on the
caller's own labels, rather than the checkpoint's published defaults. Without it this server
answers with raw published probabilities and no band, which is not what anyone who went to the
trouble of labelling data wants deployed. Each answer also carries an `audit` block naming the
engine, the reader and the calibration version that produced it, because a decision nobody can
re-run is a decision nobody can defend.
"""
from __future__ import annotations

import argparse
import threading
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class DecideRequest(BaseModel):
    state: Any
    questions: Dict[str, Dict[str, Any]]
    mode: Optional[Literal["auto", "engine", "ttt", "both"]] = None
    alpha: str = "0.1"
    views: int = Field(1, ge=1, le=3)
    # Per-call review line, for a caller who wants a band without one being configured on the
    # server. Overrides the server's band; a loaded calibration's band still applies after it.
    review_below: Optional[float] = Field(None, ge=0.0, le=1.0)


class BatchRequest(BaseModel):
    items: List[DecideRequest] = Field(..., min_length=1, max_length=256)


def create_app(model=None, model_name: Optional[str] = None, load_kw: Optional[Dict] = None,
               calibration=None, band=None) -> FastAPI:
    """The FastAPI app. Pass a loaded model (tests do), or a name and the first request loads it.

    `calibration` is a :class:`vegaml.calibrate.Calibration` applied to every answer; `band` is a
    :class:`vegaml.band.Band` for a review line without a full calibration.
    """
    app = FastAPI(title="vegaml", docs_url="/docs")
    app.state.model = model
    app.state.model_name = model_name
    app.state.load_kw = load_kw or {}
    app.state.calibration = calibration
    app.state.band = band
    app.state.lock = threading.Lock()
    # Checked once, on the first request that has both a model and a calibration to compare.
    app.state.checked = False

    def _model():
        with app.state.lock:
            if app.state.model is None:
                import vegaml
                app.state.model = vegaml.load(app.state.model_name, **app.state.load_kw)
            if app.state.calibration is not None and not app.state.checked:
                app.state.checked = True
                for why in app.state.calibration.mismatches(app.state.model):
                    # Served anyway, because refusing to start is worse than saying so, but said
                    # once and loudly: a temperature fitted for another reader is noise.
                    print("vegaml-serve: this calibration does not match the loaded model (%s)" % why)
        return app.state.model

    def _audit(v) -> Dict:
        """Who produced this answer: engine, reader, calibration version. One cheap dict."""
        eng = getattr(v, "engine", v)
        cfg = getattr(eng, "cfg", {}) or {}
        cal = app.state.calibration
        out = {"reader": cfg.get("backbone_name"), "vegaml": _version()}
        try:
            out["engine_sha256"] = eng.base_sha256()
        except Exception:
            out["engine_sha256"] = None
        out["calibration_version"] = cal.version() if cal is not None else None
        out["heads_fitted"] = sorted(getattr(v, "heads", {}))
        return out

    def _version() -> str:
        import vegaml
        return vegaml.__version__

    def _decide(req: DecideRequest) -> Dict:
        v = _model()
        with app.state.lock:
            try:
                out = v.decide(req.state, req.questions, mode=req.mode, alpha=req.alpha,
                               views=req.views, review_below=req.review_below)
            except (ValueError, RuntimeError) as e:
                # A bad question (unknown type, no fitted head in mode="ttt") is the caller's to fix.
                raise HTTPException(status_code=400, detail=str(e)) from e
        # Outside the lock: re-scaling probabilities and comparing them to a threshold touches
        # nothing the model owns.
        if app.state.calibration is not None:
            out = app.state.calibration.apply(out)
        elif app.state.band is not None and req.review_below is None:
            out = app.state.band.apply(out)
        out["audit"] = _audit(v)
        return out

    @app.get("/health")
    def health() -> Dict:
        return {"status": "ok", "model_loaded": app.state.model is not None}

    @app.get("/v1/model")
    def model_info() -> Dict:
        v = _model()
        eng = getattr(v, "engine", None)
        cfg = getattr(eng, "cfg", {}) or {}
        cal = app.state.calibration
        return {"model": app.state.model_name or "default",
                "backbone": cfg.get("backbone_name"), "max_len": cfg.get("max_len"),
                "mode": getattr(v, "mode", None), "heads_fitted": len(getattr(v, "heads", {})),
                "threads": getattr(eng, "threads", None),
                "quantized": getattr(getattr(eng, "perc", None), "quantized", None),
                "calibration": None if cal is None else {
                    "version": cal.version(), "checksum": cal.checksum(), "alpha": cal.alpha,
                    "questions": sorted(cal.questions), "provenance": cal.provenance,
                    "band": None if cal.band is None else {
                        "default": cal.band.default,
                        "questions": {q: b.get("threshold") for q, b in cal.band.questions.items()}},
                },
                "band": None if app.state.band is None else {
                    "default": app.state.band.default,
                    "questions": {q: b.get("threshold")
                                  for q, b in app.state.band.questions.items()}},
                "audit": _audit(v)}

    @app.post("/v1/decide")
    def decide(req: DecideRequest) -> Dict:
        return _decide(req)

    @app.post("/v1/decide/batch")
    def decide_batch(req: BatchRequest) -> Dict:
        return {"results": [_decide(item) for item in req.items]}

    return app


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="vegaml-serve", description="Serve decide() over HTTP.")
    ap.add_argument("--model", default=None,
                    help="checkpoint: nothing for the 0.8B, '4b', a Hub id or a path")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None,
                    help="torch threads for a CPU decision; see docs/cpu.md for the sweep")
    ap.add_argument("--quantize", default=None, choices=["int8"],
                    help="int8 weights for the reader, CPU only (docs/cpu.md)")
    ap.add_argument("--calibration", default=None, metavar="FILE",
                    help="a calibration.json from `vegaml-calibrate fit`, applied to every answer")
    ap.add_argument("--review-below", type=float, default=None, metavar="T",
                    help="route answers under this confidence to review, when no calibration "
                         "carries a band of its own")
    ap.add_argument("--preload", action="store_true",
                    help="load the checkpoint before binding the port instead of on the first request")
    args = ap.parse_args(argv)
    import uvicorn
    load_kw = {}
    if args.device:
        load_kw["device"] = args.device
    if args.max_len:
        load_kw["max_len"] = args.max_len
    if args.threads:
        load_kw["threads"] = args.threads
    if args.quantize:
        load_kw["quantize"] = args.quantize
    # Read before the checkpoint loads: a mistyped path should cost one line, not a model load.
    calibration = band = None
    if args.calibration:
        from .calibrate import Calibration
        try:
            calibration = Calibration.load(args.calibration)
        except OSError as e:
            raise SystemExit("vegaml-serve: --calibration cannot read %r: %s"
                             % (args.calibration, e.strerror)) from None
        except ValueError as e:
            raise SystemExit("vegaml-serve: --calibration %s" % e) from None
        print("vegaml-serve: calibration %s over %d question(s)%s"
              % (calibration.version(), len(calibration.questions),
                 "" if calibration.band is None else ", with a review band"))
    if args.review_below is not None:
        from .band import Band
        band = Band(default=args.review_below)
    model = None
    if args.preload:
        import vegaml
        model = vegaml.load(args.model, **load_kw)
    app = create_app(model=model, model_name=args.model, load_kw=load_kw,
                     calibration=calibration, band=band)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
