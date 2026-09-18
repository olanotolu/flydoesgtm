"""Live think endpoint on Modal — the real serving path over HTTPS.

    modal deploy modal_serve.py
    curl -X POST https://<workspace>--clayfly-serve-think.modal.run/think \
        -H 'content-type: application/json' \
        -d '{"evidence": {...}, "context": {}}'
    curl https://<workspace>--clayfly-serve-think.modal.run/health

Runs exactly what serving/serve.py's /api/demo/think runs locally:
evidence dict -> adapt_evidence_to_signals -> FlyService.decide (12-step
MaleCNS connectome sim + the trained cosine-v2 readout) ->
build_think_response, plus the dense MLP baseline arm. The cloud judge
arm degrades to absent without CF credentials, same as locally.

Brain (~260MB) and checkpoints live on the `clayfly-data` volume:
  /data/fly-data/{brain,weights}.npz   (via FLY_DATA env var)
  /data/results_v2/fly_policy.pt       (the served policy)
  /data/results_mlp_v2/mlp_policy.pt   (the dense baseline)
The FlyService singleton is built once per container when the ASGI app
factory runs; warm requests pay only the ~2s sim. min_containers=1 keeps
one container warm — that is a 24/7 cost; drop it to tolerate a cold
start that is mostly the image pull + ~10s of brain/checkpoint load.
"""
import modal

# Same image spec as modal_train/modal_eval (plus fastapi for the ASGI
# endpoint); Modal caches the shared layers so only the delta builds.
img = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04",
        add_python="3.12",
    )
    .pip_install("flybrain[gpu]", "torch", "numpy", "scipy", "fastapi")
    .add_local_python_source("brain", "environment", "learning",
                             "experiments", "serving")
)

vol = modal.Volume.from_name("clayfly-data", create_if_missing=True)
app = modal.App("clayfly-serve")


def _build_service():
    """FlyService pointed at the volume checkpoints, serve.py unmodified.

    FlyService resolves policies as ``ROOT / "results" / <file>`` with
    ROOT fixed at the repo root. On Modal the checkpoints live under
    /data/results_*/, so hand it a staging ROOT whose results/ directory
    is symlinks into the volume.
    """
    import os
    from pathlib import Path

    import serving.serve as srv

    root = Path("/tmp/fly-root")
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    for name, target in {
        "fly_policy.pt": "/data/results_v2/fly_policy.pt",
        "mlp_policy.pt": "/data/results_mlp_v2/mlp_policy.pt",
    }.items():
        link = results / name
        if not link.exists() and Path(target).exists():
            os.symlink(target, link)
    srv.ROOT = root
    return srv.FlyService()


@app.function(image=img, volumes={"/data": vol}, cpu=2, memory=8192,
              timeout=120, min_containers=1, scaledown_window=300,
              env={"FLY_DATA": "/data/fly-data"})
@modal.asgi_app(label="think")
def think_api():
    import threading

    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    service = _build_service()   # brain + policies load once per container
    lock = threading.Lock()      # decide() mutates brain state — serialize

    api = FastAPI()
    # Called cross-origin from the hosted demo; middleware answers the
    # OPTIONS preflight too.
    api.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    def _think(req: dict):
        """serving/serve.py's /api/demo/think handler, verbatim."""
        from serving.think import (build_think_response, clamp_energies,
                                   recommend)
        ctx = req.get("context") or {}
        if "evidence" in req:
            from serving.evidence import adapt_evidence_to_signals
            adapted = adapt_evidence_to_signals(req["evidence"])
            signals = clamp_energies(adapted["signals"])
            out = service.decide(
                signals, provenance=adapted["provenance"],
                researched=float(ctx.get("researched", 0)),
                enriched=float(ctx.get("enriched", 0)),
                budget=float(ctx.get("budget", 100)),
                pain=ctx.get("pain"))
            out["sources"] = adapted["sources"]
            out["excluded"] = adapted["excluded"]
        else:
            signals = clamp_energies(req.get("signals", {}))
            out = service.decide(
                signals,
                researched=float(ctx.get("researched", 0)),
                enriched=float(ctx.get("enriched", 0)),
                budget=float(ctx.get("budget", 100)),
                pain=ctx.get("pain"))
        resp = build_think_response(signals, out)
        baseline = service.baseline_eval(out.get("observation"))
        if baseline:
            resp["baseline"] = baseline
            resp["baseline_recommendation"], _ = recommend(
                baseline["policy_action"], 0)
        from serving.judge import judge_decision
        judge = judge_decision(signals, ctx, evidence=req.get("evidence"))
        if judge:
            resp["judge"] = judge
            resp["judge_recommendation"], _ = recommend(
                judge["policy_action"], 0)
        # The local response omits the raw action; the hosted demo's live
        # contract lists it, so surface it alongside what it already sends.
        resp["policy_action"] = out["policy_action"]
        return resp

    @api.post("/think")
    @api.post("/api/demo/think")  # same path the demo page already posts to
    def think(req: dict):
        try:
            with lock:
                return _think(req)
        except (ValueError, TypeError):
            return JSONResponse({"error": "request body is invalid"},
                                status_code=400)
        except Exception:
            return JSONResponse({"error": "request could not be completed"},
                                status_code=500)

    @api.get("/health")
    @api.get("/api/health")
    def health():
        return {**service.health(), "live": True}

    return api
