"""Local Fly Brain × Clay demonstration server.

The only real-data mode is ``live_draft``. It can search and enrich Clay
records, score them, and produce unsent drafts. There is deliberately no
send/enroll/reply/CRM mutation endpoint.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
import torch

ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain.loader import get_brain, weights_checksum
from brain.populations import (action_pools, build_channel_map,
                               dan_populations, tracked_set)
from environment.clay_world import ACTIONS, ACTION_COST
from learning.encoder import Encoder
from learning.policy import FlyPolicy, MLPPolicy, READOUT_ARCH
from learning.trace import BatchTrace
from serving.clay_live import DemoBudget, credit_balance, run_routine
from serving.clay_records import safe_record
from serving.contracts import (MAX_RECORDS, demo_action, as_signal_map,
                               utc_now)
from serving.drafts import draft_artifact
from serving.normalize import from_request, normalize_row

MAX_BODY = 64 * 1024
MAX_QUERY_LENGTH = 2000
SIGNAL_KEYS = ("funding", "hiring", "intent", "job_change", "negative", "trigger")
BASE_COSTS = np.array([2.0, 1.0, 5.0, 15.0])
SIM_STEPS = int(os.environ.get("FLY_SIM_STEPS", "4"))
DEFAULT_QUERY = (
    'select from people where location_country = "United States" '
    'and experiences.any(is_current = true and '
    'job_title is_similar_to ("Owner", "Founder", "Chief Operating Officer", '
    '"Vice President of Operations", "Director of Operations", '
    '"Regional Director of Operations") and '
    'company.industry in ("Restaurants", "Food and Beverage Services") and '
    'company.company_size in ("11-50", "51-200", "201-500"))'
)


def neuron_activity_block(brain, spike_counts, per_step, frames=None):
    window_s = max(1e-6, len(per_step) * float(brain.dt))
    top = []
    for nid, spikes in sorted(spike_counts.items(),
                              key=lambda kv: (-kv[1], kv[0]))[:8]:
        idx = int(nid)
        superclass = (str(brain.superclass[idx])
                      if brain.superclass is not None else "")
        label = str(brain.cell_type[idx]).strip() or superclass
        top.append({
            "id": idx,
            "label": label,
            "side": str(brain.side[idx]) if brain.side is not None else "",
            "region": superclass,
            "spikes": int(spikes),
        })
    rates = [c / window_s for c in spike_counts.values()] or [0.0]
    return {
        "steps": len(per_step),
        "dt_ms": round(float(brain.dt) * 1000),
        "window_ms": round(window_s * 1000),
        "active": len(spike_counts),
        "spikes": int(sum(spike_counts.values())),
        "per_step": [int(c) for c in per_step],
        "frames": [[int(n) for n in frame] for frame in (frames or [])],
        "hz_max": float(np.ceil(max(rates) / 10.0) * 10.0),
        "top": top,
    }


class FlyService:
    def __init__(self):
        self.brain = get_brain(batch=1, device="cpu")
        self.channels = build_channel_map(self.brain)
        self.tracked = tracked_set(self.brain, self.channels)
        self.encoder = Encoder(self.channels)
        self.trace = BatchTrace(self.brain, self.tracked,
                                tau=(0.05, 0.15, 0.5))
        self.policies = {}
        self.checkpoints = {}
        self.rejected = {}
        for filename in ("fly_policy_post.pt", "fly_policy.pt"):
            if self._load_policy(filename, "post"):
                break
        self._load_policy("fly_policy_pre.pt", "pre")
        self._load_baseline()
        self.dans = dan_populations(self.brain)
        self.pools = action_pools(self.brain)
        self._tracked_pos = {int(n): i for i, n in enumerate(self.tracked)}
        self._pool_rows = {
            name: [self._tracked_pos[int(n)] for n in ids
                   if int(n) in self._tracked_pos]
            for name, ids in self.pools.items()
        }
        self._channel_rows = {
            int(ch): [self._tracked_pos[int(n)] for n in ids
                      if int(n) in self._tracked_pos]
            for ch, ids in self.channels.items()
        }

    def _load_baseline(self):
        """Parameter-matched dense baseline — same obs input, no brain."""
        path = ROOT / "results" / "mlp_policy.pt"
        if not path.exists():
            return
        try:
            checkpoint = torch.load(path, map_location="cpu",
                                    weights_only=False)
            state = checkpoint.get("state", checkpoint)
            hidden = int(state["net.0.weight"].shape[0])
            mlp = MLPPolicy(16, hidden)
            mlp.load_state_dict(state, strict=True)
            mlp.eval()
            self.policies["mlp"] = mlp
            self.checkpoints["mlp"] = {
                "filename": "mlp_policy.pt",
                "model_version": checkpoint.get(
                    "model_version", "mlp-baseline"),
                "sim_steps": int(checkpoint.get("sim_steps") or SIM_STEPS),
                "n_feat": 12012,
            }
        except Exception as exc:
            self.rejected["mlp"] = {"filename": "mlp_policy.pt",
                                    "reason": str(exc)}

    def baseline_eval(self, observation):
        """Score the same observation with the dense baseline readout."""
        mlp = self.policies.get("mlp")
        if mlp is None or not observation:
            return None
        obs = torch.as_tensor(
            np.asarray([observation], dtype=np.float32))
        feat = torch.zeros((1, 12012), dtype=torch.float32)
        with torch.no_grad():
            logits, _ = mlp(feat, obs)
        probs = torch.softmax(logits, 1)[0].numpy()
        return {
            "policy_action": ACTIONS[int(probs.argmax())],
            "confidence": round(float(probs.max()), 4),
            "probabilities": {a: round(float(p), 4)
                              for a, p in zip(ACTIONS, probs)},
        }

    @property
    def model_version(self):
        return next(iter(self.checkpoints.values()), {}).get(
            "model_version", "untrained-fallback")

    def _load_policy(self, filename, name):
        path = ROOT / "results" / filename
        if not path.exists():
            return False
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            state = checkpoint.get("state", checkpoint)
            weight = state.get("readout.weight")
            expected = len(self.tracked) * 3
            n_feat = int(weight.shape[1]) if weight is not None else int(
                checkpoint.get("n_feat", expected))
            if n_feat != expected:
                # Do not fall through silently. A checkpoint trained against
                # a different tracked population is incompatible, and quietly
                # serving a different file instead is how a stale artifact
                # gets presented as the trained policy.
                self.rejected[name] = {
                    "filename": filename,
                    "reason": "n_feat mismatch",
                    "checkpoint_n_feat": n_feat,
                    "expected_n_feat": expected,
                    "expected_tracked": len(self.tracked),
                }
                return False
            # The readout's semantics have changed twice (linear -> cosine ->
            # bounded cosine) and weights do not port between them: a linear
            # readout encoded its logit scale in |W|, which a cosine readout
            # discards, so the weights would load and produce arbitrary
            # decisions. Compare the declared architecture rather than
            # guessing from which keys happen to be present.
            if checkpoint.get("readout_arch") != READOUT_ARCH:
                declared = checkpoint.get("readout_arch")
                if declared is None:
                    # Older checkpoints predate the stamp. Distinguish the
                    # two by whether they carry the temperature parameter.
                    declared = ("cosine-v1" if "temperature" in state
                                else "linear (pre-cosine)")
                self.rejected[name] = {
                    "filename": filename,
                    "reason": "readout architecture mismatch: checkpoint is "
                              f"{declared} but this build serves "
                              f"{READOUT_ARCH}; retrain",
                    "checkpoint_n_feat": n_feat,
                }
                return False
            policy = FlyPolicy(
                n_feat,
                use_raw_head=bool(checkpoint.get("use_raw_head", True)),
            )
            policy.load_state_dict(state, strict=True)
            policy.eval()
            self.policies[name] = policy
            gains = checkpoint.get("encoder_gains")
            if gains is not None and len(gains) != 16:
                gains = None
            self.checkpoints[name] = {
                "filename": filename,
                "model_version": checkpoint.get("model_version", filename),
                "w_sha256": checkpoint.get("w_sha256"),
                "checkpoint_sha256": checkpoint.get("checkpoint_sha256"),
                "sim_steps": int(checkpoint.get("sim_steps") or SIM_STEPS),
                "seed": checkpoint.get("seed"),
                "use_raw_head": checkpoint.get("use_raw_head", True),
                "encoder_gains": [float(g) for g in gains] if gains else None,
                "n_feat": n_feat,
                "tracked": len(self.tracked),
            }
            # The sensory interface is part of the checkpoint. Loading the
            # readout without its gains silently changes what the fly sees,
            # which is how a trained policy turns into a constant.
            if name == "post" and gains is not None:
                self.encoder.gains = np.asarray(gains, np.float32)
            return True
        except Exception as exc:
            # A corrupt checkpoint (EOFError, BadZipFile, UnpicklingError...)
            # is a rejected policy, not a server crash.
            self.rejected[name] = {"filename": filename,
                                   "reason": f"{type(exc).__name__}: {exc}"}
            return False

    def reset_session(self):
        self.brain.reset()
        self.trace.reset()

    def _population_activity(self, rows):
        if not rows:
            return 0.0
        return float(self.trace.trace[:, rows, 0].sum(axis=0).mean())

    def decide(self, signals, prices=None, budget=100.0, researched=0.0,
               enriched=0.0, policy="post", *, reset=True, mode="replay",
               provenance=None, pain=None):
        policy_model = self.policies.get(policy, self.policies.get("post"))
        if policy_model is None:
            raise RuntimeError("no compatible trained policy checkpoint found")
        # The horizon is a property of the checkpoint, not of the server.
        # Serving a 12-step readout at 4 steps (or the reverse) puts the
        # policy off the feature distribution it was trained on.
        meta = self.checkpoints.get(policy) or self.checkpoints.get("post") or {}
        sim_steps = int(meta.get("sim_steps") or SIM_STEPS)
        if reset:
            self.reset_session()

        obs = np.zeros((1, 16), np.float32)
        for i, key in enumerate(SIGNAL_KEYS):
            obs[0, i] = np.clip(float(signals.get(key, 0.5)), 0, 1)
        obs[0, 6] = 1.0
        obs[0, 7] = 0.5
        obs[0, 8] = np.clip(researched, 0, 1)
        obs[0, 9] = np.clip(enriched, 0, 1)
        costs = ACTION_COST.copy()
        if prices:
            for name, value in prices.items():
                if name in ("research", "enrich", "email", "escalate"):
                    idx = ACTIONS.index(name.upper())
                    costs[idx] = max(0.0, float(value))
        obs[0, 10:14] = np.clip(costs[[2, 3, 4, 5]] / BASE_COSTS, 0, 2)
        obs[0, 14] = np.clip(budget / 100.0, 0, 2)
        obs[0, 15] = 0.5 if pain is None else np.clip(float(pain), 0, 1)

        spike_counts = {}
        per_step = []
        frames = []
        for _ in range(sim_steps):
            self.brain.step(inject=self.encoder.inject(obs, np.array([0]), 1))
            fired = np.asarray(self.brain.fired).reshape(-1)
            fired = fired[fired < self.brain.n]
            unique = np.unique(fired)
            per_step.append(int(unique.size))
            frames.append([int(n) for n in unique])
            for nid in unique:
                key = int(nid)
                spike_counts[key] = spike_counts.get(key, 0) + 1
            self.trace.observe(self.brain)
        feat = torch.as_tensor(self.trace.features([0]))
        x = torch.as_tensor(obs)
        with torch.no_grad():
            logits, _ = policy_model(feat, x)
            probs = torch.softmax(logits, 1)[0].numpy()
        policy_action = ACTIONS[int(probs.argmax())]
        action = demo_action(policy_action) if mode == "live_draft" else policy_action
        return {
            "decision": action,
            "policy_action": policy_action,
            "confidence": round(float(probs.max()), 4),
            "probabilities": {a: round(float(p), 4)
                              for a, p in zip(ACTIONS, probs)},
            "signals": {k: round(float(signals.get(k, 0.5)), 4)
                        for k in SIGNAL_KEYS},
            "pools": {name: round(self._population_activity(rows), 4)
                      for name, rows in self._pool_rows.items()},
            "channel_activity": {
                ch: round(self._population_activity(rows), 4)
                for ch, rows in self._channel_rows.items()
            },
            "costs": {a: round(float(c), 2)
                      for a, c in zip(ACTIONS, costs)},
            "activity": np.round(
                self.trace.trace[..., 0].reshape(-1), 4).tolist(),
            "observation": np.round(obs[0], 4).tolist(),
            "neuron_activity": neuron_activity_block(
                self.brain, spike_counts, per_step, frames),
            "neuron_count": self.brain.n,
            "connectome": "MaleCNS v1.0",
            "model_version": self.model_version,
            "mode": mode,
            "sim_steps": sim_steps,
            "provenance": provenance or [],
        }

    def live(self, query=DEFAULT_QUERY, limit=MAX_RECORDS):
        """Search real Clay data and produce only unsent draft artifacts."""
        from serving.clay_public import search_records

        query = (query or DEFAULT_QUERY).strip()
        if len(query) > MAX_QUERY_LENGTH:
            raise ValueError("query is too long")
        budget = DemoBudget()
        try:
            records = search_records(query, min(int(limit), MAX_RECORDS))
        except Exception as exc:
            budget.events.append({"source": "clay_search", "status": "fallback",
                                  "reason": str(exc)[:180]})
            return {
                "query": query, "rows": [], "distribution": {},
                "budget": budget.as_dict(), "preflight": {"status": "not_requested"},
                "safe_to_contact": False, "mode": "live_draft",
                "source": "clay public api / safe fallback",
            }
        rows, seen = [], set()
        live_routines = os.environ.get("CLAYFLY_LIVE_ROUTINES") == "1"
        preflight = None
        if live_routines:
            try:
                preflight = credit_balance()
                remaining = float(preflight.get("balance", 0))
                if remaining - budget.max_credits < budget.min_remaining_credits:
                    live_routines = False
                    budget.events.append({
                        "source": "clay_credits",
                        "status": "fallback_replay",
                        "reason": "credit floor cannot be guaranteed",
                        "credits_remaining": remaining,
                    })
            except RuntimeError as exc:
                live_routines = False
                budget.events.append({
                    "source": "clay_credits", "status": "preflight_unavailable",
                    "reason": str(exc)[:180],
                })
        for record in records:
            raw = safe_record(record)
            company = raw.get("company") or "unknown"
            key = str(company).strip().lower()
            if key in seen:
                continue
            seen.add(key)
            budget.admit_record()
            normalized = normalize_row(raw)
            provenance = [{
                "source": "clay_search",
                "retrieved_at": utc_now(),
                "record_id": record.get("clay_profile_id") or record.get("clay_company_id"),
                "fields": sorted(k for k, v in raw.items() if v is not None),
            }]
            thin = self.decide(normalized, mode="live_draft",
                               provenance=provenance)
            final = thin
            enrichment = None
            if thin["policy_action"] == "RESEARCH" and live_routines:
                try:
                    enrichment = run_routine(
                        "RESEARCH",
                        {**raw, "record_id": record.get("clay_profile_id")
                         or record.get("clay_company_id"),
                         "routine_name": "Enrich Company"},
                        live=True, budget=budget,
                    )
                    final = self.decide(
                        normalized, researched=1.0, mode="live_draft",
                        reset=False,
                        provenance=provenance + [{
                            "source": "clay_workflow",
                            "run_id": enrichment.get("run_id"),
                            "status": enrichment.get("status"),
                        }],
                    )
                except RuntimeError as exc:
                    enrichment = {"mode": "live_draft", "status": "skipped",
                                  "reason": str(exc)[:180]}
                    provenance.append({"source": "clay_workflow",
                                       "status": "skipped",
                                       "reason": str(exc)[:180]})
            row = {
                "company": company,
                "domain": raw.get("domain"),
                "title": raw.get("title"),
                "size": record.get("size") or record.get("company_size"),
                "industry": raw.get("industry"),
                "location": raw.get("location"),
                "signals": final["signals"],
                "decision": final["decision"],
                "policy_action": final["policy_action"],
                "confidence": final["confidence"],
                "probabilities": final["probabilities"],
                "pools": final["pools"],
                "channel_activity": final["channel_activity"],
                "costs": final["costs"],
                "provenance": final["provenance"],
                "enrichment": enrichment,
                "draft": draft_artifact(raw, final)
                if final["decision"] == "DRAFT_EMAIL" else None,
            }
            rows.append(row)
        dist = Counter(r["decision"] for r in rows)
        return {
            "query": query,
            "rows": rows,
            "distribution": dict(dist),
            "budget": budget.as_dict(),
            "preflight": preflight or {"status": "not_requested"},
            "safe_to_contact": False,
            "mode": "live_draft",
            "source": "clay public api / safe workflow",
        }

    def graph(self, max_points=24000):
        rng = np.random.default_rng(42)
        keep = rng.choice(self.brain.n,
                          size=min(max_points, self.brain.n), replace=False)
        keep.sort()
        positions = np.nan_to_num(self.brain.positions[keep], nan=0.0)
        positions -= positions.mean(0)
        scale = np.abs(positions).max() or 1.0
        positions = (positions / scale).round(4)
        return {
            "n": self.brain.n,
            "points": positions.tolist(),
            "point_indices": keep.tolist(),
            "groups": {k: v.tolist() for k, v in self.brain.groups.items()},
            "pools": {k: v.tolist() for k, v in self.pools.items()},
            "channels": {str(k): v.tolist() for k, v in self.channels.items()},
            "signal_keys": list(SIGNAL_KEYS),
        }

    def health(self):
        served = self.checkpoints.get("post") or {}
        return {
            "ok": bool(self.policies),
            "mode": "replay/live_draft",
            "policy_loaded": sorted(self.policies),
            "model_version": self.model_version,
            # Report what is actually being served, not what the server
            # would default to. These two drifting apart is the bug.
            "sim_steps": int(served.get("sim_steps") or SIM_STEPS),
            "encoder_gains": served.get("encoder_gains"),
            "use_raw_head": served.get("use_raw_head"),
            "tracked": served.get("tracked"),
            "n_feat": served.get("n_feat"),
            "rejected_checkpoints": self.rejected,
            "connectome_sha256": weights_checksum(brain=self.brain),
            "checkpoints": self.checkpoints,
            "safe_to_contact": False,
        }

    def artifacts(self):
        return {
            "model_version": self.model_version,
            "connectome_sha256": weights_checksum(brain=self.brain),
            "checkpoints": self.checkpoints,
            "rejected_checkpoints": self.rejected,
            "experiment": "legacy-compatible serving until a corrected bundle is installed",
            "claims": "real Clay mode is orchestration evidence, not revenue evidence",
        }


def resolve_static_asset(path: str) -> Path | None:
    if not path.startswith("/") or Path(path).suffix not in {".png", ".js"}:
        return None
    name = path[1:]
    if name.startswith("demo/"):
        name = name[len("demo/"):]
    if "/" in name or name.startswith(".") or not name:
        return None
    asset = ROOT / "demo" / name
    return asset if asset.is_file() else None


class Handler(BaseHTTPRequestHandler):
    service = None

    def _allowed_origin(self):
        allowed = {x.strip() for x in os.environ.get(
            "FLY_ALLOWED_ORIGINS",
            "http://127.0.0.1:8090,http://localhost:8090,null",
        ).split(",") if x.strip()}
        origin = self.headers.get("Origin")
        return origin if origin in allowed else next(iter(allowed), "")

    def send_json(self, status, value):
        body = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", self._allowed_origin())
        self.send_header("Vary", "Origin")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", self._allowed_origin())
        self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlsplit(self.path).path
        try:
            if path == "/graph":
                self.send_json(200, self.service.graph())
            elif path in ("/replay", "/api/demo/replay/default"):
                self.send_json(200, json.loads(
                    (ROOT / "demo" / "replay.json").read_text()))
            elif path in ("/health", "/api/health"):
                self.send_json(200, self.service.health())
            elif path == "/api/artifacts/current":
                self.send_json(200, self.service.artifacts())
            elif path == "/api/demo/eval":
                summary_path = ROOT / "results" / "eval_summary.json"
                if not summary_path.exists():
                    self.send_json(404, {"error": "eval not run yet"})
                else:
                    self.send_json(200, json.loads(summary_path.read_text()))
            elif path == "/live":
                qs = parse_qs(urlsplit(self.path).query)
                query = qs.get("query", [DEFAULT_QUERY])[0]
                limit = min(int(qs.get("limit", [str(MAX_RECORDS)])[0]), MAX_RECORDS)
                self.send_json(200, self.service.live(query, limit))
            elif (asset := resolve_static_asset(path)) is not None:
                body = asset.read_bytes()
                self.send_response(200)
                content_type = ("application/javascript; charset=utf-8"
                                if asset.suffix == ".js" else "image/png")
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                page = ROOT / "demo" / "web" / "index.html"
                body = page.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Security-Policy",
                                 "default-src 'self'; connect-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; img-src 'self' data:")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        except Exception:
            self.send_json(500, {"error": "request could not be completed"})

    def do_POST(self):
        path = urlsplit(self.path).path
        if path not in ("/decide", "/api/demo/run", "/api/demo/think"):
            self.send_json(404, {"error": "route not found"})
            return
        try:
            size = int(self.headers.get("Content-Length", 0))
            if size < 0 or size > MAX_BODY:
                raise ValueError("request body too large")
            req = json.loads(self.rfile.read(size) or b"{}")
            if path == "/api/demo/think":
                from serving.think import (build_think_response,
                                           clamp_energies, recommend)
                ctx = req.get("context") or {}
                if "evidence" in req:
                    from serving.evidence import adapt_evidence_to_signals
                    adapted = adapt_evidence_to_signals(req["evidence"])
                    signals = clamp_energies(adapted["signals"])
                    out = self.service.decide(
                        signals, provenance=adapted["provenance"],
                        researched=float(ctx.get("researched", 0)),
                        enriched=float(ctx.get("enriched", 0)),
                        budget=float(ctx.get("budget", 100)),
                        pain=ctx.get("pain"))
                    out["sources"] = adapted["sources"]
                    out["excluded"] = adapted["excluded"]
                else:
                    signals = clamp_energies(req.get("signals", {}))
                    out = self.service.decide(
                        signals,
                        researched=float(ctx.get("researched", 0)),
                        enriched=float(ctx.get("enriched", 0)),
                        budget=float(ctx.get("budget", 100)),
                        pain=ctx.get("pain"))
                resp = build_think_response(signals, out)
                baseline = self.service.baseline_eval(
                    out.get("observation"))
                if baseline:
                    resp["baseline"] = baseline
                    resp["baseline_recommendation"], _ = recommend(
                        baseline["policy_action"], 0)
                from serving.judge import judge_decision
                judge = judge_decision(
                    signals, ctx, evidence=req.get("evidence"))
                if judge:
                    resp["judge"] = judge
                    resp["judge_recommendation"], _ = recommend(
                        judge["policy_action"], 0)
                self.send_json(200, resp)
                return
            if path == "/api/demo/run":
                mode = req.get("mode", "replay")
                if mode == "replay":
                    result = json.loads((ROOT / "demo" / "replay.json").read_text())
                elif mode == "live_draft":
                    result = self.service.live(req.get("query", DEFAULT_QUERY),
                                               min(int(req.get("limit", MAX_RECORDS)), MAX_RECORDS))
                else:
                    raise ValueError("mode must be replay or live_draft")
                self.send_json(200, result)
                return
            out = self.service.decide(
                from_request(req), prices=req.get("prices"),
                budget=float(req.get("budget", 100)),
                researched=float(req.get("researched", 0)),
                enriched=float(req.get("enriched", 0)),
                policy=str(req.get("policy", "post")), mode="replay",
            )
            self.send_json(200, out)
        except (ValueError, TypeError, json.JSONDecodeError):
            self.send_json(400, {"error": "request body is invalid"})
        except Exception:
            self.send_json(500, {"error": "request could not be completed"})

    def log_message(self, *_):
        pass


def main():
    Handler.service = FlyService()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
    print(f"Fly Brain × Clay — http://127.0.0.1:{port}")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
