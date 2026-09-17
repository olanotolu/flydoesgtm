"""Clayfly HTTP demo server.

  python -m serving.serve                 # http://127.0.0.1:8080
  POST /decide   raw/signals -> decision + signals + pools + costs
                 body {"policy": "pre"|"post"} for training comparison
  GET  /live     real Clay search -> fly decision per company + distribution
  GET  /graph    real soma positions + populations + channel maps
  GET  /replay   deterministic scripted demo JSON

The server is local/demo-oriented; the Clay key stays server-side and
never reaches the browser.
"""
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

from brain.loader import get_brain
from brain.populations import (action_pools, build_channel_map,
                               dan_populations, tracked_set)
from environment.clay_world import ACTIONS, ACTION_COST
from learning.encoder import Encoder
from learning.policy import FlyPolicy
from learning.trace import BatchTrace
from serving.normalize import from_request, normalize_row

MAX_BODY = 64 * 1024
SIGNAL_KEYS = ("funding", "hiring", "intent", "job_change", "negative",
               "trigger")
BASE_COSTS = np.array([2.0, 1.0, 5.0, 15.0])


class FlyService:
    def __init__(self):
        self.brain = get_brain(batch=1, device="cpu")
        self.channels = build_channel_map(self.brain)
        self.tracked = tracked_set(self.brain, self.channels)
        self.encoder = Encoder(self.channels)
        self.trace = BatchTrace(self.brain, self.tracked,
                                 tau=(0.05, 0.15, 0.5))
        self.policies = {}
        for name in ("fly_policy_post.pt", "fly_policy.pt"):
            self._load_policy(name, "post")
            break
        self._load_policy("fly_policy_pre.pt", "pre")
        self.dans = dan_populations(self.brain)
        self.pools = action_pools(self.brain)
        # neuron id -> row in the tracked feature vector, for pool/channel
        # activity readouts
        self._tracked_pos = {int(n): i for i, n in enumerate(self.tracked)}
        self._pool_rows = {
            name: [self._tracked_pos[int(n)] for n in ids
                   if int(n) in self._tracked_pos]
            for name, ids in self.pools.items()}
        self._channel_rows = {
            int(ch): [self._tracked_pos[int(n)] for n in ids
                      if int(n) in self._tracked_pos]
            for ch, ids in self.channels.items()}

    def _load_policy(self, filename, name):
        path = ROOT / "results" / filename
        if not path.exists():
            return
        checkpoint = torch.load(path, map_location="cpu")
        state = checkpoint.get("state", checkpoint)
        weight = state.get("readout.weight")
        n_feat = int(weight.shape[1]) if weight is not None \
            else int(checkpoint.get("n_feat", len(self.tracked)))
        policy = FlyPolicy(n_feat)
        policy.load_state_dict(state, strict=False)
        policy.eval()
        self.policies[name] = policy

    def _population_activity(self, rows):
        """Mean spike trace over a set of tracked rows, summed over scales."""
        if not rows:
            return 0.0
        return float(self.trace.trace[:, rows, 0].sum(axis=0).mean())

    def decide(self, signals, prices=None, budget=100.0,
               researched=0.0, enriched=0.0, policy="post"):
        policy = self.policies.get(policy, self.policies.get("post"))
        if policy is None:
            raise RuntimeError("no trained policy checkpoint found")

        obs = np.zeros((1, 16), np.float32)
        for i, key in enumerate(SIGNAL_KEYS):
            obs[0, i] = signals.get(key, 0.0)
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
        obs[0, 15] = 0.5

        self.brain.reset()
        self.trace.reset()
        for _ in range(10):
            self.brain.step(inject=self.encoder.inject(obs, np.array([0]), 1))
            self.trace.observe(self.brain)
        feat = torch.tensor(self.trace.features([0]))
        x = torch.tensor(obs)
        with torch.no_grad():
            logits, _ = policy(feat, x)
            probs = torch.softmax(logits, 1)[0].numpy()
        decision = ACTIONS[int(probs.argmax())]
        return {
            "decision": decision,
            "confidence": round(float(probs.max()), 4),
            **{f"p_{a.lower()}": round(float(p), 4)
               for a, p in zip(ACTIONS, probs)},
            "probabilities": {a: round(float(p), 4)
                              for a, p in zip(ACTIONS, probs)},
            "signals": {k: round(float(signals.get(k, 0.0)), 4)
                        for k in SIGNAL_KEYS},
            "pools": {name: round(self._population_activity(rows), 4)
                      for name, rows in self._pool_rows.items()},
            "channel_activity": {
                ch: round(self._population_activity(rows), 4)
                for ch, rows in self._channel_rows.items()},
            "costs": {a: round(float(c), 2)
                      for a, c in zip(ACTIONS, costs)},
            "activity": np.round(
                self.trace.trace[..., 0].reshape(-1), 4).tolist(),
            "neuron_count": self.brain.n,
            "connectome": "MaleCNS v1.0",
        }

    def live(self, query, limit=12):
        """Real Clay search -> one fly decision per company."""
        from serving.clay_public import search_companies
        from serving.clay_replay import safe_record
        records = search_companies(query, limit)
        rows = []
        for rec in records:
            raw = safe_record(rec)
            signals = normalize_row(raw)
            d = self.decide(signals)
            rows.append({
                "company": rec.get("name"),
                "domain": rec.get("domain"),
                "size": rec.get("size"),
                "industry": rec.get("industry"),
                "location": rec.get("location"),
                "signals": signals,
                "decision": d["decision"],
                "confidence": d["confidence"],
                "probabilities": d["probabilities"],
                "pools": d["pools"],
                "channel_activity": d["channel_activity"],
                "costs": d["costs"],
            })
        dist = Counter(r["decision"] for r in rows)
        return {"query": query, "rows": rows,
                "distribution": dict(dist),
                "source": "clay public api (read-only)"}

    def graph(self, max_points=24000):
        rng = np.random.default_rng(42)
        keep = rng.choice(self.brain.n,
                          size=min(max_points, self.brain.n), replace=False)
        keep.sort()
        positions = np.nan_to_num(self.brain.positions[keep], nan=0.0)
        positions -= positions.mean(0)
        scale = np.abs(positions).max() or 1.0
        positions = (positions / scale).round(4)
        return {"n": self.brain.n, "points": positions.tolist(),
                "point_indices": keep.tolist(),
                "groups": {k: v.tolist() for k, v in self.brain.groups.items()},
                "pools": {k: v.tolist() for k, v in self.pools.items()},
                "channels": {str(k): v.tolist()
                             for k, v in self.channels.items()},
                "signal_keys": list(SIGNAL_KEYS)}


class Handler(BaseHTTPRequestHandler):
    service = None

    def send_json(self, status, value):
        body = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlsplit(self.path).path
        try:
            if path == "/graph":
                self.send_json(200, self.service.graph())
            elif path == "/replay":
                self.send_json(200, json.loads(
                    (ROOT / "demo" / "replay.json").read_text()))
            elif path == "/live":
                qs = parse_qs(urlsplit(self.path).query)
                query = qs.get("query", [""])[0]
                limit = int(qs.get("limit", ["12"])[0])
                self.send_json(200, self.service.live(query, limit))
            else:
                page = ROOT / "demo" / "web" / "index.html"
                body = page.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        except Exception as e:
            self.send_json(500, {"error": str(e)})

    def do_POST(self):
        if urlsplit(self.path).path != "/decide":
            self.send_json(404, {"error": "not found"})
            return
        try:
            size = int(self.headers.get("Content-Length", 0))
            if size < 0 or size > MAX_BODY:
                raise ValueError("request body too large")
            req = json.loads(self.rfile.read(size) or b"{}")
            out = self.service.decide(
                from_request(req), prices=req.get("prices"),
                budget=float(req.get("budget", 100)),
                researched=float(req.get("researched", 0)),
                enriched=float(req.get("enriched", 0)),
                policy=str(req.get("policy", "post")))
            self.send_json(200, out)
        except Exception as e:
            self.send_json(400, {"error": str(e)})

    def log_message(self, *_):
        pass


def main():
    import demo.scenarios as scenarios
    scenarios.write(str(ROOT / "demo" / "replay.json"))
    Handler.service = FlyService()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    print(f"the fly got a job at clay — http://127.0.0.1:{port}")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
