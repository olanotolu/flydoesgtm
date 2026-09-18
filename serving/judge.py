"""Cloud judge arm — scores the same account state via Cloudflare Workers AI.

Two engines:
- ``typesafe/jev`` — calibrated structured evaluation (noul/choice/score).
  Requires Unified Billing credits on the account; when the gateway has no
  balance the call fails and we degrade to Llama.
- ``@cf/meta/llama-3.1-8b-instruct-fast`` — free first-party model. Returns an
  action word, not probabilities — reported as uncalibrated.

Returns the same shape as ``baseline_eval`` so the response can carry a third
arm. All failures return ``None`` — the UI note simply stays hidden.
"""
import json
import os
import urllib.request
from pathlib import Path

BASE = "https://api.cloudflare.com/client/v4/accounts"
JEV_MODEL = "typesafe/jev"
LLAMA_MODEL = "@cf/meta/llama-3.1-8b-instruct-fast"

ACTIONS = ["WAIT", "OBSERVE", "RESEARCH", "ENRICH", "EMAIL", "ESCALATE", "IGNORE"]

_ACTION_CRITERIA = {
    "WAIT": "Hold the account; revisit later — spend nothing now.",
    "OBSERVE": "Keep watching passively; spend nothing.",
    "RESEARCH": "Pay to research the account further before acting.",
    "ENRICH": "Pay to enrich/verify contact data.",
    "EMAIL": "Send outreach now — the signals justify the spend.",
    "ESCALATE": "Escalate to a senior, human-led motion.",
    "IGNORE": "Write the account off permanently.",
}


def _env_or_file(name):
    if os.environ.get(name):
        return os.environ[name]
    for fname in (".env.local", ".env"):
        p = Path(__file__).parents[1] / fname
        if p.exists():
            for line in p.read_text().splitlines():
                if line.startswith(name + "="):
                    return line.split("=", 1)[1].strip()
    return None


def _cf_token():
    token = _env_or_file("CF_API_TOKEN") or _env_or_file("CLOUDFLARE_API_TOKEN")
    if token:
        return token
    # Fall back to wrangler's OAuth token — already a valid Bearer for the API.
    for p in (Path.home() / "Library/Preferences/.wrangler/config/default.toml",
              Path.home() / ".wrangler/config/default.toml"):
        if p.exists():
            try:
                import tomllib
                return tomllib.loads(p.read_text()).get("oauth_token")
            except Exception:
                return None
    return None


def _account():
    return _env_or_file("CF_ACCOUNT_ID") or _env_or_file("CLOUDFLARE_ACCOUNT_ID")


def _post(path_or_model, payload, timeout):
    account, token = _account(), _cf_token()
    if not account or not token:
        return None
    url = f"{BASE}/{account}/ai/run"
    if path_or_model.startswith("@"):
        url += "/" + path_or_model
        body = payload
    else:
        body = {"model": path_or_model, "input": payload}
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
    except Exception:
        return None
    if not data.get("success", True):
        return None
    return data.get("result", data)


def _state(signals, context, evidence):
    state = {"signals": signals, "context": context or {}}
    if isinstance(evidence, dict):
        state["evidence"] = {
            k: v for k, v in evidence.items() if isinstance(v, dict)}
    return state


def _jev(signals, context, evidence):
    result = _post(JEV_MODEL, {
        "state": _state(signals, context, evidence),
        "questions": {"action": {
            "type": "choice",
            "instructions": "Best next GTM action for this account, weighing "
                            "signal strength against the cost of acting?",
            "criteria": _ACTION_CRITERIA}}},
        timeout=8)
    if not result:
        return None
    ans = (result.get("answers") or {}).get("action") or {}
    choice = ans.get("choice")
    if choice not in ACTIONS:
        return None
    probs = {k: round(float(v), 4)
             for k, v in (ans.get("probabilities") or {}).items()}
    return {"policy_action": choice, "engine": "jev",
            "model": result.get("model", "jev"),
            "confidence": round(max(probs.values() or [0]), 4),
            "probabilities": probs}


def _llama(signals, context, evidence):
    state = _state(signals, context, evidence)
    result = _post(LLAMA_MODEL, {"messages": [
        {"role": "system", "content":
         "You are a GTM triage judge. Reply with exactly one action word "
         "(WAIT, OBSERVE, RESEARCH, ENRICH, EMAIL, ESCALATE, or IGNORE), "
         "then one short reason."},
        {"role": "user", "content":
         "Account state as JSON: " + json.dumps(state)}]}, timeout=8)
    text = ((result or {}).get("response") or "").strip()
    action = next((a for a in ACTIONS
                   if text.upper().startswith(a)), None)
    if not action:
        return None
    reason = text.split("\n", 1)[-1].split(" ", 1)[-1].strip() or None
    return {"policy_action": action, "engine": "llama",
            "model": "llama-3.1-8b", "confidence": None,
            "reason": reason, "probabilities": None}


def judge_decision(signals, context=None, evidence=None):
    """Score the same account state with a cloud model. ``None`` on any
    failure — mirrors ``baseline_eval``'s degrade-to-absent contract."""
    mode = os.environ.get("FLY_JUDGE", "auto")
    if mode == "off":
        return None
    if mode in ("auto", "jev"):
        out = _jev(signals, context, evidence)
        if out or mode == "jev":
            return out
    return _llama(signals, context, evidence)
