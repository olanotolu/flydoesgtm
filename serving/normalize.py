"""Raw Clay-style row -> six normalized sensory channels [0,1]."""
import math

RAW_FIELDS = {
    "funding": [("total_funding_usd", 25e6), ("latest_raise_usd", 5e6)],
    "hiring": [("open_roles", 15), ("headcount_growth_pct", 10)],
    "intent": [("website_visits_mom", 5000), ("intent_score", 0.5),
                ("pricing_page_views", 5)],
    "job_change": [("champion_changed_job", 0.5), ("new_execs_90d", 1.5)],
    "negative": [("negative_news", 0.5), ("layoff_pct", 8),
                  ("bad_reviews", 3)],
    "trigger": [("trigger_events_30d", 3), ("recent_news_events", 2)],
}
SIGNAL_KEYS = tuple(RAW_FIELDS)


def _soft_num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _scale(x, mid):
    z = max(-60.0, min(60.0, (x - mid) / (mid * 0.4 + 1e-9)))
    return 1.0 / (1.0 + math.exp(-z))


def normalize_row(raw):
    if not isinstance(raw, dict):
        raise ValueError("raw must be an object")
    return {
        ch: max(_scale(_soft_num(raw.get(key, 0)), mid)
                for key, mid in fields)
        for ch, fields in RAW_FIELDS.items()
    }


def from_request(req):
    if not isinstance(req, dict):
        raise ValueError("request body must be an object")
    if "signals" in req:
        sig = req["signals"]
        if not isinstance(sig, dict):
            raise ValueError("signals must be an object")
        return {k: max(0.0, min(1.0, _soft_num(sig.get(k, 0))))
                for k in SIGNAL_KEYS}
    return normalize_row(req.get("raw", req))
