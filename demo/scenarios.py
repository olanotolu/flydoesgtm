"""Deterministic 30-second demo replay — every displayed number is
explicitly demo/simulated unless loaded from a recorded metrics file."""
import json
from pathlib import Path

SCENES = [
    {"at": 0.0, "kind": "title", "text": "they mapped a fruit fly's brain."},
    {"at": 2.0, "kind": "title", "text": "so i gave it a clay account."},
    {"at": 4.0, "kind": "account", "company": "Stealth", "employees": 2,
     "funding": "unknown", "signal": "changed LinkedIn headline",
     "action": "OBSERVE"},
    {"at": 7.5, "kind": "action", "action": "RESEARCH",
     "research": ["$8M raised yesterday", "ex-OpenAI founder",
                  "hiring first GTM team"]},
    {"at": 11.5, "kind": "action", "action": "EMAIL"},
    {"at": 14.0, "kind": "outcome", "outcome": "MEETING BOOKED", "reward": 20},
    {"at": 17.0, "kind": "account", "company": "DefinitelyRealAI",
     "employees": 1, "funding": "$400M claimed", "signal": "carrd.co",
     "founder": "CEO / visionary / thought leader", "action": "THINKING"},
    {"at": 21.0, "kind": "action", "action": "IGNORE"},
    {"at": 22.2, "kind": "title", "text": "THE FLY HAS LEARNED QUALIFICATION."},
    {"at": 25.0, "kind": "stats", "label": "demo / simulated",
     "stats": {"accounts": 847, "enriched": 126, "contacted": 31,
               "meetings": 4, "spend": "$18.42"}},
    {"at": 29.0, "kind": "title", "text": "166,700 neurons. zero SDRs."},
]


def write(path="demo/replay.json"):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"duration": 31, "scenes": SCENES}, indent=2))
    return p


if __name__ == "__main__":
    print(write())
