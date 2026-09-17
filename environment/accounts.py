"""Cosmetic account fields for the demo table.

The environment is numeric; the demo needs company names, domains,
employee counts, claimed funding and a signal blurb. All of it is
generated FROM the hidden state (+ noise) so the table tells the same
story the fly is actually seeing — including the liars.
"""
import numpy as np

ARCHETYPES = [
    # name fragments chosen to read well in a dense table
    ("Stealth", [".ai", ".dev", ".run", ".systems", " Labs", "HQ"]),
    ("Definitely", ["RealAI", "Scale", "Growth", "Vision", "Capital"]),
    ("North", ["beam", "loop", "stack", "forge", "base"]),
    ("Quick", ["ly", "base", "hire", "pipe", "shift"]),
    ("Vanta", ["core", "line", "point", "wise", "ful"]),
    ("Kudo", ["s", "works", "able", "kit", "io"]),
    ("Paper", ["trail", "plane", "route", "mint", "dash"]),
    ("Hyper", ["local", "flux", "mesh", "dash", "sync"]),
]

FOUNDER_BIOS_BAD = [
    "CEO / visionary / thought leader / Forbes 30u30 nominee",
    "building something huge",
    "ex-anthropic (adjacent)",
    "serial founder (3 exits, all acqui-hires)",
    "posting daily on linkedin",
]
FOUNDER_BIOS_GOOD = [
    "ex-OpenAI infra",
    "prev. scaled GTM to $40M ARR",
    "2nd-time founder, first exit $900M",
    "phd dropout, shipping weekly",
    "built the thing we all use",
]

SIGNALS = [
    "liked one linkedin post",
    "changed LinkedIn headline",
    "hiring {n} GTM roles",
    "pricing page views x{n}",
    "competitor removed from stack",
    "new VP Sales started",
    "website rewrite shipped",
    "attended 3 webinars",
    "quiet changelog update",
    "opened an office in austin",
]


def fmt_money(x):
    if x >= 1e9:
        return f"${x / 1e9:.1f}B"
    if x >= 1e6:
        return f"${x / 1e6:.0f}M"
    if x >= 1e3:
        return f"${x / 1e3:.0f}K"
    return "$0"


def account_fields(world, i, rng=None):
    """Cosmetic row for account i, derived from world hidden state.

    `inflate` shows up exactly where it should: in the *claimed* funding
    cell, not the truth.
    """
    rng = rng or np.random.default_rng(10_000 + i)
    a, b = ARCHETYPES[int(rng.integers(len(ARCHETYPES)))]
    name = a + b[int(rng.integers(len(b)))]
    domain = name.lower().replace(" ", "").replace("/", "") + (
        ".com" if rng.random() > 0.35 else rng.choice(
            [".ai", ".dev", ".site", ".co", "carrd.co", "notion.site"]))

    employees = int(np.clip(
        np.exp(rng.normal(np.log(4 + 60 * world.icp[i]), 0.9)), 1, 8000))

    real_funding = world.intent[i] * 4e7 * rng.uniform(0.3, 1.4)
    claimed = real_funding * (1 + 8 * float(world.inflate[i])) \
        + world.inflate[i] * 3e8
    funding_label = fmt_money(claimed) if claimed > 5e4 else "unknown"

    bio_pool = FOUNDER_BIOS_GOOD if world.intent[i] > 0.55 \
        else FOUNDER_BIOS_BAD
    bio = bio_pool[int(rng.integers(len(bio_pool)))]

    sig = SIGNALS[int(rng.integers(len(SIGNALS)))]
    sig = sig.replace("{n}", str(int(1 + 14 * world.urgency[i])))

    return {
        "company": name, "domain": domain, "employees": employees,
        "funding": funding_label, "signal": sig, "founder": bio,
    }
