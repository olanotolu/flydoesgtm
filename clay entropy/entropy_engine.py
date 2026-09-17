"""Deterministic Clay Entropy signal gate and relic concept generator."""
from __future__ import annotations

import re
from typing import Any

MEANINGFUL_TYPES = {
    "funding",
    "product_launch",
    "acquisition",
    "expansion",
    "customer_milestone",
}

MILESTONE_WORDS = re.compile(
    r"(raised|funding|series [a-z]|launch|launched|acqui|opened|opening|"
    r"milestone|anniversary|expansion|expanded|award|customer)",
    re.I,
)
SENSITIVE_WORDS = re.compile(
    r"(death|died|funeral|grief|illness|cancer|health|pregnan|divorce|"
    r"family|child|religion|politic|protest|personal crisis|protected|medical)",
    re.I,
)
WEAK_WORDS = re.compile(
    r"(liked|commented|headline|profile|repost|followed|viewed|vague|"
    r"rumor|unconfirmed)",
    re.I,
)
TEMPLATE_WORDS = re.compile(
    r"(same artifact|repeat|template|generic trophy|rocket again|already sent)",
    re.I,
)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _urls(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_text(item) for item in value if _text(item)]
    return [part.strip() for part in re.split(r"[\n,]+", _text(value)) if part.strip()]


def _valid_url(url: str) -> bool:
    return bool(re.match(r"^https?://.+", url, re.I))


def _context_level(signal_text: str, known_context: str, source_count: int) -> str:
    total = len(signal_text) + len(known_context)
    if total >= 180 and source_count:
        return "high"
    if total >= 80:
        return "medium"
    return "low"


def _sensitivity(input_data: dict[str, Any]) -> str:
    haystack = " ".join(
        _text(input_data.get(key))
        for key in ("signal_text", "known_context", "sensitivity_notes")
    )
    if SENSITIVE_WORDS.search(haystack) or _text(input_data.get("sensitivity_notes")):
        return "high"
    return "low"


def _novelty(input_data: dict[str, Any], context: str) -> str:
    haystack = " ".join(
        _text(input_data.get(key)) for key in ("signal_text", "known_context")
    )
    if TEMPLATE_WORDS.search(haystack):
        return "high"
    if input_data.get("signal_type") == "other" or context == "low":
        return "medium"
    return "low"


def _concept(input_data: dict[str, str]) -> dict[str, str]:
    account = input_data.get("account_name") or "the company"
    observed = input_data.get("observed_at") or "the announced date"
    source = (input_data.get("source_urls") or ["the cited public source"])[0]
    concepts = {
        "funding": {
            "name": "Threshold Contour",
            "physical_description": "A compact metal topographic marker with one new contour ring added around the company’s current chapter.",
            "symbolic_mapping": "The added contour represents new capacity after a public funding milestone; it avoids literal money imagery.",
            "provenance_card_draft": f"{account} publicly announced a funding milestone on {observed}. This object marks the moment as a new contour in the company’s terrain, based on {source}.",
            "materials_notes": "Small machined brass or recycled aluminum; matte finish; no logo-forward treatment.",
        },
        "product_launch": {
            "name": "Operating Layer",
            "physical_description": "A palm-sized stack of two precisely aligned planes, with the top plane offset to reveal a newly active layer.",
            "symbolic_mapping": "The offset layer represents a product moving from private work into public use.",
            "provenance_card_draft": f"{account} publicly launched a new product chapter on {observed}. This object marks the shift from hidden layer to working layer, based on {source}.",
            "materials_notes": "Recycled acrylic or anodized aluminum; restrained two-material construction.",
        },
        "acquisition": {
            "name": "Convergence Study",
            "physical_description": "Two distinct material forms that meet in a single stable interlocking structure.",
            "symbolic_mapping": "The joined forms represent two organizations becoming one operating shape without erasing either side.",
            "provenance_card_draft": f"{account} publicly announced an acquisition on {observed}. This object marks the convergence of two operating paths, based on {source}.",
            "materials_notes": "Two contrasting but compatible materials; no predatory or winner/loser symbolism.",
        },
        "expansion": {
            "name": "Coordinate Marker",
            "physical_description": "A small architectural plinth with a single coordinate notch marking the new public location.",
            "symbolic_mapping": "The coordinate notch represents expansion as a real place, not just a growth metric.",
            "provenance_card_draft": f"{account} publicly announced an expansion on {observed}. This object marks the new coordinate in the company’s map, based on {source}.",
            "materials_notes": "Stone, ceramic, or recycled composite; modest footprint and low waste.",
        },
        "customer_milestone": {
            "name": "Load Path",
            "physical_description": "A minimal bridge form where a visible load travels cleanly from one support to another.",
            "symbolic_mapping": "The load path represents customer value moving through a system that now carries more responsibility.",
            "provenance_card_draft": f"{account} publicly announced a customer milestone on {observed}. This object marks the new load the system is carrying, based on {source}.",
            "materials_notes": "Bent recycled metal or ceramic; stable, desk-safe, and not customer-specific.",
        },
        "other": {
            "name": "Moment Specimen",
            "physical_description": "A small abstract object that records one material transition without literal company imagery.",
            "symbolic_mapping": "The form represents a before/after shift that can be explained from the public signal.",
            "provenance_card_draft": f"{account} shared a public business moment on {observed}. This object preserves the transition described in {source}.",
            "materials_notes": "Recycled material preferred; final form requires human review.",
        },
    }
    return concepts.get(input_data.get("signal_type"), concepts["other"])


def _empty_concept(reason: str) -> dict[str, str]:
    return {
        "name": "No relic proposed",
        "physical_description": "No physical artifact should be generated from this signal in its current form.",
        "symbolic_mapping": reason,
        "provenance_card_draft": "No provenance card drafted because the signal did not clear the review gate.",
        "materials_notes": "None.",
    }


def normalize_input(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "account_name": _text(raw.get("account_name")),
        "signal_type": _text(raw.get("signal_type")) or "other",
        "signal_text": _text(raw.get("signal_text")),
        "source_urls": _urls(raw.get("source_urls")),
        "observed_at": _text(raw.get("observed_at")),
        "known_context": _text(raw.get("known_context")),
        "relationship_context": _text(raw.get("relationship_context")) or "none",
        "recipient_role": _text(raw.get("recipient_role")),
        "sensitivity_notes": _text(raw.get("sensitivity_notes")),
    }


def evaluate(raw: dict[str, Any]) -> dict[str, Any]:
    input_data = normalize_input(raw)
    meaningful = (
        input_data["signal_type"] in MEANINGFUL_TYPES
        or bool(MILESTONE_WORDS.search(input_data["signal_text"]))
    )
    public = bool(input_data["source_urls"]) and all(
        _valid_url(url) for url in input_data["source_urls"]
    )
    context = _context_level(
        input_data["signal_text"], input_data["known_context"], len(input_data["source_urls"])
    )
    sensitivity = _sensitivity(input_data)
    distance = "medium" if input_data["relationship_context"] == "none" else "high"
    novelty = _novelty(input_data, context)

    failed = []
    if not meaningful:
        failed.append("meaningful_event")
    if not public:
        failed.append("public_and_attributable")
    if context == "low":
        failed.append("context_sufficiency")
    if sensitivity == "high":
        failed.append("low_sensitivity")
    if novelty == "high":
        failed.append("novelty_preservation")

    if not meaningful or not public or context == "low" or sensitivity == "high":
        action = "wait"
        reason = (
            "The signal does not clear the minimum gate for a physical gesture. "
            "Waiting preserves trust and avoids manufacturing meaning."
        )
    elif novelty == "high" or context == "medium" or WEAK_WORDS.search(input_data["signal_text"]):
        action = "digital"
        reason = (
            "The event may merit acknowledgment, but the artifact path has novelty, "
            "context, or interpretation risk. A concise human-reviewed digital note is safer."
        )
    else:
        action = "relic"
        reason = (
            "The public milestone, context, recipient distance, and novelty checks "
            "support a non-promotional symbolic artifact for human review."
        )

    uncertainty = "; ".join(
        [
            f"context confidence is {context}",
            "the source is attributable" if public else "the source is missing or invalid",
            "no sensitive signal was detected" if sensitivity == "low" else "sensitive context is present",
            "no obvious template repetition was detected" if novelty == "low" else f"novelty risk is {novelty}",
        ]
    ) + "."

    risks = []
    if sensitivity == "high":
        risks.append("sensitive_or_private_context")
    if novelty != "low":
        risks.append("template_or_repetition_risk")
    if context != "high":
        risks.append("insufficient_context")
    if not public:
        risks.append("missing_public_source")
    if WEAK_WORDS.search(input_data["signal_text"]):
        risks.append("weak_or_ambiguous_signal")
    if re.search(r"security|procurement", input_data["recipient_role"], re.I):
        risks.append("workplace_delivery_policy_risk")
    if action == "relic":
        risks.append("requires_human_review_before_send")

    artifact = _concept(input_data) if action == "relic" else _empty_concept(reason)
    economics = (
        {
            "estimated_cost_band": "$180–$420 concept cost before fulfillment review",
            "estimated_lead_time_band": "7–14 days",
            "waste_or_delivery_concerns": "Use a small durable object, recycled material where possible, and verify workplace delivery rules.",
        }
        if action == "relic"
        else {
            "estimated_cost_band": "$0 physical production",
            "estimated_lead_time_band": "Same day after human review" if action == "digital" else "Not applicable",
            "waste_or_delivery_concerns": "No physical delivery should occur.",
        }
    )

    return {
        "signal_assessment": {
            "meaningful_event": meaningful,
            "public_and_attributable": public,
            "context_sufficiency": context,
            "sensitivity_risk": sensitivity,
            "appropriate_distance": distance,
            "novelty_risk": novelty,
            "failed_checks": failed,
        },
        "interpretation": {
            "what_happened": input_data["signal_text"] or "No signal text supplied.",
            "why_it_matters": (
                f"{input_data['account_name'] or 'The account'} has a public business transition that may carry meaning beyond routine engagement."
                if meaningful
                else "The supplied signal does not establish a material business transition."
            ),
            "uncertainty": uncertainty,
        },
        "artifact_concept": artifact,
        "risk_flags": risks,
        "economics": economics,
        "recommended_action": action,
        "recommendation_reason": reason,
        "human_review_checklist": [
            "Verify the public source and observed date.",
            "Confirm the interpretation uses only public business context.",
            "Check recipient workplace gift and delivery policies.",
            "Review the artifact for invasive, literal, trivializing, or promotional meaning.",
            "Confirm no pitch, QR code, booking link, tracking, or hidden CTA is included.",
            "Confirm this is not a repeated template before approving any real-world send.",
        ],
    }


SAMPLES = [
    {
        "id": "funding",
        "label": "Funding announcement",
        "input": {
            "account_name": "Contour Grid",
            "signal_type": "funding",
            "signal_text": "Contour Grid publicly announced a $15M Series A to expand its climate-data platform and hire its first commercial team.",
            "source_urls": ["https://example.com/contour-grid-series-a"],
            "observed_at": "2026-09-16",
            "known_context": "The company has spent two years building public-sector climate infrastructure and says the round funds the first dedicated customer team.",
            "relationship_context": "prior_contact",
            "recipient_role": "Founder",
            "sensitivity_notes": "",
        },
    },
    {
        "id": "launch",
        "label": "Product launch",
        "input": {
            "account_name": "Northbeam Systems",
            "signal_type": "product_launch",
            "signal_text": "Northbeam Systems publicly launched a real-time inventory layer for regional grocery chains after a six-month pilot.",
            "source_urls": ["https://example.com/northbeam-launch"],
            "observed_at": "2026-09-16",
            "known_context": "The launch moves the company from pilot software into an operational layer used daily by store teams.",
            "relationship_context": "none",
            "recipient_role": "CEO",
            "sensitivity_notes": "",
        },
    },
    {
        "id": "acquisition",
        "label": "Acquisition",
        "input": {
            "account_name": "Papertrail Labs",
            "signal_type": "acquisition",
            "signal_text": "Papertrail Labs announced it acquired a small route-planning company to combine dispatch records with field operations.",
            "source_urls": ["https://example.com/papertrail-acquisition"],
            "observed_at": "2026-09-16",
            "known_context": "The two products have shared customers, and the public announcement frames the deal as combining two operating workflows.",
            "relationship_context": "customer",
            "recipient_role": "COO",
            "sensitivity_notes": "",
        },
    },
    {
        "id": "expansion",
        "label": "Expansion",
        "input": {
            "account_name": "Quickbase Forge",
            "signal_type": "expansion",
            "signal_text": "Quickbase Forge publicly opened a second production office in Austin to support customers in the central United States.",
            "source_urls": ["https://example.com/quickbase-austin"],
            "observed_at": "2026-09-16",
            "known_context": "The announcement names the location, the customer-support reason, and the team responsible for the expansion.",
            "relationship_context": "prior_contact",
            "recipient_role": "VP Operations",
            "sensitivity_notes": "",
        },
    },
    {
        "id": "weak",
        "label": "Weak signal",
        "input": {
            "account_name": "Definitely Real AI",
            "signal_type": "other",
            "signal_text": "A founder liked a post and changed a LinkedIn headline.",
            "source_urls": ["https://example.com/social-activity"],
            "observed_at": "2026-09-16",
            "known_context": "No material business event is described.",
            "relationship_context": "none",
            "recipient_role": "Founder",
            "sensitivity_notes": "Possible family-health context mentioned in comments.",
        },
    },
]
