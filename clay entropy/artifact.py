"""Exportable artifact packages and simple ASCII STL mockups."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

Facet = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


def _tri(facets: list[Facet], a, b, c) -> None:
    facets.append((tuple(a), tuple(b), tuple(c)))


def _add_box(facets: list[Facet], center, size, angle=0.0) -> None:
    cx, cy, cz = center
    sx, sy, sz = (v / 2 for v in size)
    corners = []
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    for x in (-sx, sx):
        for y in (-sy, sy):
            rx = x * cos_a - y * sin_a
            ry = x * sin_a + y * cos_a
            corners.append((cx + rx, cy + ry))
    bottom_z, top_z = cz - sz, cz + sz
    b = [(x, y, bottom_z) for x, y in corners]
    t = [(x, y, top_z) for x, y in corners]
    order = [0, 2, 3, 1]
    b = [b[i] for i in order]
    t = [t[i] for i in order]
    _tri(facets, b[0], b[2], b[1])
    _tri(facets, b[0], b[3], b[2])
    _tri(facets, t[0], t[1], t[2])
    _tri(facets, t[0], t[2], t[3])
    for i in range(4):
        j = (i + 1) % 4
        _tri(facets, b[i], b[j], t[j])
        _tri(facets, b[i], t[j], t[i])


def _add_cylinder(facets: list[Facet], center, radius, height, segments=16) -> None:
    cx, cy, cz = center
    z0, z1 = cz - height / 2, cz + height / 2
    bottom, top = [], []
    for i in range(segments):
        a = 2 * math.pi * i / segments
        bottom.append((cx + radius * math.cos(a), cy + radius * math.sin(a), z0))
        top.append((cx + radius * math.cos(a), cy + radius * math.sin(a), z1))
    for i in range(segments):
        j = (i + 1) % segments
        _tri(facets, (cx, cy, z0), bottom[j], bottom[i])
        _tri(facets, (cx, cy, z1), top[i], top[j])
        _tri(facets, bottom[i], bottom[j], top[j])
        _tri(facets, bottom[i], top[j], top[i])


def _add_frustum(facets: list[Facet], center, bottom_radius, top_radius, height, sides=4) -> None:
    cx, cy, cz = center
    z0, z1 = cz - height / 2, cz + height / 2
    bottom, top = [], []
    for i in range(sides):
        a = 2 * math.pi * i / sides + math.pi / 4
        bottom.append((cx + bottom_radius * math.cos(a), cy + bottom_radius * math.sin(a), z0))
        top.append((cx + top_radius * math.cos(a), cy + top_radius * math.sin(a), z1))
    for i in range(1, sides - 1):
        _tri(facets, bottom[0], bottom[i + 1], bottom[i])
        _tri(facets, top[0], top[i], top[i + 1])
    for i in range(sides):
        j = (i + 1) % sides
        _tri(facets, bottom[i], bottom[j], top[j])
        _tri(facets, bottom[i], top[j], top[i])


def mesh_for(signal_type: str) -> list[Facet]:
    facets: list[Facet] = []
    if signal_type == "funding":
        _add_cylinder(facets, (0, 0, 4), 42, 8, 20)
        _add_cylinder(facets, (0, 0, 12), 31, 8, 20)
        _add_cylinder(facets, (0, 0, 22), 13, 12, 16)
    elif signal_type == "product_launch":
        _add_box(facets, (0, 0, 4), (82, 54, 8))
        _add_box(facets, (10, -3, 17), (62, 38, 6))
    elif signal_type == "acquisition":
        _add_box(facets, (-8, 0, 12), (76, 20, 18), -0.35)
        _add_box(facets, (8, 0, 12), (76, 20, 18), 0.35)
    elif signal_type == "expansion":
        _add_box(facets, (0, 0, 4), (82, 52, 8))
        _add_box(facets, (24, 0, 24), (12, 12, 40))
    elif signal_type == "customer_milestone":
        _add_box(facets, (-25, 0, 15), (14, 24, 30))
        _add_box(facets, (25, 0, 15), (14, 24, 30))
        _add_box(facets, (0, 0, 34), (70, 20, 8))
    else:
        _add_frustum(facets, (0, 0, 20), 38, 16, 40, 4)
    return facets


def write_stl(path: Path, name: str, signal_type: str) -> None:
    facets = mesh_for(signal_type)
    lines = [f"solid {name.replace(' ', '_')}"]
    for a, b, c in facets:
        lines.append("  facet normal 0 0 0")
        lines.append("    outer loop")
        for vertex in (a, b, c):
            lines.append(f"      vertex {vertex[0]:.3f} {vertex[1]:.3f} {vertex[2]:.3f}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {name.replace(' ', '_')}")
    path.write_text("\n".join(lines) + "\n")


def export_package(record: dict[str, Any], out_root: Path) -> Path:
    package = out_root / record["id"]
    package.mkdir(parents=True, exist_ok=True)
    decision = record["decision"]
    input_data = record["input"]
    concept = decision["artifact_concept"]

    write_stl(package / "artifact.stl", concept["name"], input_data.get("signal_type", "other"))
    (package / "decision.json").write_text(
        json.dumps({"input": input_data, "decision": decision}, indent=2, sort_keys=True)
    )
    (package / "provenance_card.md").write_text(
        "# Provenance card\n\n"
        f"{concept['provenance_card_draft']}\n\n"
        "## Human review\n\n"
        + "\n".join(f"- {item}" for item in decision["human_review_checklist"])
        + "\n"
    )
    (package / "manifest.json").write_text(
        json.dumps(
            {
                "record_id": record["id"],
                "account_name": input_data.get("account_name"),
                "artifact_name": concept["name"],
                "recommended_action": decision["recommended_action"],
                "files": ["artifact.stl", "decision.json", "provenance_card.md"],
                "boundary": "Concept/export package only. No autonomous contact or fulfillment.",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return package
