# Clayfly: The GTM Brain — design spec

Date: 2026-09-17. Status: approved, pending implementation plan.
Thesis: we gave a fruit fly the same job every GTM engineer has — deciding
what deserves attention. Signal enters → the fly decides whether it deserves
intelligence → a GTM plan emerges. The fly is allowed to disagree with the
human, and it never sends email.

## What exists (measured, not assumed)

- No anatomical brain mesh. We have 24,000 real 3D soma positions + channel
  and pool membership via `/graph`, the full 25M-synapse MaleCNS weights
  (loads in 1.5 s), and a live sim: `decide()` runs in 0.56 s returning
  real `channel_activity`, `pools`, `probabilities`, `confidence`.
- Three.js r147 vendored inline; a procedural clay fly already on stage.
- Recorded replay rows + capped live Clay Search; no-send enforced;
  `demo/verify_ui.py` Playwright gate.

## D1 — Real neural data (approved)

RUN always runs the real connectome on the account's signals and displays
measured values. Nothing decorative. New endpoint:

- `POST /api/demo/think` with `{signals: {funding, hiring, intent,
  job_change, negative, trigger}}` → `{channel_activity, pools,
  probabilities, confidence, decision, crowdedness, sim_steps}`.
  Synchronous (0.56 s measured). Local compute only: no Clay credits, no
  budget, no external calls. Works identically in replay and live modes
  (replay rows get live sim, which is the honest story: same brain,
  recorded inputs).

## D2 — Signal-pack fixture + computed crowdedness (approved)

- Hero account ships a hand-built signal pack (Anthropic-style layered
  signals: funding event, enterprise/product expansion, hiring, ICP match,
  each with source + type + tags). No live news pipe exists; live mode
  keeps real Clay Search people/company data. Fixture is presentation,
  never a send path.
- Crowdedness is computed, not hardcoded: `crowdedness` = count of
  channels with energy > 0.6. Contrarian NO triggers when the policy
  would engage AND crowdedness ≥ 4, with reason lines naming the hot
  channels and a wait of `7 * (crowdedness - 2)` days. All numbers shown
  are measured (confidence %, crowded count, channel energies);
  qualitative lines only otherwise — no invented percentages.

## D3 — Thinking theater in the fly stage (approved)

RUN on the featured card launches a staged sequence reusing the hero:

1. Constellation dims; real 3D point-cloud brain fades in (positions from
   `/graph`), channels lighting by measured `channel_activity`.
2. Sensory bars render measured per-channel energies.
3. Reasoning lines derive from measurements: top-2 channels → "detected"
   lines; crowdedness + confidence → information value (HIGH/MEDIUM/LOW)
   and recommended action (RESEARCH/IGNORE).
4. Steps 1–3 populate the execution column from row data: Find ICPs
   (provenance/industry), Enrich (enrichment status/decision makers),
   Outreach decision (draft preview or WAIT reasoning). Never auto-email.
5. FLY DECISION card: verdict + confidence + why (top channels named) +
   motion (Research → buyers → artifact → outreach, or WAIT + days).

Manual scrub/stepping reuses the existing episode controller states;
thinking sequence is one pass per RUN (no autoplay coupling).

## Copy rules

- The fly's voice states measurements first, interpretation second.
- YES states reason + motion; NO states reason + wait (crowdedness path
  included). Both end in artifacts, never sends.
- Product name: Clayfly: The GTM Brain.

## Testing

- `pytest tests/ -q` stays green (add think-endpoint tests: shape,
  crowdedness math, no-credit/no-network).
- Extend `demo/verify_ui.py`: RUN reaches verdict; bars match returned
  energies; verdict names decision; crowded fixture yields WAIT wording.
- Desktop + mobile screenshots read by implementer; independent audit.

## Non-goals

Live news ingestion; meshed brain anatomy; policy/budget/cap changes;
Vercel live-path changes; touching the Three.js fly, the icon sprite, or
the static-file branch. One new serve.py route (`/api/demo/think`) is
expected and in scope.
