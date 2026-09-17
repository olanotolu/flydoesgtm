# Better-than-Ramp rebuild for Clay — design spec

Date: 2026-09-17. Status: approved, pending implementation plan.
Goal: a Clay GTM demo that uses Ramp fly-review's choreography (gate → guided
episode → verdict + evidence) but beats it where Ramp is fake: every panel is
wired to real replay/live data, real `decide()` outcomes, and real connectome
activity, with no-send honesty guarantees throughout.

## Background (evidence, not opinion)

- Ramp teardown (`labs.ramp.com/fly-review`): gated photoreal intro → single-record
  guided form → policy strip → auto-cycling review table (Play/Stop) → center
  fly + YES/NO keys + verdict + threshold sentence + per-check pass/fail rows →
  live neural side panels (firing neurons, spike counts, Hz scrubber) → honest
  footer. Premium feel comes from live data everywhere + choreographed states.
- Current page (`demo/web/index.html`, 735,917 B): other session's
  `#intro-screen` → `#review-screen` skeleton (firing-list, vision-strip,
  verdict placeholders) above the existing clay hero + queue + evidence
  workspace. Structure mirrors Ramp; panels have nothing behind them.
- Audit blockers: (1) 5 broken images — `../clay-logo.png` ×2 and
  `../fly-review-fly.png` ×3 return `index.html` as `text/html` because
  `serving/serve.py` has no static branch; (2) `demo/verify_ui.py` red —
  intro covers viewport indefinitely, `.row` never visible; (3) demo gated
  behind click/Enter with workspace hidden. Passes: pytest 36/36, HTML valid,
  CSP covers inline content, zero console errors, no mobile overflow.
- Data flow intact: replay → `demo/replay.json`; live → Clay search →
  normalize → `decide()` → gated routine → unsent draft; caps enforced.

## Architecture

1. Static assets: add a guarded static branch to `serving/serve.py` `do_GET`
   serving only `*.png` directly under `demo/` (reject any name containing
   `/`, return 404 otherwise) with `Content-Type: image/png`. Change
   `deploy/vercel.json` rewrite source to exclude `demo/*.png` so Vercel
   serves them as static files. Downscale in place with PIL LANCZOS:
   `demo/clay-logo.png` → 320 px wide (displayed ~76 px),
   `demo/fly-review-fly.png` → 1000 px wide (displayed ~500 px). Copy both
   to `deploy/demo/`. Everything else stays inline in the single file.
2. No new endpoints, no schema changes, no dependency changes, no changes
   to `brain/`, `learning/`, `environment/`.

## Components (all in `demo/web/index.html` unless noted)

1. Intro gate (keep, fix): asset refs work once static lands; click/Enter
   dismiss stays; no auto-dismiss. Gating is the choreography.
2. Episode engine (new JS unit): per-account states select → normalize →
   decide → checks → verdict → next. Autoplay ON after gate dismiss in
   replay mode (3 s per step, 4 s verdict hold, mirroring Ramp's Play/Stop);
   any manual row click stops autoplay (Play button resumes). Reuses
   `state.selected`, `selectRow`, `updateHero` — one source of truth for
   hero + review-screen. Scrubber index = step within the current account's
   derived timeline steps; scrubbing pauses autoplay.
3. Neural panels (wired): firing list = top-8 channels by trace signal
   energy with pool counts from the loaded graph; vision strip = 6 blocks
   from `SIGNAL_KEYS` energies; verdict sentence = `The fly says
   {decision} — confidence {pct}, {above|below} threshold 0.50`; per-check
   rows derived from row data: Signals normalized (provenance[0].source
   present), Research budgeted (policy_action/research or enrichment
   present), Draft unsent (decision DRAFT_EMAIL with draft attached),
   Provenance attached (provenance non-empty).
4. Hero: behavior unchanged; shares selection state.
5. Verifier (`demo/verify_ui.py`): press Enter to dismiss gate first, then
   existing asserts plus: autoplay advances hero-account to row 2,
   verdict names the decision, firing list non-empty.

## Data flow

Replay/live rows → episode controller → `selectRow`/`updateHero` → hero,
review-screen, and neural panels render from the same row + graph data.
Only new network fetches are the two static PNGs; the graph fetch is
unchanged.

## Error handling

PNG 404 → `onerror` hides the `img` (layout holds on alt box); graph
fetch fail → idle panels (existing pattern); live fail → safe empty state
(existing); autoplay pauses on any error and never spins; gate always
dismissible (existing handlers untouched).

## Testing

`pytest tests/ -q` stays green (add a static-serving test: PNG → 200 +
`image/png`; traversal path → 404); extended `verify_ui.py` as above;
implementer reads desktop + mobile screenshots; independent runtime audit
after; `deploy/` mirror + `vercel.json` ship in the same change.

## Non-goals

Hosted Vercel path stays recorded-proof (live stub untouched); no copy or
brand changes beyond wiring; no performance work beyond the PNG downscale.
