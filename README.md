# Fly Brain × Clay

> they mapped a fruit fly's brain, so i gave it a Clay account.
>
> **166,700 neurons. zero SDRs.**

This is a technically serious meme: a real MaleCNS v1.0 fruit-fly
connectome receives GTM/account signals and chooses whether more
information is worth paying for.

## The honest version

The fly's biology is **frozen**. We use `flybrain` — the MIT-licensed
Python implementation of the MaleCNS v1.0 network — for its 166,700
leaky-integrate-and-fire neurons and 25,582,938 signed connections.
PPO trains only an artificial linear readout over three decaying spike
traces (50ms / 150ms / 500ms) from the fly's descending/ascending/
Kenyon-cell populations. GAE propagates delayed account outcomes through
account-local action histories, and a reward-weighted sensory interface
adapts channel injection gains. The connectome checksum is asserted
before and after training.

So the experiment is not "a fly learned sales." It is:

> Can a fixed biological circuit be a useful nonlinear substrate for a
> learned GTM decision layer under information costs?

The full biology/engineering boundary is in
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## What it does

An account arrives with noisy signals. The policy can:

```
IGNORE · RESEARCH · EMAIL → DRAFT_EMAIL · ESCALATE
```

`OBSERVE` is cheap but never removes uncertainty. `RESEARCH` is
expensive and reveals latent pain. `EMAIL` has delayed replies,
meetings, opportunities and closed-won outcomes — plus unsubscribe and
spam penalties. Some accounts intentionally inflate claimed funding.
The fly has to learn qualification, not maximize emails.

## Run it

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m serving.serve
open http://127.0.0.1:8090
```

The hosted replay is available at [clayfly.vercel.app](https://clayfly.vercel.app).
It serves the captured Clay proof without credentials or live enrichment; the
full live-draft path remains the local/server deployment described below.
To redeploy the lightweight hosted surface:

```bash
(cd deploy && vercel --prod)
```

The local server exposes a recorded proof path by default:

- **Live Clay mode** is an explicit toggle — a capped read-only search can
  produce provenance-complete, visibly unsent drafts, but never sends,
  enrolls, replies, updates a CRM, or mutates a campaign.
- `POST /api/demo/run` — `replay` or capped `live_draft` mode.
- `GET /api/demo/replay/default` — deterministic captured proof with no network dependency.
- `GET /api/health` and `GET /api/artifacts/current` — readiness and provenance.
- `POST /decide` — compatibility route for one normalized record.
- `GET /graph` — 24,000 sampled real MaleCNS soma positions, channels,
  and motor groups for the visualization.
- `GET /replay` — deterministic 31-second demo sequence.
The fixed live query targets US operations decision-makers at 11–500 employee
restaurant and food-service companies. It deduplicates by company, processes
at most ten records, permits at most four enrichments per record, and caps
the demo at 50 credits while preserving a 300-credit floor.

## Train on Modal

The training image uses NVIDIA CUDA 12.8 and the **B200** GPU. Modal's
batch dimension maps one fly lane to one account, so the full connectome
runs in parallel across a day of accounts.

```bash
modal token set
.venv/bin/modal run modal_train.py --probe-only
.venv/bin/modal run modal_train.py
.venv/bin/modal volume get clayfly-data results/fly_policy.pt results/
```

The earlier 1,256.1 Modal output is preserved as a dated legacy artifact and
is not final evidence. New B200 runs must use immutable experiment manifests,
paired held-out worlds, twenty seeds per arm, bootstrap confidence intervals,
and multiplicity correction before any architecture claim is made. Training
metrics are written by the run; no dashboard number should be copied into the
demo unless it came from a recorded run.

## Demo beat

1. `they mapped a fruit fly's brain.`
2. `so i gave it a clay account.`
3. Stealth: 2 employees, unknown funding → OBSERVE → RESEARCH.
4. Research returns sourced evidence; the policy re-scores.
5. The result is `UNSENT DRAFT`, `IGNORE`, or `ESCALATE` — never a contact.

The interface is dense and neutral rather than a generic purple AI
landing page. The connectome is the hero; Clay is the job.

A real held-out feature ablation on the downloaded checkpoint is saved in
`results/fly_ablation.json`: fly readout returned 98.1, half-feature lesion
returned 364.0, and shuffled features returned 0.0 on that small seed.
That is not evidence that more features are always better; it is evidence
that the representation is still under-regularized and that the shuffled
control destroys useful alignment. We report the surprising lesion result
instead of hiding it.

## Repository

```
brain/          frozen fly loader, population census, dopamine pulses
environment/    GTM POMDP, account cosmetics, channel specification
learning/       encoder, spike traces, readout policy, PPO, Hebbian arm
experiments/    baselines and ablations
serving/        stdlib server, Clay API adapter, live routine flag
 demo/          scripted replay and visual demo
tests/          environment, frozen-brain, policy invariants
modal_train.py  B200 training entrypoint
```

## Current limitations

- The first compact run is intentionally a smoke-quality training run;
  the learned policy is conservative and does not beat the handwritten
  rules yet. That failure is recorded, not hidden.
- `PPL1`/`PAM` stimulation is a real pulse into annotated DAN cells for
  the visualization. It is not a claim that the fly's biology learns;
  optional plasticity belongs only on our readout interface.
- Clay live mode is opt-in (`CLAYFLY_LIVE=1`, `CLAYFLY_LIVE_ROUTINES=1`,
  `CLAYFLY_ROUTINE_ID=...`) and falls back safely when credentials, the
  workflow routine, or the credit preflight are unavailable. It does not send
  email.
- The visual mapping from GTM channels to sensory populations is
  engineered. `ORN` does not biologically mean funding.

## Source and licensing

MaleCNS v1.0 is the Janelia/Google full male CNS connectome and is
licensed CC-BY. The `flybrain` runtime is MIT. See the authoritative
data/download page and the upstream runtime before redistributing brain
data files.
