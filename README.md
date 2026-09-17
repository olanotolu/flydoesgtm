# the fly got a job at clay

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
WAIT · OBSERVE · RESEARCH · ENRICH · EMAIL · ESCALATE · IGNORE
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

The local server exposes:

- **Live mode** (`http://127.0.0.1:8090`) — real Clay search auto-fires on load and
  streams companies into the table, the fly decides each one through the
  real connectome, stats show the live decision distribution, and
  clicking a row opens the decision provenance panel (sensory channels,
  P(action), named DN pool activity, credit cost) with a PRE/POST
  training comparison on the same company.
- `POST /decide` — raw Clay-style row or normalized signals → flat
  Clay-mappable action probabilities plus 14,700 multiscale activity
  features. Body `{"policy": "pre"}` diffs the no-warm-start checkpoint
  against the warm-started one.
- `GET /live?query=...&limit=N` — read-only Clay search → per-company fly
  decisions + distribution (server-side key only).
- `GET /graph` — 24,000 sampled real MaleCNS soma positions, channels,
  and motor groups for the visualization.
- `GET /replay` — deterministic 31-second demo sequence.
- `demo/clayfly.gif` — recorded from the live system
  (`demo/capture_gif.py`, also a console-error-checking UI verification).

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

The probe measured 166,700 neurons at batch 512 and ~220ms per spiking
step on B200. The upgraded multiscale+GAE run initially exposed a zero-spend basin.
After adding transparent teacher warm-starting plus a small learned
sensory-calibration head and imitation regularization, the B200 run
returned a best held-out value of **1,256.1**. The preceding fly-only
run returned -1,114, so the improvement is recorded as an architectural
change, not hidden. A two-seed B200 smoke sweep before the teacher pass
returned -182.0 and -105.8, so we do not present one lucky seed as proof. Training metrics are written by the run; no dashboard number should be copied into the
demo unless it came from a recorded run. Scripted scene numbers are
labeled `demo / simulated`.

## Demo beat

1. `they mapped a fruit fly's brain.`
2. `so i gave it a clay account.`
3. Stealth: 2 employees, unknown funding → OBSERVE → RESEARCH.
4. `$8M raised · ex-OpenAI · 14 GTM roles` → EMAIL → MEETING BOOKED.
5. DefinitelyRealAI: `$400M claimed`, `carrd.co` → long pause → IGNORE.
6. **THE FLY HAS LEARNED QUALIFICATION.**

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
- Clay live mode is opt-in (`CLAYFLY_LIVE=1`, `CLAYFLY_ROUTINE=...`) and
  never required for the simulation or demo. It does not send email.
- The visual mapping from GTM channels to sensory populations is
  engineered. `ORN` does not biologically mean funding.

## Source and licensing

MaleCNS v1.0 is the Janelia/Google full male CNS connectome and is
licensed CC-BY. The `flybrain` runtime is MIT. See the authoritative
data/download page and the upstream runtime before redistributing brain
data files.
