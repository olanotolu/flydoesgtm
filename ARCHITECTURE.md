# Architecture — what's the fly, what's us

**the fly got a job at clay**

This document exists so that nobody has to guess which parts are biology
and which parts are engineered. Short version: the wiring is real, the
job skills are ours.

## The real part (MaleCNS v1.0, CC-BY, Janelia/Google)

- The network: 166,700 neurons and 25,582,938 connections, reconstructed
  by electron microscopy, loaded via the `flybrain` package
  (`alextitonis/fly.ai`, MIT). We never modify the connectome. A test
  asserts the weight checksum is identical before and after training.
- The dynamics: leaky integrate-and-fire over the real signed adjacency
  (synapse-count weights, neurotransmitter-inferred signs, per-neuron
  input normalization) — the same approximation every current
  fly-plays-X project uses. It is a wiring diagram with plausible
  dynamics, not a validated fly emulation.
- The populations we talk to: real annotated cell types — olfactory
  receptor neurons (`ORN_*`), Johnston's organ (`JO_*`), gustatory
  (`GRN*`), visual projection (`LC4`, `LPLC2`, `LC10*`), ascending and
  descending neurons, Kenyon cells (`KC*`, the fly's own learning
  center), and `PPL1`/`PAM` dopaminergic clusters.
- `sensory_input=False`: all synapses *onto* sensory neurons are removed
  (the published model needs this — otherwise olfactory receptors excite
  each other into a runaway). Our observation channels are literally the
  fly's sensory world.

## The engineered part

- **Encoder** (`learning/encoder.py`): maps each of 16 observation
  channels to a real sensory population and injects current. The mapping
  is an arbitrary engineering choice — `ORN` doesn't "mean" funding;
  it's a distinguished input port.
- **Environment** (`environment/clay_world.py`): a simulated GTM economy
  (ported from C. Claygans). Accounts carry hidden intent/fit/urgency/
  champion/competitor/pain; OBSERVE shrinks noise but never below
  σ=0.35; RESEARCH costs 40× more and reveals latent pain; EMAIL
  outcomes are delayed and probabilistic; budgets and prices are real.
  One deliberate adversarial feature: ~10% of low-intent accounts have
  inflated *claimed* funding — the channel can lie.
- **Readout** (`learning/policy.py`): a linear layer over three decaying
  spike traces (50ms / 150ms / 500ms) of descending/ascending/Kenyon/DAN
  populations → action logits + value. **This is the only thing PPO
  trains** (plus action bias/gain and price margins). The fly's brain is
  a frozen feature extractor; the "job skill" is a set of engineered
  synapses on top of it.
- **Sensory interface adaptation** (`learning/encoder.py`): after each
  rollout, reward-weighted channel covariance adjusts the 16 injection
  gains. This is a black-box interface update, not backpropagation into
  biology; predictive channels get more current and harmful channels get
  less.
- **Credit assignment** (`learning/ppo.py`, `learning/rollout.py`):
  delayed account return is attached to the account's final decision and
  propagated backward through that account's action history with GAE;
  done masks prevent one account's outcome leaking into another.
- **Teacher warm-start** (`environment/teacher.py`, `learning/pretrain.py`):
  an explicit threshold policy initializes the engineered readout, then a
  small imitation term remains during PPO so economic optimization cannot
  collapse immediately to zero-spend inaction. The teacher is reported as
  a separate baseline, never as biological behavior.
- **GPU path**: CuPy keeps spike traces on the connectome device. CUDA
  features cross into Torch through DLPack rather than a CPU round-trip.
- **Dopamine** (`brain/dopamine.py`): reward events pulse real PPL1/PAM
  cells — visible in the visualization. Optionally, a three-factor
  Hebbian rule (`Δw = η · RPE · eligibility`) updates *the readout
  synapses* — plasticity on our interface, not on the fly.
- **WAIT** is decoded from insufficient commitment — if no pool clears
  threshold, the fly does nothing. Very fly.

## What we are NOT claiming

- That a fly brain is "naturally good at sales." The connectome is a
  fixed, unusually-shaped nonlinear reservoir; the question is whether
  its structure is a useful substrate for a sequential economic
  decision problem.
- That dopamine stimulation rewires the fly. It doesn't — the viz pulse
  is real spikes in real DANs, and any plasticity is on our own
  interface synapses.
- That demo metrics are real unless they came from a recorded run;
  scripted-scenario numbers are labeled `demo/simulated`.
