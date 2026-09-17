"""Channel -> real sensory population map, and the readout tracked set.

Every pick is an engineered input port, not a claimed biological
correspondence (see ARCHITECTURE.md). Populations are chosen by
cell_type prefix / superclass against the loaded brain, with explicit
fallbacks so a missing annotation degrades loudly, not silently.

Census (MaleCNS v1.0, this machine):
  ORN* 2,635 · JO* 672 · GRN 60 · LC4 326 · LPLC2 185 · LC10* 964
  KC* 4,064 · PPL1 16 · PAM 316 · AN 1,954 · DN 1,354
  superclasses: descending_neuron 1,314 · ascending_neuron 1,846
  cb_sensory 4,868 · vnc_sensory 6,370 · sensory_ascending 537 · ENS 50
"""
import numpy as np

POP_SIZE = 64          # cells per input channel (redundant drive > 1 cell)
KC_SAMPLE = 512        # Kenyon-cell feature subset


def _idx_by_prefix(brain, prefix):
    ct = brain.cell_type
    return np.flatnonzero(np.char.startswith(ct.astype(str), prefix))


def _idx_by_superclass(brain, sc):
    if brain.superclass is None:
        return np.empty(0, np.int64)
    return np.flatnonzero(brain.superclass == sc)


def _take(candidates, k, used, rng, fallback=None):
    """k unused cells from candidates, topped up from `fallback` when the
    primary pool is smaller than k (ENS has 50 cells — the gut is small).
    Never reuses a cell already claimed by another channel."""
    cand = np.setdiff1d(candidates, np.fromiter(used, np.int64)) \
        if used else candidates
    pick = list(rng.choice(cand, size=min(k, len(cand)),
                           replace=False)) if len(cand) else []
    if len(pick) < k and fallback is not None:
        fb = np.setdiff1d(fallback, np.fromiter(used, np.int64)) \
            if used else fallback
        fb = np.setdiff1d(fb, np.asarray(pick))
        pick += list(rng.choice(
            fb, size=min(k - len(pick), len(fb)), replace=False))
    if not pick:
        raise RuntimeError("population census returned zero neurons")
    used.update(int(i) for i in pick)
    return np.asarray(pick, dtype=np.int64)


def build_channel_map(brain, pop_size=POP_SIZE, seed=7):
    """16 observation channels -> disjoint neuron index arrays.

    Returns dict {channel: idx_array}. Raises if any channel is empty —
    an empty channel silently kills that sense, so fail loudly.
    """
    rng = np.random.default_rng(seed)
    used = set()

    orn = _idx_by_prefix(brain, "ORN")
    jo = _idx_by_prefix(brain, "JO")
    grn = _idx_by_prefix(brain, "GRN")
    lc4 = _idx_by_prefix(brain, "LC4")
    lplc2 = _idx_by_prefix(brain, "LPLC2")
    lc10 = _idx_by_prefix(brain, "LC10")
    ens = _idx_by_superclass(brain, "ENS")
    asc = _idx_by_superclass(brain, "ascending_neuron")
    cbs = _idx_by_superclass(brain, "cb_sensory")
    vncs = _idx_by_superclass(brain, "vnc_sensory")
    sasc = _idx_by_superclass(brain, "sensory_ascending")

    # fallback pool for any modality that resolves empty
    def need(primary, fallback):
        return primary if len(primary) else fallback

    ch = {}
    # --- company signals -> distinct sensory modalities ---
    ch[0] = _take(need(orn, cbs), pop_size, used, rng)          # funding → smell
    ch[1] = _take(need(jo, vncs), pop_size, used, rng)          # hiring → hearing
    ch[2] = _take(need(lc4, cbs), pop_size, used, rng)          # intent → looming vision
    ch[3] = _take(need(lc10, cbs), pop_size, used, rng)         # job_change → vision
    ch[4] = _take(need(grn, cbs), pop_size, used, rng)          # negative → taste
    ch[5] = _take(need(lplc2, cbs), pop_size, used, rng)        # trigger → looming
    # --- internal/economic state -> ascending + enteric ---
    ch[6] = _take(need(ens, asc), pop_size, used, rng,
                  fallback=cbs)                                 # budget_frac → the gut
    ch[7] = _take(asc, pop_size, used, rng)                     # day_frac → ascending clock
    ch[8] = _take(cbs, pop_size, used, rng)                     # researched flag
    ch[9] = _take(cbs, pop_size, used, rng)                     # enriched flag
    # --- prices -> body/VNC sensing (nociception-adjacent) ---
    ch[10] = _take(vncs, pop_size, used, rng)                   # research price
    ch[11] = _take(vncs, pop_size, used, rng)                   # enrich price
    ch[12] = _take(vncs, pop_size, used, rng)                   # email price
    ch[13] = _take(vncs, pop_size, used, rng)                   # escalate price
    ch[14] = _take(asc, pop_size, used, rng)                    # budget level
    ch[15] = _take(need(sasc, asc), pop_size, used, rng)        # revealed pain
    return ch


def tracked_set(brain, channel_map, kc_sample=KC_SAMPLE, seed=7):
    """Neuron populations whose spike traces feed the policy readout.

    descending (motor command, 1,314) + ascending (body->brain state,
    1,846) + a Kenyon-cell sample (the fly's own learning center) +
    dopaminergic clusters (PPL1 aversive, PAM appetitive) + the injected
    channel populations themselves (input echo) + named motor groups.
    """
    rng = np.random.default_rng(seed)
    dn = _idx_by_superclass(brain, "descending_neuron")
    an = _idx_by_superclass(brain, "ascending_neuron")
    kc = np.flatnonzero(np.char.startswith(
        brain.cell_type.astype(str), "KC"))
    if len(kc) > kc_sample:
        kc = rng.choice(kc, kc_sample, replace=False)
    dans = np.concatenate([_idx_by_prefix(brain, "PPL1"),
                           _idx_by_prefix(brain, "PAM")])
    motors = np.concatenate(list(brain.groups.values())) \
        if brain.groups else np.empty(0, np.int64)
    chans = np.concatenate(list(channel_map.values()))
    tracked = np.unique(np.concatenate([dn, an, kc, dans, motors, chans]))
    return tracked.astype(np.int64)


def dan_populations(brain):
    """Dopamine populations for the reward visualization / pulses.
    PPL1 = aversive reinforcement cluster, PAM = appetitive."""
    return {
        "punish": _idx_by_prefix(brain, "PPL1"),
        "reward": _idx_by_prefix(brain, "PAM"),
    }


def action_pools(brain):
    """Named single-cell command neurons for storytelling + pooled-decode
    ablation. Each action gets the L/R pair of a real motor group."""
    g = brain.groups
    return {
        "OBSERVE": np.concatenate([g.get("steer_L", []), g.get("steer_R", [])]),
        "RESEARCH": np.concatenate([g.get("kick_L", []), g.get("kick_R", [])]),
        "ENRICH": np.concatenate([g.get("punch_L", []), g.get("punch_R", [])]),
        "EMAIL": np.concatenate([g.get("forward_L", []), g.get("forward_R", [])]),
        "ESCALATE": np.concatenate([g.get("escape_L", []), g.get("escape_R", [])]),
        "IGNORE": np.concatenate([g.get("backward_L", []), g.get("backward_R", [])]),
    }
