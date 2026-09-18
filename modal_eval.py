"""Paired multi-seed eval on B200s: one remote job per (arm, economy).

  modal run modal_eval.py            # all arms × both economies × 6 seeds
  modal run modal_eval.py --econ v2  # only the triage economy
  modal run modal_eval.py --arms fly_v2,mlp_v2
  modal run modal_eval.py --arms kp6=results_kp6/fly_policy.pt
  modal run modal_eval.py --arms 'x=results_x/p.pt@/data/fly-data-shuffled'

v1 economy = the original balanced world (no rent).
v2 economy = triage: rent + kill_bonus + expire_all.
Every (arm, economy) cell runs the same held-out seeds so the deltas are
paired. Rows aggregate into eval_matrix.json / eval_summary.json locally.
"""
import json
import modal

img = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04",
        add_python="3.12",
    )
    .pip_install("flybrain[gpu]", "torch", "numpy", "scipy")
    .add_local_python_source("brain", "environment", "learning",
                             "experiments", "serving")
)
vol = modal.Volume.from_name("clayfly-data", create_if_missing=True)
app = modal.App("clayfly-eval")

V1 = {"rent": 0.0, "kill_bonus": 0.0, "expire_all": False}
V2 = {"rent": 0.05, "kill_bonus": 3.0, "expire_all": True}

ARMS = {
    "fly_v1":      ("results/fly_policy.pt", None),
    "fly_noraw":   ("results_noraw/fly_policy.pt", None),
    "fly_shuffled": ("results_shuffled/fly_policy.pt",
                     "/data/fly-data-shuffled"),
    "mlp_v1":      ("results_mlp/mlp_policy.pt", None),
    "fly_v2":      ("results_v2/fly_policy.pt", None),
    "fly_noraw_v2": ("results_noraw_v2/fly_policy.pt", None),
    "fly_shuffled_v2": ("results_shuffled_v2/fly_policy.pt",
                        "/data/fly-data-shuffled"),
    "mlp_v2":      ("results_mlp_v2/mlp_policy.pt", None),
}


def _parse_arm(spec: str):
    """'name' -> (name, ARMS[name]); 'name=ckpt[@braindir]' -> ad-hoc arm.

    Returns (name, (ckpt_rel, brain_data)) or (name, None) if unresolvable.
    """
    if "=" not in spec:
        return spec, ARMS.get(spec)
    name, _, rest = spec.partition("=")
    ckpt, _, brain = rest.partition("@")
    return name, (ckpt, brain or None) if name and ckpt else None


# per-seed remote job: a 160-account × 45-day episode on the B200.
# Cheap and embarrassingly parallel — the eval is inference, not training.


@app.function(image=img, gpu="B200", volumes={"/data": vol}, timeout=3600)
def eval_cell(arm: str, econ: str, seed: int, ckpt_rel: str,
              brain_data: str | None, n: int = 160, days: int = 45):
    import os
    import numpy as np
    os.environ["FLY_DATA"] = "/data/fly-data"
    world_kwargs = V1 if econ == "v1" else V2

    from brain.loader import get_brain
    from brain.populations import build_channel_map, tracked_set
    from learning.encoder import Encoder
    from learning.policy import FlyPolicy, MLPPolicy
    from learning.rollout import evaluate
    from learning.trace import BatchTrace
    import torch

    brain = get_brain(batch=n, device="cuda", data=brain_data)
    ch_map = build_channel_map(brain)
    tracked = tracked_set(brain, ch_map)
    trace = BatchTrace(brain, tracked, tau=(0.05, 0.15, 0.5))

    ck = torch.load(f"/data/{ckpt_rel}", map_location="cpu",
                    weights_only=False)
    state = ck["state"]
    n_feat = ck.get("n_feat", state["readout.weight"].shape[1]
                    if "readout.weight" in state else 12012)
    if "net.0.weight" in state:
        policy = MLPPolicy(16, int(state["net.0.weight"].shape[0]))
    else:
        policy = FlyPolicy(n_feat, use_raw_head=ck.get("use_raw_head", True))
    policy.load_state_dict(state)
    policy.eval().to("cuda")

    enc = Encoder(ch_map,
                  gains=np.asarray(ck["encoder_gains"], np.float32)
                  if ck.get("encoder_gains") is not None else None)
    val, stats = evaluate(brain, trace, enc, policy, "cuda",
                          seed=seed, n=n, days=days,
                          world_kwargs=world_kwargs)
    return {"arm": arm, "econ": econ, "seed": seed,
            "return": float(val),
            **{k: (float(v) if isinstance(v, np.floating)
                   else int(v) if isinstance(v, np.integer) else v)
               for k, v in stats.items()}}


@app.local_entrypoint()
def main(econ: str = "both", seeds: str = "777,779,787,797,809,811",
         n: int = 160, days: int = 45, arms: str = ""):
    seed_list = [int(s) for s in seeds.split(",")]
    econs = ["v1", "v2"] if econ == "both" else [econ]
    arm_list = [a for a in arms.split(",") if a] or list(ARMS)
    resolved = {}  # name -> (ckpt_rel, brain_data)
    for a in arm_list:
        name, spec = _parse_arm(a)
        resolved[name or a] = spec
    missing = [k for k, v in resolved.items() if v is None]
    if missing:
        raise SystemExit(f"unknown arms: {missing}; known: {list(ARMS)}; "
                         "or name=ckpt_rel[@brain_data]")

    cells = [(name, e, s, spec[0], spec[1])
             for name, spec in resolved.items()
             for e in econs for s in seed_list]
    print(f"eval cells: {len(cells)} "
          f"({len(resolved)} arms × {len(econs)} econ × {len(seed_list)} seeds)")
    rows = list(eval_cell.map(*zip(*cells), kwargs={"n": n, "days": days}))
    print(json.dumps(rows, indent=2))
    with open("results/eval_remote.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote results/eval_remote.json ({len(rows)} rows)")
