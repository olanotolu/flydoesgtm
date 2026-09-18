"""Modal GPU training for the fly.

    modal run modal_train.py                 # full curriculum on B200
    modal run modal_train.py --mlp           # dense baseline, same worlds
    modal volume get clayfly-data results/ ./results/

The brain batch == account count: one sparse matmul on CUDA steps all
flies at once, so a 500-account day costs ~12 sim-steps total. Brain
data (~260MB) and checkpoints persist on the `clayfly-data` volume.
Auth: `modal token set` or MODAL_TOKEN_ID/MODAL_TOKEN_SECRET env vars.

The horizon below must match the horizon the demo serves. `serving/serve.py`
reads `sim_steps` back out of the checkpoint, so a checkpoint trained here
at 12 steps is served at 12 steps automatically — but a checkpoint trained
at 4 steps while the demo expects 12 (or the reverse) puts the readout off
its training distribution. 4 steps measured 52.8% linear separability of
account quality with 3.4% of features firing; 12 steps measured 85.7% with
76% firing. See experiments/sensitivity.py.
"""
import os
import subprocess

import modal

app = modal.App("clayfly")

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


@app.function(image=img, gpu="B200", volumes={"/data": vol},
              timeout=3600)
def probe(batch: int = 512, steps: int = 50):
    """Sanity: image builds, brain downloads to the volume, CUDA steps.

    Also reports the tracked-population census. `n_feat` must equal the
    value the local server expects, or `serving/serve.py` will reject the
    trained checkpoint and silently fall back to another one — which is
    exactly how a stale volume artifact ended up being served as if it
    were the trained policy.
    """
    import glob
    import os, time
    import numpy as np
    print("cuda candidates:", glob.glob("/usr/local/cuda*"),
          glob.glob("/usr/local/lib/python*/site-packages/nvidia/*"))
    os.environ["CUDA_PATH"] = "/usr/local/cuda"
    os.environ["FLY_DATA"] = "/data/fly-data"
    from flybrain import FlyBrain
    from brain.loader import weights_checksum
    from brain.populations import build_channel_map, tracked_set
    from learning.encoder import DEFAULT_GAIN
    brain = FlyBrain(device="cuda", batch=batch, sensory_input=False)
    channels = build_channel_map(brain)
    tracked = tracked_set(brain, channels)
    dn = brain.cells(["descending_neuron"])
    an = brain.cells(["ascending_neuron"])
    print(f"census: descending={len(dn)} ascending={len(an)} "
          f"tracked={len(tracked)} n_feat={len(tracked) * 3} "
          f"channels={len(channels)}")
    print(f"connectome sha256 (in-memory): {weights_checksum(brain=brain)}")
    print(f"encoder DEFAULT_GAIN: {DEFAULT_GAIN}")
    dn64 = dn[:64]
    inject = [(dn64, np.full(batch, DEFAULT_GAIN))]
    brain.step(inject=inject)          # warm
    t0 = time.time()
    for _ in range(steps):
        brain.step(inject=inject)
    ms = (time.time() - t0) / steps * 1000
    vol.commit()
    return {"n": brain.n, "batch": batch, "ms_per_step": ms, "gpu": True,
            "tracked": len(tracked), "n_feat": len(tracked) * 3,
            "connectome_sha256": weights_checksum(brain=brain),
            "encoder_gain": DEFAULT_GAIN}


@app.function(image=img, gpu="B200", volumes={"/data": vol},
              timeout=24 * 3600)
def train_remote(mlp: bool = False, seed0: int = 10_000, sim_steps: int = 12,
                 commit: str = "working-tree", tag: str = "",
                 use_raw_head: bool = True, shuffled: bool = False,
                 rent: float = 0.0, kill_bonus: float = 0.0,
                 kill_penalty: float = 0.0, expire_all: bool = False,
                 ep_mult: float = 1.0, imitation_final: float = 0.2,
                 ignore_anchor_exempt: bool = False):
    import os
    os.environ["FLY_DATA"] = "/data/fly-data"   # persists across runs
    os.environ["FLY_SIM_STEPS"] = str(sim_steps)
    # `add_local_python_source` copies source without .git, so the remote
    # cannot derive the commit itself. Take it from the local checkout so
    # the checkpoint and manifest keep real provenance.
    os.environ["FLY_GIT_COMMIT"] = commit
    from learning.train import FULL, run
    brain_data = None
    if shuffled:
        # Same cells and channel populations, postsynaptic targets
        # re-permuted once on the volume — the random-reservoir control.
        from brain.loader import make_shuffled_connectome
        brain_data = str(make_shuffled_connectome(
            "/data/fly-data-shuffled", data="/data/fly-data"))
        vol.commit()
    # Tagged output dirs keep prior runs on the volume. The 1,256.1 run is
    # cited in README as a dated legacy artifact and must not be clobbered.
    suffix = ("_mlp" if mlp else "") + ("_shuffled" if shuffled else "") \
        + (f"_{tag}" if tag else "")
    world_kwargs = {"rent": rent, "kill_bonus": kill_bonus,
                    "kill_penalty": kill_penalty,
                    "expire_all": expire_all}
    val, path = run(FULL, brain_device="cuda", torch_device="cuda",
                    out_dir=f"/data/results{suffix}", train_mlp=mlp,
                    seed0=seed0, use_raw_head=use_raw_head,
                    brain_data=brain_data, world_kwargs=world_kwargs,
                    ep_mult=ep_mult, imitation_final=imitation_final,
                    ignore_anchor_exempt=ignore_anchor_exempt)
    vol.commit()
    return {"best_val": val, "checkpoint": path, "sim_steps": sim_steps,
            "use_raw_head": use_raw_head, "shuffled": shuffled,
            "world_kwargs": world_kwargs,
            "imitation_final": imitation_final,
            "ignore_anchor_exempt": ignore_anchor_exempt,
            "out_dir": f"/data/results{suffix}"}


@app.local_entrypoint()
def main(mlp: bool = False, probe_only: bool = False, sim_steps: int = 12,
         tag: str = "", no_raw_head: bool = False, shuffled: bool = False,
         rent: float = 0.0, kill_bonus: float = 0.0,
         kill_penalty: float = 0.0, expire_all: bool = False,
         ep_mult: float = 1.0, imitation_final: float = 0.2,
         ignore_anchor_exempt: bool = False):
    if probe_only:
        print(probe.remote())
        return
    commit = subprocess.run(
        ["git", "-C", str(os.path.dirname(os.path.abspath(__file__))),
         "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip() or "working-tree"
    dirty = subprocess.run(
        ["git", "-C", str(os.path.dirname(os.path.abspath(__file__))),
         "status", "--porcelain"],
        capture_output=True, text=True).stdout.strip()
    if dirty:
        commit += "-dirty"
    print(f"training commit: {commit}  sim_steps: {sim_steps}  tag: {tag or '-'}"
          f"  use_raw_head: {not no_raw_head}  shuffled: {shuffled}"
          f"  world: rent={rent} kill_bonus={kill_bonus}"
          f" kill_penalty={kill_penalty} expire_all={expire_all}"
          f" ep_mult={ep_mult}  imitation_final={imitation_final}"
          f"  ignore_anchor_exempt={ignore_anchor_exempt}")
    print(train_remote.remote(mlp=mlp, sim_steps=sim_steps, commit=commit,
                              tag=tag, use_raw_head=not no_raw_head,
                              shuffled=shuffled, rent=rent,
                              kill_bonus=kill_bonus,
                              kill_penalty=kill_penalty,
                              expire_all=expire_all, ep_mult=ep_mult,
                              imitation_final=imitation_final,
                              ignore_anchor_exempt=ignore_anchor_exempt))
