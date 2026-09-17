"""Modal GPU training for the fly.

    modal run modal_train.py                 # full curriculum on L40S
    modal run modal_train.py --mlp           # dense baseline, same worlds
    modal volume get clayfly-data results/ ./results/

The brain batch == account count: one sparse matmul on CUDA steps all
flies at once, so a 500-account day costs ~10 sim-steps total. Brain
data (~260MB) and checkpoints persist on the `clayfly-data` volume.
Auth: `modal token set` or MODAL_TOKEN_ID/MODAL_TOKEN_SECRET env vars.
"""
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
    """Sanity: image builds, brain downloads to the volume, CUDA steps."""
    import glob
    import os, time
    import numpy as np
    print("cuda candidates:", glob.glob("/usr/local/cuda*"),
          glob.glob("/usr/local/lib/python*/site-packages/nvidia/*"))
    os.environ["CUDA_PATH"] = "/usr/local/cuda"
    os.environ["FLY_DATA"] = "/data/fly-data"
    from flybrain import FlyBrain
    brain = FlyBrain(device="cuda", batch=batch, sensory_input=False)
    dn = brain.cells(["descending_neuron"])[:64]
    inject = [(dn, np.full(batch, 0.6))]
    brain.step(inject=inject)          # warm
    t0 = time.time()
    for _ in range(steps):
        brain.step(inject=inject)
    ms = (time.time() - t0) / steps * 1000
    vol.commit()
    return {"n": brain.n, "batch": batch, "ms_per_step": ms,
            "gpu": True}


@app.function(image=img, gpu="B200", volumes={"/data": vol},
              timeout=24 * 3600)
def train_remote(mlp: bool = False, seed0: int = 10_000):
    import os
    os.environ["FLY_DATA"] = "/data/fly-data"   # persists across runs
    os.environ["FLY_SIM_STEPS"] = "4"           # demo uses 10
    from learning.train import FULL, run
    val, path = run(FULL, brain_device="cuda", torch_device="cuda",
                    out_dir="/data/results" + ("_mlp" if mlp else ""),
                    train_mlp=mlp, seed0=seed0)
    vol.commit()
    return {"best_val": val, "checkpoint": path}


@app.local_entrypoint()
def main(mlp: bool = False, probe_only: bool = False):
    if probe_only:
        print(probe.remote())
    else:
        print(train_remote.remote(mlp=mlp))
