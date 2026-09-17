"""B200 seed sweep: one remote job per seed, same frozen connectome.

  modal run modal_sweep.py --seeds 4

The output is a JSON list of held-out returns, not a cherry-picked run.
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
app = modal.App("clayfly-sweep")


@app.function(image=img, gpu="B200", volumes={"/data": vol}, timeout=3600)
def run_seed(seed: int):
    import os
    os.environ["FLY_DATA"] = "/data/fly-data"
    os.environ["FLY_SIM_STEPS"] = "4"
    from learning.train import SMOKE, run
    val, path = run(SMOKE, brain_device="cuda", torch_device="cuda",
                    out_dir=f"/data/sweeps/seed-{seed}", seed0=20_000 + seed,
                    verbose=False)
    vol.commit()
    return {"seed": seed, "validation": float(val), "path": path}


@app.local_entrypoint()
def main(seeds: int = 4):
    results = list(run_seed.map(range(seeds)))
    print(json.dumps(results, indent=2))
