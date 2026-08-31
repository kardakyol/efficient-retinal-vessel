# extra seeds for the anomalous sweep points
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from retinal_selflabel.core.datasets import (
    RetinalVesselDataset,
    feature_discovery,
    transform_images,
    sample_splitting,
)
from retinal_selflabel.experiments.experiments_random_sweep_v2 import (
    run_single_point_v2,
)

ANOMALOUS_NS = [144, 196, 576]
EXTRA_SEEDS = [314, 271]

def run_stability(data_dir, out_dir, ns = None, extra_seeds = None, min_size = 32, 
                  max_size = 256, img_size = 512, bs = 4, num_epochs = 80, split_seed = 42,
                  no_cuda = False):
    ns = ns if ns is not None else list(ANOMALOUS_NS)
    extra_seeds = extra_seeds if extra_seeds is not None else list(EXTRA_SEEDS)

    device = torch.device("cpu")
    if not no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")

    all_samples = feature_discovery(data_dir)
    train, test = sample_splitting(all_samples, test_frac=0.2, seed=split_seed)
    val_ds = RetinalVesselDataset(test, transform=transform_images("validation", img_size))
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    sweep_results_dir = os.path.join(out_dir, "logs", "random_sweep_v2")
    log_dir = os.path.join(sweep_results_dir, "placements")
    cache_dir = os.path.join(out_dir, "cache", "frangi")
    os.makedirs(log_dir, exist_ok=True)
    sweep_results_path = os.path.join(sweep_results_dir, "sweep_results_v2.json")

    # headline sweep error
    if not os.path.exists(sweep_results_path):
        raise FileNotFoundError(f"Headline sweep results not found at {sweep_results_path}.")
    with open(sweep_results_path) as f:
        payload = json.load(f)
    runs = payload["runs"]
    done = {(r["n_boxes"], r["seed"]) for r in runs}

    print(f"Device: {device}")
    print(f"Stability: {len(ns)} N points x {len(extra_seeds)} extra seeds")

    total = len(ns) * len(extra_seeds)
    completed = 0
    for n in ns:
        for seed in extra_seeds:
            if (n, seed) in done:
                print(f"Skip N={n} seed={seed} already in JSON")
                continue
            completed += 1
            print(f"\n[{completed}/{total}] Stability run: N={n}  seed={seed}")
            try:
                result = run_single_point_v2(train=train, val_loader=val_loader, n_boxes=n, seed=seed,
                                             min_size=min_size, max_size=max_size, img_size=img_size, 
                                             bs=bs, num_epochs=num_epochs, device=device, log_dir=log_dir, 
                                             cache_dir=cache_dir, verbose=True)
            except Exception as e:
                print(f"Error N={n} seed={seed}: {e}")
                import traceback; traceback.print_exc()
                continue
            runs.append(result)
            payload["runs"] = runs
            with open(sweep_results_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f" dice={result['final_metrics']['dice']:.4f}")
    print(f"\nDone Stability runs appended to {sweep_results_path}")
    return {"results_path": sweep_results_path}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./outputs_new")
    p.add_argument("--ns", type=int, nargs="+", default=ANOMALOUS_NS)
    p.add_argument("--extra_seeds", type=int, nargs="+", default=EXTRA_SEEDS)
    p.add_argument("--num_epochs", type=int, default=80)
    p.add_argument("--no_cuda", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    run_stability(data_dir=args.data_dir, out_dir=args.out_dir, ns=args.ns, 
                  extra_seeds=args.extra_seeds, num_epochs=args.num_epochs, 
                  no_cuda=args.no_cuda)

if __name__ == "__main__":
    main()
