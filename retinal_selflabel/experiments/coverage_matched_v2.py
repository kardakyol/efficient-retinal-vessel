import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from retinal_selflabel.core.configurable_sampler import (
    fos_cells,
    config_sampler,
    factorial_cell,
)
from retinal_selflabel.core.datasets import (
    RetinalVesselDataset,
    feature_discovery,
    transform_images,
    sample_splitting,
    dataset_splitting,
)
from retinal_selflabel.core.full_image_sparse_dataset import (
    FullImageSparseDataset,
    get_full_image_sparse_transforms,
)
from retinal_selflabel.core.models import create_loss
from retinal_selflabel.core.random_box_sampler import save_placements
from retinal_selflabel.core.train import evaluate
from retinal_selflabel.core.utils import set_seed
from retinal_selflabel.experiments.experiments_random_sweep_v2 import (
    train_full_image_sparse,
)

DEFAULT_TARGETS = [0.37, 0.75, 1.47, 2.82]   # percent
DEFAULT_SEEDS   = [42, 123, 7]


# Calibration
def calibrate_n_for_target(samples, cell, target_coverage_pct, cache_dir, 
                           fixed_size, min_size, max_size, seed, 
                           probe_n = 100, safety = 1.05, abs_cap = 1500):
    sampler = config_sampler( samples=samples, cell=cell, seed=seed, fixed_size=fixed_size,
                             min_size=min_size, max_size=max_size, cache_dir=cache_dir,)
    probe = sampler.sample_boxes(probe_n)
    cov_pct = 100.0 * config_sampler.coverage_fraction(probe, sampler.total_image_area())
    if cov_pct <= 0:
        return abs_cap
    boxes_per_pct = probe_n / cov_pct
    n_est = int(np.ceil(boxes_per_pct * target_coverage_pct * safety))
    return min(max(1, n_est), abs_cap)


# per-point runner
def run_coverage_point(train, val_loader, test_loader, cell, target_pct, seed, fixed_size,
                       min_size, max_size, img_size, bs, num_epochs, device, log_dir, 
                       cache_dir, verbose = True):
    set_seed(seed)

    n_est = calibrate_n_for_target(samples=train, cell=cell, target_coverage_pct=target_pct,
                                   cache_dir=cache_dir, fixed_size=fixed_size, min_size=min_size,
                                   max_size=max_size, seed=seed)
    if verbose:
        print(f"calibrated N={n_est} for target {target_pct:.2f}% coverage.")

    sampler = config_sampler(samples=train, cell=cell, seed=seed, fixed_size=fixed_size,
                             min_size=min_size, max_size=max_size, cache_dir=cache_dir)
    
    placements = sampler.sample_boxes(n_est)
    by_sample = config_sampler.group_by_sample(placements)
    coverage = config_sampler.coverage_fraction(placements, sampler.total_image_area())
    covered_image_count = len(by_sample)

    placement_path = os.path.join(log_dir, f"placements_{cell.short}_tgt{int(target_pct*100):04d}_s{seed}.json")
    
    save_placements(placements, placement_path,
        metadata={**sampler.as_metadata(), "target_coverage_pct": target_pct, 
                  "n_calibrated": n_est, "seed": seed, "effective_coverage": float(coverage),
                  "covered_image_count": covered_image_count,})

    train_ds = FullImageSparseDataset(samples=train, placements_by_sample=by_sample,
                                      img_size=img_size, transform=get_full_image_sparse_transforms(img_size, train=True),
                                      include_uncovered=True)
    
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=False)

    t0 = time.time()
    model, best_val_dice = train_full_image_sparse( train_loader=train_loader, 
                                                   val_loader=val_loader, device=device,
                                                   num_epochs=num_epochs, verbose=verbose)
    elapsed = time.time() - t0
    val_criterion = create_loss("bce_dice")
    final = evaluate(model, test_loader, val_criterion, device)

    return {
        "cell": cell.as_dict(),
        "target_coverage_pct": target_pct,
        "n_boxes": n_est, "seed": seed,
        "effective_coverage": float(coverage),
        "coverage_pct": float(coverage * 100.0),
        "covered_image_count": int(covered_image_count),
        "best_val_dice": float(best_val_dice),
        "final_metrics": {k: float(v) for k, v in final.items()},
        "training_seconds": float(elapsed),
        "placements_file": placement_path,
    }


# orchestration
def run_coverage_matched(data_dir, out_dir, targets = None, seeds = None, cells = None,
                         fixed_size = 128, min_size = 32, max_size = 256, img_size = 512,
                         bs = 4, num_epochs = 80, split_seed = 42, no_cuda = False):
    
    targets = targets if targets is not None else list(DEFAULT_TARGETS)
    seeds = seeds if seeds is not None else list(DEFAULT_SEEDS)
    cells = cells if cells is not None else list(fos_cells)

    device = torch.device("cpu")
    if not no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"Device: {device}")
    print(f"Coverage-matched: {len(cells)} cells x {len(targets)} targets x {len(seeds)} seeds = {len(cells)*len(targets)*len(seeds)} runs")

    all_samples = feature_discovery(data_dir)
    train, val, test = dataset_splitting(all_samples, val_frac=0.15, test_frac=0.2, seed=split_seed)

    val_ds = RetinalVesselDataset(val, transform=transform_images("validation", img_size))
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    test_ds = RetinalVesselDataset(test, transform=transform_images("validation", img_size))
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    log_dir = os.path.join(out_dir, "logs", "coverage_matched_v2")
    log_dir = os.path.join(log_dir, "placements")
    cache_dir = os.path.join(out_dir, "cache", "frangi")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    results_path = os.path.join(log_dir, "coverage_matched_results.json")

    all_runs = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_runs = json.load(f).get("runs", [])
        done = {(r["cell"]["label"], r["target_coverage_pct"], r["seed"])
                for r in all_runs}
        print(f"{len(done)} runs done.")
    else:
        done = set()

    total = len(cells) * len(targets) * len(seeds)
    completed = len(done)

    for cell in cells:
        for target in targets:
            for seed in seeds:
                key = (cell.label, target, seed)
                if key in done:
                    print(f"\npass {cell.label} tgt={target}% s={seed}")
                    continue
                completed += 1
                print(f"[{completed}/{total}] {cell.label} ({cell.short}) target={target:.2f}%  seed={seed}")
                try:
                    result = run_coverage_point(
                        train=train, val_loader=val_loader, test_loader=test_loader,
                        cell=cell, target_pct=target, seed=seed, fixed_size=fixed_size,
                        min_size=min_size, max_size=max_size, img_size=img_size, bs=bs,
                        num_epochs=num_epochs, device=device, log_dir=log_dir, cache_dir=cache_dir,
                        verbose=True)
                except Exception as e:
                    print(f"error! {cell.label} tgt={target} s={seed}: {e}")
                    import traceback; traceback.print_exc()
                    continue
                all_runs.append(result)
                with open(results_path, "w") as f:
                    json.dump({
                        "config": {"cells": [c.as_dict() for c in cells],"targets": targets,
                                   "seeds": seeds, "fixed_box_size": fixed_size, "min_box_size": min_size,
                                   "max_box_size": max_size, "image_size": img_size, "batch_size": bs, "epochs": num_epochs,
                                   "split_seed": split_seed}, "runs": all_runs}, f, indent=2)
                print(f"N={result['n_boxes']}  cov={result['coverage_pct']:.3f}%  "
                f"dice={result['final_metrics']['dice']:.4f}  ({result['training_seconds']/60:.1f} min)")

    print(f"\nDone. {results_path}")
    return {"results_path": results_path, "runs": all_runs}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./outputs_new")
    p.add_argument("--targets", type=float, nargs="+", default=DEFAULT_TARGETS)
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--fixed_size", type=int, default=128)
    p.add_argument("--min_size", type=int, default=32)
    p.add_argument("--max_size", type=int, default=256)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--num_epochs", type=int, default=80)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--no_cuda", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    run_coverage_matched(data_dir=args.data_dir, out_dir=args.out_dir, targets=args.targets, 
        seeds=args.seeds, fixed_size=args.fixed_size, min_size=args.min_size,
        max_size=args.max_size, img_size=args.img_size, bs=args.bs, num_epochs=args.num_epochs, 
        split_seed=args.split_seed, no_cuda=args.no_cuda)
    
if __name__ == "__main__":
    main()
