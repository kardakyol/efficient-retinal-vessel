# Sensitivity of Frangi parameters

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from retinal_selflabel.core.configurable_sampler import (
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
from retinal_selflabel.core.frangi_density import frangi_config
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

SIGMA_CONFIGS = {
    "dataset_adaptive": None,   # use per-dataset defaults
    "global_narrow":    frangi_config(sigma_min=1.0, sigma_max=6.0,  n_scales=6),
    "global_wide":      frangi_config(sigma_min=0.5, sigma_max=12.0, n_scales=8),
}


def run_sigma_point(train, val_loader, test_loader, sigma_name, sigma_cfg, n_boxes, 
                    seed, img_size, bs, num_epochs, device, log_dir, cache_dir, verbose = True):
    set_seed(seed)
    cell = factorial_cell(frangi=True, non_overlap=True, variable_size=True)

    # per-config cache 
    cache_dir = os.path.join(cache_dir, sigma_name)
    os.makedirs(cache_dir, exist_ok=True)

    sampler = config_sampler(samples=train, cell=cell, seed=seed, cache_dir=cache_dir, frangi_config_override=sigma_cfg)
    placements = sampler.sample_boxes(n_boxes)
    by_sample = config_sampler.group_by_sample(placements)
    coverage = config_sampler.coverage_fraction(placements, sampler.total_image_area())

    placement_path = os.path.join(log_dir, f"placements_{sigma_name}_n{n_boxes}_s{seed}.json")
    save_placements(
        placements, placement_path,
        metadata={
            "n_boxes": n_boxes, "seed": seed,
            "sigma_config_name": sigma_name,
            "sigma_config": (
                {"sigma_min": sigma_cfg.sigma_min,
                 "sigma_max": sigma_cfg.sigma_max,
                 "n_scales":  sigma_cfg.n_scales}
                if sigma_cfg is not None else "dataset_adaptive"), 
                "effective_coverage": float(coverage), "covered_image_count": len(by_sample)})

    train_ds = FullImageSparseDataset(samples=train, placements_by_sample=by_sample, img_size=img_size,
                                      transform=get_full_image_sparse_transforms(img_size, train=True), include_uncovered=True)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=False)
    t0 = time.time()
    model, best_val = train_full_image_sparse(train_loader=train_loader, val_loader=val_loader,
                                              device=device, num_epochs=num_epochs, verbose=verbose)
    elapsed = time.time() - t0
    val_criterion = create_loss("bce_dice")
    final = evaluate(model, test_loader, val_criterion, device)

    return {
        "sigma_config_name": sigma_name,
        "n_boxes": n_boxes, "seed": seed,
        "effective_coverage": float(coverage),
        "coverage_pct": float(coverage * 100.0),
        "covered_image_count": int(len(by_sample)),
        "best_val_dice": float(best_val),
        "final_metrics": {k: float(v) for k, v in final.items()},
        "training_seconds": float(elapsed),
        "placements_file": placement_path,
    }


def run_sigma_robustness(data_dir, out_dir, n_boxes = 225, seeds = None, 
                         sigma_configs = None, img_size = 512, bs = 4, num_epochs = 80,
                         split_seed = 42, no_cuda = False,):
    
    seeds = seeds if seeds is not None else [42, 123, 7]
    sigma_configs = sigma_configs if sigma_configs is not None else SIGMA_CONFIGS

    device = torch.device("cpu")
    if not no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"device: {device}")
    print(f"sigma robustness: {len(sigma_configs)} configs x {len(seeds)} seeds = {len(sigma_configs)*len(seeds)} runs at N={n_boxes}")

    all_samples = feature_discovery(data_dir)
    train, val, test = dataset_splitting(all_samples, val_frac=0.15, test_frac=0.2, seed=split_seed)

    val_ds = RetinalVesselDataset(val, transform=transform_images("validation", img_size))
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    test_ds = RetinalVesselDataset(test, transform=transform_images("validation", img_size))
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    log_dir = os.path.join(out_dir, "logs", "frangi_sigma_v2")
    log_dir = os.path.join(log_dir, "placements")
    cache_dir = os.path.join(out_dir, "cache", "frangi_sigma")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    results_path = os.path.join(log_dir, "sigma_robustness_results.json")

    all_runs = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_runs = json.load(f).get("runs", [])
        done = {(r["sigma_config_name"], r["seed"]) for r in all_runs}
        print(f"Resuming: {len(done)} runs done.")
    else:
        done = set()

    total = len(sigma_configs) * len(seeds)
    completed = len(done)

    for name, cfg in sigma_configs.items():
        for seed in seeds:
            if (name, seed) in done:
                continue
            completed += 1
            print(f"[{completed}/{total}] sigma={name}  seed={seed}")
            try:
                result = run_sigma_point(train=train, val_loader=val_loader, 
                                         test_loader=test_loader,sigma_name=name, 
                                         sigma_cfg=cfg, n_boxes=n_boxes, seed=seed,
                                         img_size=img_size, bs=bs, num_epochs=num_epochs, 
                                         device=device, log_dir=log_dir, cache_dir=cache_dir,
                                         verbose=True)
            except Exception as e:
                print(f"Error {name} s={seed}: {e}")
                import traceback; traceback.print_exc()
                continue
            all_runs.append(result)
            with open(results_path, "w") as f:
                json.dump({
                    "config": {"n_boxes": n_boxes, "seeds": seeds,
                        "sigma_configs": {
                            name: (None if cfg is None else
                                   {"sigma_min": cfg.sigma_min,
                                    "sigma_max": cfg.sigma_max,
                                    "n_scales":  cfg.n_scales})
                            for name, cfg in sigma_configs.items()
                        },"image_size": img_size, "batch_size": bs,
                        "epochs": num_epochs, "split_seed": split_seed},"runs": all_runs}, f, indent=2)
            print(f"  -> cov={result['coverage_pct']:.3f}%, dice={result['final_metrics']['dice']:.4f}  "
                f"({result['training_seconds']/60:.1f} min)")
    print(f"\nDone {results_path}")
    return {"results_path": results_path, "runs": all_runs}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./outputs_new")
    p.add_argument("--n_boxes", type=int, default=225)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--num_epochs", type=int, default=80)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--no_cuda", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    run_sigma_robustness(data_dir=args.data_dir, out_dir=args.out_dir, n_boxes=args.n_boxes, 
                         seeds=args.seeds, img_size=args.img_size, bs=args.bs, 
                         num_epochs=args.num_epochs, split_seed=args.split_seed, no_cuda=args.no_cuda)

if __name__ == "__main__":
    main()
