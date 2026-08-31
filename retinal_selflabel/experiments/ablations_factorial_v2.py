# 2x2x2 factorial ablation runner.
import argparse
import json
import os
import time

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

# default factorial grid
from retinal_selflabel.experiments.experiments_random_sweep_v2 import (
    DEFAULT_SWEEP_GRID,
    train_full_image_sparse,
)

DEFAULT_BUDGETS = list(DEFAULT_SWEEP_GRID)   # 27-point quadratic grid
DEFAULT_SEEDS = [42, 123, 7]

# smoke test grid
SMOKE_BUDGETS = [1, 81, 400, 729]


# per-cell, per-budget, per-seed runner
def run_factorial_point(train, val_loader, test_loader, cell, n_boxes, seed, fixed_size,
                        min_size, max_size, img_size, bs, num_epochs, device, log_dir,
                        cache_dir, verbose=True):
    
    set_seed(seed)

    sampler = config_sampler(samples=train, cell=cell, seed=seed, fixed_size=fixed_size,
                             min_size=min_size, max_size=max_size, dataset_weighting="by_image_count",
                             cache_dir=cache_dir)
    placements = sampler.sample_boxes(n_boxes)
    by_sample = config_sampler.group_by_sample(placements)
    coverage = config_sampler.coverage_fraction(placements, sampler.total_image_area())
    covered_image_count = len(by_sample)

    placement_path = os.path.join(log_dir, f"placements_{cell.short}_n{n_boxes}_s{seed}.json")
    
    save_placements(
        placements, placement_path,
        metadata={
            **sampler.as_metadata(),
            "n_boxes": n_boxes, "seed": seed,
            "effective_coverage": coverage,
            "covered_image_count": covered_image_count,
            "total_training_images": len(train),
        },
    )

    train_ds = FullImageSparseDataset(samples=train, placements_by_sample=by_sample, img_size=img_size, 
                                      transform=get_full_image_sparse_transforms(img_size, train=True),
                                      include_uncovered=True)
    
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=False)

    t0 = time.time()
    model, best_val_dice = train_full_image_sparse(
        train_loader=train_loader, val_loader=val_loader,
        device=device, num_epochs=num_epochs, verbose=verbose,
    )
    elapsed = time.time() - t0
    val_criterion = create_loss("bce_dice")
    final = evaluate(model, test_loader, val_criterion, device)

    return {
        "cell": cell.as_dict(), "n_boxes": n_boxes, "seed": seed, "fixed_box_size": fixed_size,
        "min_box_size": min_size, "max_box_size": max_size, "effective_coverage": float(coverage),
        "coverage_pct": float(coverage * 100.0), "covered_image_count": int(covered_image_count),
        "best_val_dice": float(best_val_dice), "final_metrics": {k: float(v) for k, v in final.items()},
        "training_seconds": float(elapsed), "placements_file": placement_path,
    }


# factorial orchestrator
def run_factorial(data_dir, out_dir, budgets=None, seeds=None, cells=None,
                  fixed_size=128, min_size=32, max_size=256, img_size=512, 
                  bs=4, num_epochs=80, split_seed=42, no_cuda=False):
    
    budgets = budgets if budgets is not None else list(DEFAULT_BUDGETS)
    seeds = seeds if seeds is not None else list(DEFAULT_SEEDS)
    cells = cells if cells is not None else list(fos_cells)

    device = torch.device("cpu")
    if not no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"Factorial: {len(cells)} cells x {len(budgets)} budgets "
          f"x {len(seeds)} seeds = {len(cells)*len(budgets)*len(seeds)} runs")

    all_samples = feature_discovery(data_dir)
    train, val, test = dataset_splitting(all_samples, val_frac=0.15, test_frac=0.2, seed=split_seed)

    val_ds = RetinalVesselDataset(val, transform=transform_images("validation", img_size))
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    
    test_ds = RetinalVesselDataset(test, transform=transform_images("validation", img_size))
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    log_dir = os.path.join(out_dir, "logs", "ablations_factorial_v2")
    log_dir = os.path.join(log_dir, "placements")
    cache_dir = os.path.join(out_dir, "cache", "frangi")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    results_path = os.path.join(log_dir, "factorial_results.json")

    all_runs = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_runs = json.load(f).get("runs", [])
        done = {(r["cell"]["label"], r["n_boxes"], r["seed"]) for r in all_runs}
        print(f"resuming: {len(done)} runs already done.")
    else:
        done = set()

    total = len(cells) * len(budgets) * len(seeds)
    completed = len(done)

    for cell in cells:
        for n in budgets:
            for seed in seeds:
                key = (cell.label, n, seed)
                if key in done:
                    print(f"\n{cell.label} N={n} seed={seed}")
                    continue

                completed += 1
                print(f"[{completed}/{total}] {cell.label} ({cell.short})  N={n}  seed={seed}")
                try:
                    result = run_factorial_point(train=train, val_loader=val_loader, test_loader=test_loader,
                                                 cell=cell, n_boxes=n, seed=seed, fixed_size=fixed_size,
                                                 min_size=min_size, max_size=max_size, img_size=img_size,
                                                 bs=bs, num_epochs=num_epochs, device=device, log_dir=log_dir, 
                                                 cache_dir=cache_dir, verbose=True)
                except Exception as e:
                    print(f"error!! {cell.label} N={n} seed={seed}: {e}")
                    import traceback; traceback.print_exc()
                    continue

                all_runs.append(result)
                with open(results_path, "w") as f:
                    json.dump({
                        "config": {"cells":[c.as_dict() for c in cells], "budgets": budgets, "seeds": seeds,
                            "fixed_box_size": fixed_size,"min_box_size": min_size, "max_box_size": max_size,
                            "image_size": img_size, "batch_size": bs, "epochs": num_epochs,
                            "split_seed": split_seed}, "runs": all_runs}, f, indent=2)
                print(f"  -> cov={result['coverage_pct']:.3f}%  covered={result['covered_image_count']}  "
                      f"dice={result['final_metrics']['dice']:.4f}  ({result['training_seconds']/60:.1f} min)")

    print(f"\ndone. {results_path}")
    return {"results_path": results_path, "runs": all_runs}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./outputs_new")
    p.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS)
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
    run_factorial(
        data_dir=args.data_dir, out_dir=args.out_dir, budgets=args.budgets, 
        seeds=args.seeds, fixed_size=args.fixed_size, min_size=args.min_size,
        max_size=args.max_size, img_size=args.img_size, bs=args.bs,
        num_epochs=args.num_epochs, split_seed=args.split_seed, no_cuda=args.no_cuda)

if __name__ == "__main__":
    main()
