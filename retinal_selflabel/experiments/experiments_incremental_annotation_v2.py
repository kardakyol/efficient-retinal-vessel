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
from retinal_selflabel.core.full_image_sparse_dataset import (
    FullImageSparseDataset,
    get_full_image_sparse_transforms,
)
from retinal_selflabel.core.models import create_loss, create_model
from retinal_selflabel.core.random_box_sampler import save_placements
from retinal_selflabel.core.train import evaluate
from retinal_selflabel.core.utils import set_seed
from retinal_selflabel.experiments.experiments_random_sweep_v2 import (
    train_full_image_sparse,
)

# growing pool maintainer
class IncrementalCampaignV2:
    def __init__(self, samples, cell, seed, fixed_size = 128, min_size = 32, max_size = 256, cache_dir = "./outputs_new/cache/frangi"):
        self._sampler = config_sampler(samples=samples, cell=cell, seed=seed, fixed_size=fixed_size,
                                       min_size=min_size, max_size=max_size,cache_dir=cache_dir)
        self._sampler.reset_forbidden()
        self.cumulative = []

    def add_round(self, n_new_boxes):
        new_placements = []
        for _ in range(n_new_boxes):
            p = self._sampler._draw_one_box()
            if p is None:
                break
            new_placements.append(p)
        self.cumulative.extend(new_placements)
        return list(self.cumulative)

    def coverage_pct(self):
        total = self._sampler.total_image_area()
        if total == 0:
            return 0.0
        return 100.0 * config_sampler.coverage_fraction(self.cumulative, total)

# Per-style runner
def run_campaign(train, val, test, cell, seed, style, boxes_per_round, n_rounds, fixed_size,
                 min_size, max_size, img_size, bs, epochs_per_round, lr, device, log_dir, cache_dir,
                 verbose = True):
    
    assert style in {"warm_start", "from_scratch"}
    set_seed(seed)

    val_ds = RetinalVesselDataset(val, transform=transform_images("validation", img_size))
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    test_ds = RetinalVesselDataset(test, transform=transform_images("validation", img_size))
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    campaign = IncrementalCampaignV2(samples=train, cell=cell, seed=seed, fixed_size=fixed_size,
                                     min_size=min_size, max_size=max_size, cache_dir=cache_dir)

    model = None
    placements_dir = os.path.join(log_dir, "placements")
    os.makedirs(placements_dir, exist_ok=True)
    round_log = []

    for k in range(1, n_rounds + 1):
        print(f"Round {k}/{n_rounds}  ({style}, seed={seed}, cell={cell.short})")

        cumulative = campaign.add_round(boxes_per_round)
        coverage_pct = campaign.coverage_pct()
        n_cumulative = len(cumulative)

        save_placements(cumulative, os.path.join(placements_dir,f"placements_round{k}_{style}_{cell.short}_s{seed}.json"),
            metadata={"round": k, "style": style, "seed": seed, "boxes_per_round": boxes_per_round, "n_rounds": n_rounds,
                      "cumulative_boxes": n_cumulative, "coverage_pct": coverage_pct, "cell": cell.as_dict()})

        by_sample = config_sampler.group_by_sample(cumulative)
        train_ds = FullImageSparseDataset(samples=train, placements_by_sample=by_sample, img_size=img_size,
                                          transform=get_full_image_sparse_transforms(img_size, train=True), include_uncovered=True)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=False)

        if style == "from_scratch" or model is None:
            model = create_model(architecture="unet", encoder="resnet34", encoder_weights="imagenet",
                                 in_channels=3, classes=1,).to(device)

        t0 = time.time()
        model, best_val = train_full_image_sparse(train_loader=train_loader, val_loader=val_loader,
                                                  device=device, num_epochs=epochs_per_round,
                                                  lr=lr, verbose=verbose)
        elapsed = time.time() - t0
        val_criterion = create_loss("bce_dice")
        final = evaluate(model, test_loader, val_criterion, device)

        entry = {"round": k, "style": style, "seed": seed, "cell": cell.as_dict(),
                "cumulative_boxes": n_cumulative, "coverage_pct": float(coverage_pct),
                "best_val_dice": float(best_val), "final_metrics": {kk: float(v) for kk, v in final.items()},
                "training_seconds": float(elapsed),
        }
        round_log.append(entry)
        print(f"round {k}: cum_boxes={n_cumulative}, cov={coverage_pct:.3f}%  dice={final['dice']:.4f}")

    return {"style": style, "seed": seed, "cell": cell.as_dict(), "boxes_per_round": boxes_per_round, 
            "n_rounds": n_rounds, "rounds": round_log}


# orchestration
def run_incremental_v2(data_dir, out_dir, seeds = None, styles = None, cell = None,
                       boxes_per_round = 75, n_rounds = 6, fixed_size = 128,
                       min_size = 32, max_size = 256, img_size = 512, bs = 4,
                       epochs_per_round = 80, lr = 1e-3, split_seed = 42, no_cuda = False):
    
    seeds = seeds if seeds is not None else [42, 123, 7]
    styles = styles if styles is not None else ["warm_start", "from_scratch"]
    cell = cell if cell is not None else factorial_cell(frangi=True, non_overlap=True, variable_size=True)

    device = torch.device("cpu")
    if not no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"Device: {device}")
    print(f"Cell: {cell.label}")
    print(f"{len(styles)} styles x {len(seeds)} seeds = {len(styles)*len(seeds)} campaigns, "
          f"{n_rounds} rounds each = {len(styles)*len(seeds)*n_rounds} training calls.")

    all_samples = feature_discovery(data_dir)
    train, val, test = dataset_splitting(all_samples, val_frac=0.15, test_frac=0.2, seed=split_seed)

    log_dir = os.path.join(out_dir, "logs", "incremental_v2")
    cache_dir = os.path.join(out_dir, "cache", "frangi")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    results_path = os.path.join(log_dir, "incremental_v2_results.json")

    campaigns = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            campaigns = json.load(f).get("campaigns", [])
        done = {(c["style"], c["seed"]) for c in campaigns}
        print(f"Resuming: {len(done)} campaigns done.")
    else:
        done = set()

    for style in styles:
        for seed in seeds:
            if (style, seed) in done:
                print(f"\nSkip {style} seed={seed}")
                continue
            print(f"Campaign style={style}  seed={seed}")
            try:
                result = run_campaign(train=train, val=val, test=test,
                                      cell=cell, seed=seed, style=style,
                                      boxes_per_round=boxes_per_round,
                                      n_rounds=n_rounds, fixed_size=fixed_size,
                                      min_size=min_size, max_size=max_size,
                                      img_size=img_size, bs=bs, epochs_per_round=epochs_per_round, lr=lr,
                                      device=device, log_dir=log_dir, cache_dir=cache_dir, verbose=True)
            except Exception as e:
                print(f"Error {style} seed={seed}: {e}")
                import traceback; traceback.print_exc()
                continue

            campaigns.append(result)
            with open(results_path, "w") as f:
                json.dump({"config": {"cell": cell.as_dict(), "seeds": seeds, 
                                      "styles": styles, "boxes_per_round": boxes_per_round,
                                      "n_rounds": n_rounds, "fixed_box_size": fixed_size,
                                      "min_box_size": min_size, "max_box_size": max_size, 
                                      "image_size": img_size, "batch_size": bs, "epochs_per_round": epochs_per_round,
                                      "split_seed": split_seed}, "campaigns": campaigns}, f, indent=2)

    print(f"\nDone {results_path}")
    return {"results_path": results_path, "campaigns": campaigns}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./outputs_new")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    p.add_argument("--styles", type=str, nargs="+", default=["warm_start", "from_scratch"])
    p.add_argument("--boxes_per_round", type=int, default=75)
    p.add_argument("--n_rounds", type=int, default=6)
    p.add_argument("--fixed_size", type=int, default=128)
    p.add_argument("--min_size", type=int, default=32)
    p.add_argument("--max_size", type=int, default=256)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--epochs_per_round", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--no_cuda", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    run_incremental_v2(data_dir=args.data_dir, out_dir=args.out_dir,
                       seeds=args.seeds, styles=args.styles, 
                       boxes_per_round=args.boxes_per_round, n_rounds=args.n_rounds,
                       fixed_size=args.fixed_size, min_size=args.min_size, 
                       max_size=args.max_size, img_size=args.img_size, bs=args.bs,
                       epochs_per_round=args.epochs_per_round, lr=args.lr,
                       split_seed=args.split_seed, no_cuda=args.no_cuda)


if __name__ == "__main__":
    main()
