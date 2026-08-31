import argparse
import copy
import json
import os
import time

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# Project modules
from retinal_selflabel.core.datasets import (
    RetinalVesselDataset,
    feature_discovery,
    transform_images,
    sample_splitting,
    dataset_splitting,
)
from retinal_selflabel.core.full_image_sparse_dataset import (
    FullImageSparseDataset,
    MaskedBCEDiceLoss,
    get_full_image_sparse_transforms,
)
from retinal_selflabel.core.models import create_loss, create_model
from retinal_selflabel.core.random_box_sampler import save_placements

from retinal_selflabel.core.random_box_sampler_v2 import NewRandomizer
from retinal_selflabel.core.train import evaluate
from retinal_selflabel.core.utils import set_seed

# The parabolic sweep grid
def quadratic_grid(k_max = 27):
    return [k * k for k in range(1, k_max + 1)]


DEFAULT_SWEEP_GRID = quadratic_grid(27)


# Training loop
def train_masked_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running = 0.0
    n_batches = 0
    for images, gts, vmasks in loader:
        images = images.to(device, non_blocking=True)
        gts = gts.to(device, non_blocking=True)
        vmasks = vmasks.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, gts, vmasks)
        loss.backward()
        optimizer.step()
        running += float(loss.item())
        n_batches += 1
    return {"loss": running / max(n_batches, 1)}


def train_full_image_sparse(train_loader, val_loader, device, num_epochs=80, lr=1e-3, patience=15, verbose=True):
    model = create_model(architecture="unet", encoder="resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(device)
    train_criterion = MaskedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    val_criterion = create_loss("bce_dice")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_dice = 0.0
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        tm = train_masked_one_epoch(model, train_loader, train_criterion, optimizer, device)
        vm = evaluate(model, val_loader, val_criterion, device)
        scheduler.step()

        if vm["dice"] > best_dice:
            best_dice = vm["dice"]
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(
                f"epoch {epoch:3d}/{num_epochs} | train_loss={tm['loss']:.4f} |"
                f"val_dice={vm['dice']:.4f} | best={best_dice:.4f}"
)
        if no_improve >= patience:
            if verbose:
                print(f"Early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, best_dice

# Per-point runner
def run_single_point_v2(train, val_loader, test_loader, n_boxes, seed, min_size, max_size, 
                        img_size, bs, num_epochs, device, log_dir, cache_dir, verbose = True):
    set_seed(seed)

    # 1. Sample boxes
    sampler = NewRandomizer(train, seed=seed, min_size=min_size, max_size=max_size,
                            dataset_weighting="by_image_count",cache_dir=cache_dir)
    
    placements = sampler.sample_boxes(n_boxes)
    by_sample = NewRandomizer.group_by_sample(placements)
    coverage = NewRandomizer.coverage_fraction(placements, sampler.total_image_area())
    covered_image_count = len(by_sample)

    # persist placements
    placement_path = os.path.join(log_dir, f"placements_v2_n{n_boxes}_s{seed}.json")
    save_placements(
        placements, placement_path,
        metadata={"n_boxes": n_boxes, "seed": seed, "min_box_size": min_size, 
                  "max_box_size": max_size, "effective_coverage": coverage,
                  "covered_image_count": covered_image_count, "total_training_images": len(train),
                  "sampler": "randomv2"})

    #training loader
    train_ds = FullImageSparseDataset(samples=train, placements_by_sample=by_sample, 
                                      img_size=img_size, transform=get_full_image_sparse_transforms(img_size, train=True),
                                      include_uncovered=True)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=False)

    # train + evaluate
    t0 = time.time()
    model, best_val_dice = train_full_image_sparse(
        train_loader=train_loader, val_loader=val_loader,
        device=device, num_epochs=num_epochs, verbose=verbose,
)
    elapsed = time.time() - t0

    val_criterion = create_loss("bce_dice")
    final = evaluate(model, test_loader, val_criterion, device)

    return {"n_boxes": n_boxes, "seed": seed, "min_box_size": min_size, 
            "max_box_size": max_size, "effective_coverage": coverage, "coverage_pct": coverage * 100.0,
            "covered_image_count": covered_image_count, "best_val_dice": float(best_val_dice), "final_metrics": {k: float(v) for k, v in final.items()},
            "training_seconds": float(elapsed), "placements_file": placement_path}


# Sweep orchestration
def run_sweep_v2(data_dir, out_dir, box_counts = None, seeds = None, min_size = 32, max_size = 256,
                 img_size = 512, bs = 4, num_epochs = 80, split_seed = 42, no_cuda = False):
    if box_counts is None:
        box_counts = DEFAULT_SWEEP_GRID
    if seeds is None:
        seeds = [42]

    device = torch.device("cpu")
    if not no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"Device: {device}")
    print(f"Sweep grid ({len(box_counts)} points): {box_counts}")
    print(f"Box size range: [{min_size}, {max_size}]")
    print(f"Seeds: {seeds}")

    all_samples = feature_discovery(data_dir)
    train, val, test = dataset_splitting(all_samples, val_frac=0.15, test_frac=0.2, seed=split_seed)

    val_ds = RetinalVesselDataset(val, transform=transform_images("validation", img_size))
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    test_ds = RetinalVesselDataset(test, transform=transform_images("validation", img_size))
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    results_dir = os.path.join(out_dir, "logs", "random_sweep_v2")
    log_dir = os.path.join(results_dir, "placements")
    cache_dir = os.path.join(out_dir, "cache", "frangi")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "sweep_results_v2.json")

    all_runs = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_runs = json.load(f).get("runs", [])
        done = {(r["n_boxes"], r["seed"]) for r in all_runs}
        print(f"Info Resuming: {len(done)} runs already complete.")
    else:
        done = set()

    total_runs = len(box_counts) * len(seeds)
    completed = len(done)

    for n_boxes in box_counts:
        for seed in seeds:
            if (n_boxes, seed) in done:
                print(f"\nSkip n_boxes={n_boxes} seed={seed} already done.")
                continue

            completed += 1
            print(f"[{completed}/{total_runs}] n_boxes={n_boxes}  seed={seed}")
            try:
                result = run_single_point_v2(train=train, val_loader=val_loader, test_loader=test_loader,
                                             n_boxes=n_boxes, seed=seed, min_size=min_size,
                                             max_size=max_size, img_size=img_size, bs=bs, num_epochs=num_epochs,
                                             device=device, log_dir=log_dir, cache_dir=cache_dir, verbose=True)
            except Exception as e:
                print(f"Error n_boxes={n_boxes} seed={seed} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

            all_runs.append(result)
            with open(results_path, "w") as f:
                json.dump({"config": {"box_counts": box_counts, "seeds": seeds, "min_box_size": min_size, 
                                      "max_box_size": max_size, "image_size": img_size, "batch_size": bs,
                                      "epochs": num_epochs, "split_seed": split_seed, "total_training_images": len(train),
                                      "total_test_images": len(test), "sampler": "randomv2 (Frangi + non-overlap + variable size)"},"runs": all_runs}, f, indent=2)
            print(f"coverage={result['coverage_pct']:.3f}% | dice={result['final_metrics']['dice']:.4f} | ({result['training_seconds']/60:.1f} min)")
    print(f"\nDone All sweep results in {results_path}")
    return {"results_path": results_path, "runs": all_runs}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./outputs_new")
    p.add_argument("--box_counts", type=int, nargs="+", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
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
    run_sweep_v2(data_dir=args.data_dir, out_dir=args.out_dir, box_counts=args.box_counts,
                 seeds=args.seeds, min_size=args.min_size, max_size=args.max_size, img_size=args.img_size,
                 bs=args.bs, num_epochs=args.num_epochs, split_seed=args.split_seed, no_cuda=args.no_cuda)

if __name__ == "__main__":
    main()
