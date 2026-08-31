# frontier positioned against acquisition-strategy baselines at matched budget
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from retinal_selflabel.core.configurable_sampler import config_sampler, factorial_cell
from retinal_selflabel.core.datasets import (
    RetinalVesselDataset,
    dataset_splitting,
    feature_discovery,
    transform_images,
)
from retinal_selflabel.core.full_image_sparse_dataset import (
    FullImageSparseDataset,
    get_full_image_sparse_transforms,
)
from retinal_selflabel.core.models import create_loss
from retinal_selflabel.core.random_box_sampler import BoxPlacement, save_placements
from retinal_selflabel.core.random_box_sampler_v2 import NewRandomizer
from retinal_selflabel.core.train import evaluate
from retinal_selflabel.core.utils import set_seed
from retinal_selflabel.experiments.experiments_random_sweep_v2 import train_full_image_sparse
from retinal_selflabel.selflabel.intelligent_guidance import binary_entropy, box_mean_field

FIXED_BOX_SIZE = 128
DEFAULT_BUDGETS = [63, 100, 169, 300]
DEFAULT_SEEDS = [42, 123, 7]
ARMS = ["random", "frangi", "entropy", "margin"]

def build_loader(train, placements, img_size, bs, augment, shuffle):
    by_sample = NewRandomizer.group_by_sample(placements)
    dataset = FullImageSparseDataset(samples=train, placements_by_sample=by_sample, img_size=img_size,
        transform=get_full_image_sparse_transforms(img_size, train=augment),
        include_uncovered=True)
    return DataLoader(dataset, batch_size=bs, shuffle=shuffle,
                      num_workers=2, pin_memory=True, drop_last=False)

def run_one_shot(train, placements, val_loader, test_loader, device, img_size, bs, num_epochs, verbose):
    train_loader = build_loader(train, placements, img_size, bs, True, True)
    model, best_val = train_full_image_sparse(
        train_loader=train_loader, val_loader=val_loader, device=device,
        num_epochs=num_epochs, verbose=verbose)
    criterion = create_loss("bce_dice")
    return model, best_val, evaluate(model, test_loader, criterion, device)

def place_static(train, n_boxes, seed, frangi_on, cache_dir):
    cell = factorial_cell(frangi=frangi_on, non_overlap=True, variable_size=False)
    sampler = config_sampler(
        train, cell=cell, seed=seed, fixed_size=FIXED_BOX_SIZE,
        dataset_weighting="by_image_count", cache_dir=cache_dir)
    return sampler.sample_boxes(n_boxes)


def predict_prob_maps(model, train, img_size, device):
    model.eval()
    transform = get_full_image_sparse_transforms(img_size, train=False)
    out = {}
    with torch.no_grad():
        for idx, sample in enumerate(train):
            img = cv2.cvtColor(cv2.imread(sample["image_path"]), cv2.COLOR_BGR2RGB)
            empty = np.zeros(img.shape[:2], dtype=np.uint8)
            transformed = transform(image=img, mask=empty, validity=empty)
            x = transformed["image"].unsqueeze(0).to(device)
            out[idx] = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return out

def score_field(prob, acquisition):
    if acquisition == "entropy":
        return binary_entropy(prob)
    if acquisition == "margin":
        return 0.5 - np.abs(prob - 0.5)
    raise ValueError(acquisition)

def _argmax2d(field):
    row, col = np.unravel_index(int(np.argmax(field)), field.shape)
    return int(row), int(col)

def forbidden_map(placements_for_image, img_size, height, weight):
    forbidden = np.zeros((img_size, img_size), dtype=np.float32)
    scale_h, scale_w = img_size / height, img_size / weight
    for placement in placements_for_image:
        r0 = int(round(placement.row * scale_h))
        c0 = int(round(placement.col * scale_w))
        size = int(round(placement.size * scale_h))
        forbidden[max(0, r0):r0 + size, max(0, c0):c0 + size] = 1.0
    return forbidden

def acquire_active(train, n_boxes, seed, acquisition, cache_dir, val_loader, device, img_size, bs, n_rounds, acquire_epochs, verbose):
    set_seed(seed)
    per_round = max(1, n_boxes // n_rounds)
    dims = {i: cv2.imread(s["image_path"]).shape[:2] for i, s in enumerate(train)}
    placements = place_static(train, per_round, seed, False, cache_dir)
    acquired, rnd = len(placements), 1

    while acquired < n_boxes and rnd < n_rounds:
        train_loader = build_loader(train, placements, img_size, bs, True, True)
        model, _ = train_full_image_sparse(
            train_loader=train_loader, val_loader=val_loader, device=device,
            num_epochs=acquire_epochs, verbose=False)
        probs = predict_prob_maps(model, train, img_size, device)
        by_sample = NewRandomizer.group_by_sample(placements)
        remaining = min(per_round, n_boxes - acquired)
        order = sorted(probs, key=lambda i: float(score_field(probs[i], acquisition).mean()),
                       reverse=True)
        newly = []
        while remaining > 0:
            progressed = False
            for idx in order:
                if remaining <= 0:
                    break
                height, weight = dims[idx]
                placed_here = by_sample.get(idx, []) + [p for p in newly if p.sample_idx == idx]
                forbidden = forbidden_map(placed_here, img_size, height, weight)
                field = score_field(probs[idx], acquisition)
                box_means = box_mean_field(field, FIXED_BOX_SIZE).copy()
                forbidden_means = box_mean_field(forbidden, FIXED_BOX_SIZE)
                box_means[forbidden_means > 0] = -np.inf
                if not np.isfinite(box_means).any():
                    continue
                row, col = _argmax2d(box_means)
                if not np.isfinite(box_means[row, col]):
                    continue
                scale_h, scale_w = height / img_size, weight / img_size
                sample = train[idx]
                newly.append(BoxPlacement(
                    sample_idx=idx, dataset=sample["dataset"], image_id=sample["id"],
                    image_h=height, image_w=weight,
                    row=int(round(row * scale_h)), col=int(round(col * scale_w)),
                    size=int(round(FIXED_BOX_SIZE * (scale_h + scale_w) / 2))))
                remaining -= 1
                acquired += 1
                progressed = True
            if not progressed:
                break
        placements.extend(newly)
        if verbose:
            print(f"{acquisition} round {rnd}: {len(placements)} boxes")
        rnd += 1

    if acquired < n_boxes:
        placements.extend(place_static(train, n_boxes - acquired, seed + 999, False, cache_dir))
    return placements[:n_boxes]


def run_comparative(data_dir, out_dir, budgets=None, arms=None, seeds=None, img_size=512, bs=4, num_epochs=80, acquire_epochs=30, n_rounds=4, split_seed=42,
                    no_cuda=False):
    budgets = budgets if budgets is not None else list(DEFAULT_BUDGETS)
    arms = arms if arms is not None else list(ARMS)
    seeds = seeds if seeds is not None else list(DEFAULT_SEEDS)

    device = torch.device("cpu")
    if not no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"Device: {device}")
    print(f"Budgets: {budgets}   Arms: {arms}   Seeds: {seeds}")

    samples = feature_discovery(data_dir)
    train, val, test = dataset_splitting(samples, val_frac=0.15, test_frac=0.2, seed=split_seed)

    val_loader = DataLoader(
        RetinalVesselDataset(val, transform=transform_images("validation", img_size)),
        batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(
        RetinalVesselDataset(test, transform=transform_images("validation", img_size)),
        batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    results_dir = os.path.join(out_dir, "logs", "comparative")
    log_dir = os.path.join(results_dir, "placements")
    cache_dir = os.path.join(out_dir, "cache", "frangi")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "comparative_results.json")

    all_runs = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_runs = json.load(f).get("runs", [])
    done = {(r["arm"], r["n_boxes"], r["seed"]) for r in all_runs}
    if done:
        print(f"Resuming: {len(done)} runs already complete.")

    start_time = time.time()
    total_runs = len(budgets) * len(arms) * len(seeds)
    completed = len(done)

    for n_boxes in budgets:
        for arm in arms:
            for seed in seeds:
                if (arm, n_boxes, seed) in done:
                    print(f"\nSkip arm={arm} n_boxes={n_boxes} seed={seed} already done.")
                    continue

                completed += 1
                print(f"\n[{completed}/{total_runs}] arm={arm}  n_boxes={n_boxes}  seed={seed}")
                set_seed(seed)
                try:
                    if arm == "random":
                        placements = place_static(train, n_boxes, seed, False, cache_dir)
                    elif arm == "frangi":
                        placements = place_static(train, n_boxes, seed, True, cache_dir)
                    elif arm in ("entropy", "margin"):
                        placements = acquire_active(train, n_boxes, seed, arm, cache_dir, val_loader, device, img_size, bs, n_rounds, acquire_epochs, verbose=True)
                    else:
                        raise ValueError(arm)

                    save_placements(
                        placements,
                        os.path.join(log_dir, f"cmp_{arm}_n{n_boxes}_s{seed}.json"),
                        metadata={"arm": arm, "n_boxes": n_boxes, "seed": seed,
                                  "box_size": FIXED_BOX_SIZE})

                    _, best_val, test_metrics = run_one_shot(
                        train, placements, val_loader, test_loader, device,
                        img_size, bs, num_epochs, verbose=True)
                except Exception as e:
                    print(f"Error arm={arm} n_boxes={n_boxes} seed={seed} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

                print(f"Test Dice = {test_metrics['dice']:.4f}")
                all_runs.append({
                    "arm": arm, "n_boxes": n_boxes, "seed": seed,
                    "n_placed": len(placements),
                    "best_val_dice": float(best_val),
                    "final_metrics": {k: float(v) for k, v in test_metrics.items()},
                })
                with open(results_path, "w") as f:
                    json.dump({"config": {"budgets": budgets, "arms": arms, "seeds": seeds,
                                          "fixed_size": FIXED_BOX_SIZE, "img_size": img_size,
                                          "bs": bs, "num_epochs": num_epochs,
                                          "acquire_epochs": acquire_epochs,
                                          "n_rounds": n_rounds, "split_seed": split_seed},
                               "runs": all_runs}, f, indent=2)

    table = {}
    for run in all_runs:
        table.setdefault((run["arm"], run["n_boxes"]), []).append(run["final_metrics"]["dice"])

    print("\nMatched-budget comparison (mean Dice)")
    print(f"{'N':>5}  " + "  ".join(f"{a:>9s}" for a in arms))
    for n_boxes in budgets:
        row = " ".join(f"{np.mean(table.get((a, n_boxes), [float('nan')])):>9.4f}" for a in arms)
        print(f"{n_boxes:>5}  {row}")

    print(f"\nTotal wall time: {(time.time() - start_time) / 3600:.2f} h")
    print(f"All comparative results in {results_path}")
    return {"config": {"budgets": budgets, "arms": arms, "seeds": seeds}, "runs": all_runs}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./outputs_new")
    p.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS)
    p.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--num_epochs", type=int, default=80)
    p.add_argument("--acquire_epochs", type=int, default=30)
    p.add_argument("--n_rounds", type=int, default=4)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--no_cuda", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    run_comparative(data_dir=args.data_dir, out_dir=args.out_dir, budgets=args.budgets, arms=args.arms,
        seeds=args.seeds, img_size=args.img_size, bs=args.bs, num_epochs=args.num_epochs,
        acquire_epochs=args.acquire_epochs, n_rounds=args.n_rounds,
        split_seed=args.split_seed, no_cuda=args.no_cuda)
    
if __name__ == "__main__":
    main()
