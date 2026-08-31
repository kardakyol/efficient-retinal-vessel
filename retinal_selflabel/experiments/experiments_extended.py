import copy
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from retinal_selflabel.core.datasets import (
    RetinalVesselDataset,
    SparseAnnotationSimulator,
    SparsePatchDataset,
    feature_discovery,
    get_sparse_training_transforms,
    transform_images,
    sample_splitting,
    dataset_splitting,
)
from retinal_selflabel.core.models import create_loss
from retinal_selflabel.core.train import evaluate, run_sparse_baseline
from retinal_selflabel.core.utils import set_seed
from retinal_selflabel.selflabel.self_labelling import (
    IncrementalSelfLabeller,
    SpatialExpansionManager,
)

# expansion strategy comparison
class RandomExpansionManager:

    def __init__(self, image_shapes, initial_patches, expansion_fraction=0.005, seed=42):
        self.image_shapes = image_shapes
        self.expansion_fraction = expansion_fraction
        self.n_images = len(image_shapes)
        self.iteration = 0
        self.rng = np.random.RandomState(seed)

        self.labelled_masks = []
        self.pseudo_labels = []
        self.is_real_gt = []

        for i, (h, w) in enumerate(image_shapes):
            labelled = np.zeros((h, w), dtype=np.uint8)
            pseudo = np.zeros((h, w), dtype=np.float32)
            real_gt = np.zeros((h, w), dtype=np.uint8)
            for (r, c, ps) in initial_patches[i]:
                r_end, c_end = min(r + ps, h), min(c + ps, w)
                labelled[r:r_end, c:c_end] = 1
                real_gt[r:r_end, c:c_end] = 1
            self.labelled_masks.append(labelled)
            self.pseudo_labels.append(pseudo)
            self.is_real_gt.append(real_gt)

    # random sampling
    def get_expansion_ring(self, image_idx):
        h, w = self.image_shapes[image_idx]
        labelled = self.labelled_masks[image_idx]
        unlabelled = (1 - labelled).astype(np.float32)

        n_to_add = int(h * w * self.expansion_fraction)
        if n_to_add == 0:
            n_to_add = 1

        # unlabelled pixel coordinates
        ys, xs = np.where(unlabelled > 0)
        if len(ys) == 0:
            return np.zeros((h, w), dtype=np.uint8)

        n_to_add = min(n_to_add, len(ys))
        indices = self.rng.choice(len(ys), size=n_to_add, replace=False)

        ring = np.zeros((h, w), dtype=np.uint8)
        ring[ys[indices], xs[indices]] = 1
        return ring

    def update_with_pseudo_labels(self, image_idx, ring_mask, probs, logits, 
                                  logit_margin=1.0, min_component_size=10):
        ring_f = ring_mask.astype(np.float32)
        conf_pos = ((logits > logit_margin) * ring_f).astype(np.float32)
        conf_neg = ((logits < -logit_margin) * ring_f).astype(np.float32)
        conf_mask = np.clip(conf_pos + conf_neg, 0, 1).astype(np.uint8)

        self.pseudo_labels[image_idx] = np.where(
            conf_pos > 0, 1.0, self.pseudo_labels[image_idx])
        self.pseudo_labels[image_idx] = np.where(
            conf_neg > 0, 0.0, self.pseudo_labels[image_idx])
        self.labelled_masks[image_idx] = np.clip(
            self.labelled_masks[image_idx] + conf_mask, 0, 1).astype(np.uint8)

        rp = int(ring_mask.sum())
        cp = int(conf_mask.sum())
        return {"ring_pixels": rp, "confident_pixels": cp, "skipped_pixels": rp - cp,
                "confidence_rate": float(cp / max(rp, 1)), "new_vessel_pixels": int(conf_pos.sum())}

    def get_coverage(self):
        total_l = sum(m.sum() for m in self.labelled_masks)
        total_a = sum(h * w for h, w in self.image_shapes)
        return float(total_l / max(total_a, 1))

    def is_complete(self):
        return all(self.labelled_masks[i].sum() >= h * w for i, (h, w) in enumerate(self.image_shapes))

    def get_combined_mask(self, idx, gt_mask):
        rr = self.is_real_gt[idx]
        pr = np.clip(self.labelled_masks[idx] - rr, 0, 1)
        out = np.zeros_like(gt_mask, dtype=np.float32)
        out = np.where(rr > 0, gt_mask.astype(np.float32), out)
        out = np.where(pr > 0, self.pseudo_labels[idx], out)
        return out

    def get_label_weight_mask(self, idx, real_weight=1.0, pseudo_weight=1.0):
        w = np.zeros(self.image_shapes[idx], dtype=np.float32)
        w = np.where(self.is_real_gt[idx] > 0, real_weight, w)
        pr = np.clip(self.labelled_masks[idx] - self.is_real_gt[idx], 0, 1)
        w = np.where(pr > 0, pseudo_weight, w)
        return w

# compare ring vs random expansion
def run_expansion_comparison(train, val, test, device, patch_size=128, img_size=512, seed=42, out_dir="./outputs"):
    print("Experiment: gradual vs random expansion")

    set_seed(seed)

    # train shared sparse baseline
    simulator = SparseAnnotationSimulator(train, patch_size=patch_size, patches_per_image=1, seed=seed)

    sparse_ds = SparsePatchDataset(train, simulator.patch_info, patch_size=patch_size, transform=get_sparse_training_transforms())
    val_ds = RetinalVesselDataset(val, transform=transform_images("validation", img_size))

    sparse_loader = DataLoader(sparse_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    test_ds = RetinalVesselDataset( test, transform=transform_images("validation", img_size))
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    model_sparse, _ = run_sparse_baseline(sparse_loader, val_loader, device, num_epochs=150, ckpt_dir=os.path.join(out_dir, "checkpoints"))

    criterion = create_loss("bce_dice")
    sparse_metrics = evaluate(model_sparse, test_loader, criterion, device)

    # prepare image shapes and patches
    image_shapes, initial_patches = [], []
    for info in simulator.patch_info:
        s = train[info["sample_idx"]]
        img = cv2.imread(s["image_path"])
        h, w = img.shape[:2]
        image_shapes.append((h, w))
        initial_patches.append([(r, c, patch_size) for r, c in info["patches"]])

    results = {"sparse_baseline": {k: float(v) for k, v in sparse_metrics.items()}}

    for strategy_name, manager_cls, kwargs in [
        ("gradual", SpatialExpansionManager, {"expansion_pixels": 16}),
        ("random", RandomExpansionManager, {"expansion_fraction": 0.005, "seed": seed}),
    ]:
        print(f"\n{strategy_name} expansion")
        set_seed(seed)

        manager = manager_cls(image_shapes=image_shapes, initial_patches=initial_patches, **kwargs)

        model_copy = copy.deepcopy(model_sparse)

        labeller = IncrementalSelfLabeller(model=model_copy, train=train, val=val,
                                           expansion_manager=manager, device=device,
                                           img_size=img_size, confidence_threshold=0.7,
                                           finetune_epochs=15, pseudo_weight=0.5,
                                           max_iterations=15, patience=8, ckpt_dir=os.path.join(out_dir, "checkpoints", f"sl_{strategy_name}"))

        best_model, iteration_log = labeller.run()
        final_metrics = evaluate(best_model, test_loader, criterion, device)

        results[strategy_name] = {
            "metrics": {k: float(v) for k, v in final_metrics.items()},
            "best_dice": max(e["val_dice"] for e in iteration_log),
            "iterations": len(iteration_log)}

    print("Expansion Strategy Comparison Results")
    print(f"  {'Strategy':<15} {'Dice':<10} {'Gain':<10}")
    sp_dice = results["sparse_baseline"]["dice"]
    print(f"  {'Sparse':<15} {sp_dice:<10.4f} {'---':<10}")
    for s in ["gradual", "random"]:
        d = results[s]["best_dice"]
        print(f"  {s:<15} {d:<10.4f} {d - sp_dice:+.4f}")

    save_path = os.path.join(out_dir, "logs", "expansion_comparison.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_cross_dataset(all_samples, device, patch_size=128, img_size=512, seed=42, out_dir="./outputs"):
    print("Experiment: cross-dataset generalization")

    datasets = sorted(set(s["dataset"] for s in all_samples))
    results = {}
    criterion = create_loss("bce_dice")

    for train_ds in datasets:
        train_samps = [s for s in all_samples if s["dataset"] == train_ds]
        test_samps = [s for s in all_samples if s["dataset"] != train_ds]

        if len(train_samps) < 5 or len(test_samps) < 3:
            continue

        # select the model on a held-out slice of the training dataset
        train_samps, sel_val_samps = sample_splitting(
            train_samps, test_frac=0.2, seed=seed)
        sel_val_loader = DataLoader(RetinalVesselDataset(sel_val_samps, transform=transform_images("validation", img_size)),
                                    batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

        print(f"\n Train: {train_ds} ({len(train_samps)}) | Test: others ({len(test_samps)})")
        set_seed(seed)

        # Sparse baseline
        sim = SparseAnnotationSimulator(train_samps, patch_size=patch_size, patches_per_image=1, seed=seed)
        sp_ds = SparsePatchDataset(train_samps, sim.patch_info, patch_size=patch_size, transform=get_sparse_training_transforms())
        val_ds = RetinalVesselDataset(test_samps, transform=transform_images("validation", img_size))

        sp_loader = DataLoader(sp_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

        model, _ = run_sparse_baseline(sp_loader, sel_val_loader, device, num_epochs=100, ckpt_dir=os.path.join(out_dir, "checkpoints"))

        metrics = evaluate(model, val_loader, criterion, device)

        # Per-test-dataset breakdown
        per_ds = {}
        for test_ds in datasets:
            if test_ds == train_ds:
                continue
            ds_samps = [s for s in all_samples if s["dataset"] == test_ds]
            if len(ds_samps) == 0:
                continue
            ds_data = RetinalVesselDataset(
                ds_samps, transform=transform_images("validation", img_size))
            ds_loader = DataLoader(ds_data, batch_size=4, shuffle=False,
                                   num_workers=2, pin_memory=True)
            ds_metrics = evaluate(model, ds_loader, criterion, device)
            per_ds[test_ds] = {k: float(v) for k, v in ds_metrics.items()}

        results[train_ds] = {
            "overall": {k: float(v) for k, v in metrics.items()},
            "per_dataset": per_ds,
        }

    # Summary
    print("Cross-Dataset Results")
    for train_ds, res in results.items():
        print(f"\n  Train: {train_ds}")
        for test_ds, m in res["per_dataset"].items():
            print(f"Test {test_ds}: Dice={m['dice']:.4f}")

    save_path = os.path.join(out_dir, "logs", "cross_dataset.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_annotation_budget(train, val, test, device, img_size=512, seed=42, out_dir="./outputs"):
    print("Experiment: Annotation-Budget Curve")

    # budget configs: (patch_size, patches_per_image)
    configs = [
        (64, 1),    
        (128, 1), 
        (128, 2),
        (128, 3), 
        (256, 1),   
        (256, 2),  
    ]

    results = {}
    criterion = create_loss("bce_dice")

    val_ds = RetinalVesselDataset(val, transform=transform_images("validation", img_size))
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    test_ds = RetinalVesselDataset(test, transform=transform_images("validation", img_size))
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    print("\nFull supervision reference")
    set_seed(seed)
    full_ds = RetinalVesselDataset(train, transform=transform_images("training", img_size))
    full_loader = DataLoader(full_ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    from retinal_selflabel.core.train import run_full_supervision
    model_full, _ = run_full_supervision(
        full_loader, val_loader, device, num_epochs=100,
        ckpt_dir=os.path.join(out_dir, "checkpoints"))
    full_metrics = evaluate(model_full, test_loader, criterion, device)
    results["full_supervision"] = { "coverage": 100.0, "metrics": {k: float(v) for k, v in full_metrics.items()}}

    for ps, npp in configs:
        label = f"ps{ps}_n{npp}"
        print(f"\n{label}: patch={ps}x{ps}, n_patches={npp}")
        set_seed(seed)

        sim = SparseAnnotationSimulator(train, patch_size=ps, patches_per_image=npp, seed=seed)
        coverage = sim.get_annotation_coverage()

        sp_ds = SparsePatchDataset(train, sim.patch_info, patch_size=ps, transform=get_sparse_training_transforms())
        sp_loader = DataLoader(sp_ds, batch_size=max(2, 32 // max(1, ps // 32)), shuffle=True, num_workers=2, pin_memory=True)

        model, _ = run_sparse_baseline(sp_loader, val_loader, device, num_epochs=150, ckpt_dir=os.path.join(out_dir, "checkpoints"))
        metrics = evaluate(model, test_loader, criterion, device)

        results[label] = {"patch_size": ps, "patches_per_image": npp, "coverage": coverage * 100, "metrics": {k: float(v) for k, v in metrics.items()}}

        print(f"Coverage: {coverage*100:.2f}% | Dice: {metrics['dice']:.4f}")

    # Summary
    print("Annotation Budget Curve Results")
    print(f"  {'Config':<15} {'Coverage %':<12} {'Dice':<10} {'IoU':<10}")
    for key in sorted(results.keys(), key=lambda k: results[k].get("coverage", 999)):
        r = results[key]
        cov = r.get("coverage", 100)
        d = r["metrics"]["dice"]
        iou = r["metrics"]["iou"]
        print(f"  {key:<15} {cov:<12.2f} {d:<10.4f} {iou:<10.4f}")

    save_path = os.path.join(out_dir, "logs", "annotation_budget.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


# Plot annotation budget curve
def plot_annotation_budget(results, save_path=None, show=True):
    import matplotlib.pyplot as plt

    coverages = []
    dices = []
    labels = []

    for key, r in sorted(results.items(), key=lambda x: x[1].get("coverage", 999)):
        if key == "full_supervision":
            continue
        coverages.append(r["coverage"])
        dices.append(r["metrics"]["dice"])
        labels.append(key)

    full_dice = results.get("full_supervision", {}).get("metrics", {}).get("dice", None)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(coverages, dices, "b-o", markersize=8, linewidth=2, label="Sparse Baseline")

    if full_dice is not None:
        ax.axhline(y=full_dice, color="green", ls="--", lw=2, label=f"Full Supervision ({full_dice:.4f})")

    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (coverages[i], dices[i]), textcoords="offset points", xytext=(5, 10),fontsize=8, alpha=0.7)

    ax.set_xlabel("Annotation Coverage (%)", fontsize=13)
    ax.set_ylabel("Validation Dice Score", fontsize=13)
    ax.set_title("How Little Annotation Do You Need?", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

def run_lambda_ablation(train, val, test, device, patch_size=128, img_size=512, seed=42, out_dir="./outputs"):
    print("Experiment: Lambda (λ) Ablation for mixed supervision")

    set_seed(seed)

    simulator = SparseAnnotationSimulator(train, patch_size=patch_size, patches_per_image=1, seed=seed)

    sparse_ds = SparsePatchDataset(train, simulator.patch_info, patch_size=patch_size, transform=get_sparse_training_transforms())
    val_ds = RetinalVesselDataset(val, transform=transform_images("validation", img_size))

    sparse_loader = DataLoader(sparse_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    test_ds = RetinalVesselDataset(test, transform=transform_images("validation", img_size))
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    model_sparse, _ = run_sparse_baseline(sparse_loader, val_loader, device, num_epochs=150, ckpt_dir=os.path.join(out_dir, "checkpoints"))

    criterion = create_loss("bce_dice")
    sparse_metrics = evaluate(model_sparse, test_loader, criterion, device)

    # prepare expansion manager
    image_shapes, initial_patches = [], []
    for info in simulator.patch_info:
        s = train[info["sample_idx"]]
        img = cv2.imread(s["image_path"])
        h, w = img.shape[:2]
        image_shapes.append((h, w))
        initial_patches.append([(r, c, patch_size) for r, c in info["patches"]])

    lambdas = [0.1, 0.3, 0.5, 0.7, 1.0]
    results = {
        "sparse_baseline": {k: float(v) for k, v in sparse_metrics.items()},
        "lambda_results": {},
    }

    for lam in lambdas:
        print(f"\nλ = {lam}")
        set_seed(seed)

        manager = SpatialExpansionManager(
            image_shapes=image_shapes,
            initial_patches=initial_patches,
            expand_px=16)

        model_copy = copy.deepcopy(model_sparse)

        labeller = IncrementalSelfLabeller(model=model_copy, train=train, val=val,
                                           expansion_manager=manager, device=device, img_size=img_size, 
                                           confidence_threshold=0.7, finetune_epochs=15, pseudo_weight=lam,
                                           max_iterations=15, patience=8, ckpt_dir=os.path.join(out_dir, "checkpoints", f"lambda_{lam}"))

        best_model, iteration_log = labeller.run()
        best_dice = max(e["val_dice"] for e in iteration_log)
        best_entry = max(iteration_log, key=lambda x: x["val_dice"])

        results["lambda_results"][str(lam)] = {
            "best_dice": best_dice,
            "best_iou": best_entry["val_iou"],
            "best_sensitivity": best_entry["val_sensitivity"],
            "best_specificity": best_entry["val_specificity"],
            "best_iteration": best_entry["iteration"],
            "n_iterations": len(iteration_log),
        }

    print("Lambda Ablation Summary Results")
    sp_d = results["sparse_baseline"]["dice"]
    print(f" Sparse baseline Dice: {sp_d:.4f}")
    print(f"\n  {'λ':<8} {'Dice':<10} {'Gain':<10} {'Sens.':<10} {'Best Iter':<10}")
    for lam in lambdas:
        r = results["lambda_results"][str(lam)]
        print(f"{lam:<8.1f} {r['best_dice']:<10.4f} {r['best_dice']-sp_d:+10.4f} "
      f"{r['best_sensitivity']:<10.4f} {r['best_iteration']:<10d}")

    save_path = os.path.join(out_dir, "logs", "lambda_ablation.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_entropy_error_analysis(train, val, test, device, patch_size=128, img_size=512, seed=42, out_dir="./outputs"):
    print("Experiment : Entropy ve Error Correlation")

    set_seed(seed)

    # Train sparse baseline
    simulator = SparseAnnotationSimulator(train, patch_size=patch_size, patches_per_image=1, seed=seed)

    sparse_ds = SparsePatchDataset(train, simulator.patch_info, patch_size=patch_size, transform=get_sparse_training_transforms())
    val_ds = RetinalVesselDataset(val, transform=transform_images("validation", img_size))

    sparse_loader = DataLoader(sparse_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    test_ds = RetinalVesselDataset(test, transform=transform_images("validation", img_size))
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    model, _ = run_sparse_baseline(sparse_loader, val_loader, device, num_epochs=150, ckpt_dir=os.path.join(out_dir, "checkpoints"))

    # collecting per-pixel entropy and error
    model.eval()
    all_entropies = []
    all_errors = []

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            # binary entropy
            eps = 1e-7
            p = probs.clamp(eps, 1 - eps)
            entropy = -(p * p.log() + (1 - p) * (1 - p).log())
            # normalize to [0, 1]
            entropy = entropy / 0.6931

            # per-pixel error
            pred_binary = (probs > 0.5).float()
            error = (pred_binary - masks).abs()

            all_entropies.append(entropy.cpu().numpy().flatten())
            all_errors.append(error.cpu().numpy().flatten())

    all_entropies = np.concatenate(all_entropies)
    all_errors = np.concatenate(all_errors)

    # bin by entropy and compute mean error per bin
    n_bins = 20
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_mean_errors = []
    bin_counts = []

    for i in range(n_bins):
        mask = (all_entropies >= bin_edges[i]) & (all_entropies < bin_edges[i + 1])
        count = mask.sum()
        bin_counts.append(int(count))
        if count > 0:
            bin_mean_errors.append(float(all_errors[mask].mean()))
        else:
            bin_mean_errors.append(0.0)

    # compute Pearson correlation on non-empty bins
    valid = [i for i in range(n_bins) if bin_counts[i] > 100]
    if len(valid) > 2:
        from scipy import stats as sp_stats
        x = np.array([bin_centers[i] for i in valid])
        y = np.array([bin_mean_errors[i] for i in valid])
        corr, p_value = sp_stats.pearsonr(x, y)
    else:
        corr, p_value = 0.0, 1.0

    print(f"\n Pearson correlation: r = {corr:.4f}, p = {p_value:.6f}")
    if corr > 0.7 and p_value < 0.05:
        print("Strong positive correlation")
    elif corr > 0.4:
        print("Moderate positive correlation")
    else:
        print(" Weak correlation")

    # pixel distribution across entropy bins
    total_pixels = len(all_entropies)
    low_entropy = (all_entropies < 0.2).sum()
    mid_entropy = ((all_entropies >= 0.2) & (all_entropies < 0.8)).sum()
    high_entropy = (all_entropies >= 0.8).sum()
    print(f"\n  Pixel distribution:")
    print(f"Low entropy  (<0.2): {low_entropy/total_pixels*100:.1f}% - error rate: {all_errors[all_entropies < 0.2].mean():.4f}")
    print(f"Mid entropy  (0.2-0.8): {mid_entropy/total_pixels*100:.1f}%  - error rate: {all_errors[(all_entropies >= 0.2) & (all_entropies < 0.8)].mean():.4f}")
    if high_entropy > 0:
        print(f"High entropy (>0.8): {high_entropy/total_pixels*100:.1f}%  - error rate: {all_errors[all_entropies >= 0.8].mean():.4f}")

    results = {"correlation": float(corr), "p_value": float(p_value), "bin_centers": [float(x) for x in bin_centers],
               "bin_mean_errors": bin_mean_errors, "bin_counts": bin_counts, 
               "pixel_distribution": {
                   "low_entropy_pct": float(low_entropy / total_pixels * 100),
                   "mid_entropy_pct": float(mid_entropy / total_pixels * 100),
                   "high_entropy_pct": float(high_entropy / total_pixels * 100)},
    }

    save_path = os.path.join(out_dir, "logs", "entropy_error.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def plot_entropy_error(results, save_path=None, show=True):
    import matplotlib.pyplot as plt

    centers = results["bin_centers"]
    errors = results["bin_mean_errors"]
    counts = results["bin_counts"]
    corr = results["correlation"]
    p_val = results["p_value"]

    valid = [(c, e, n) for c, e, n in zip(centers, errors, counts) if n > 100]
    if not valid:
        print("not enough data for entropy-error plot")
        return

    x = [v[0] for v in valid]
    y = [v[1] for v in valid]
    sizes = [min(200, max(20, v[2] / 1000)) for v in valid]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.scatter(x, y, s=sizes, alpha=0.7, c="steelblue", edgecolors="navy")
    if len(x) > 2:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(x), max(x), 100)
        ax1.plot(x_line, p(x_line), "r--", lw=2, alpha=0.8)

    ax1.set_xlabel("Normalized Prediction Entropy", fontsize=12)
    ax1.set_ylabel("Mean Segmentation Error Rate", fontsize=12)
    ax1.set_title(f"Entropy vs Error (r={corr:.3f}, p={p_val:.4f})", fontsize=13)
    ax1.grid(True, alpha=0.3)

    # entropy histogram with error overlay
    ax2_hist = ax2
    ax2_err = ax2.twinx()

    all_centers = results["bin_centers"]
    all_counts = results["bin_counts"]
    all_errors_vals = results["bin_mean_errors"]
    width = all_centers[1] - all_centers[0] if len(all_centers) > 1 else 0.05

    bars = ax2_hist.bar(all_centers, all_counts, width=width * 0.8,
                         alpha=0.3, color="steelblue", label="Pixel count")
    line = ax2_err.plot(all_centers, all_errors_vals, "r-o", ms=4,
                         lw=2, label="Error rate")

    ax2_hist.set_xlabel("Normalized Entropy", fontsize=12)
    ax2_hist.set_ylabel("Pixel Count", fontsize=12, color="steelblue")
    ax2_err.set_ylabel("Error Rate", fontsize=12, color="red")
    ax2.set_title("Entropy Distribution & Error", fontsize=13)

    h1, l1 = ax2_hist.get_legend_handles_labels()
    h2, l2 = ax2_err.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=10)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


# Main runner

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extended Experiments")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--experiment", type=str, default="all", choices=["expansion", "cross_dataset", "budget", "lambda", "entropy", "all"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cpu")
    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"Info Device: {device}")

    samples = feature_discovery(args.data_dir)
    train, val, test = dataset_splitting(samples, val_frac=0.15, test_frac=0.2, seed=args.seed)

    if args.experiment in ("expansion", "all"):
        run_expansion_comparison(train, val, test, device, out_dir=args.out_dir, seed=args.seed)

    if args.experiment in ("cross_dataset", "all"):
        run_cross_dataset(samples, device, out_dir=args.out_dir, seed=args.seed)

    if args.experiment in ("budget", "all"):
        budget_results = run_annotation_budget(train, val, test, device, out_dir=args.out_dir, seed=args.seed)
        plot_annotation_budget(budget_results, save_path=os.path.join(args.out_dir, "figures", "annotation_budget.png"), show=False)

    if args.experiment in ("lambda", "all"):
        run_lambda_ablation(train, val, test, device, out_dir=args.out_dir, seed=args.seed)

    if args.experiment in ("entropy", "all"):
        entropy_results = run_entropy_error_analysis(train, val, test, device, out_dir=args.out_dir, seed=args.seed)
        plot_entropy_error(entropy_results, save_path=os.path.join(args.out_dir, "figures", "entropy_vs_error.png"), show=False)

    print("All extended exps completed")

if __name__ == "__main__":
    main()