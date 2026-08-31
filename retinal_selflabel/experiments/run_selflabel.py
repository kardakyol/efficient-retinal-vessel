import argparse
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
from retinal_selflabel.core.train import (
    evaluate,
    run_full_supervision,
    run_sparse_baseline,
)
from retinal_selflabel.core.utils import set_seed
from retinal_selflabel.experiments.ablations import (
    ablation_confidence_threshold,
    ablation_patch_size,
    ablation_pretrained,
    per_dataset_evaluation,
)
from retinal_selflabel.selflabel.self_labelling import (
    IncrementalSelfLabeller,
    SpatialExpansionManager,
    plot_selflabel_progress,
    visualize_expansion_process,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--mode", type=str, default="all", choices=["selflabel_only", "ablations_only", "all"])
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--expand_px", type=int, default=16)
    parser.add_argument("--confidence_threshold", type=float, default=0.7)
    parser.add_argument("--finetune_epochs", type=int, default=15)
    parser.add_argument("--pseudo_weight", type=float, default=0.5)
    parser.add_argument("--max_iterations", type=int, default=50)
    parser.add_argument("--selflabel_patience", type=int, default=8)
    parser.add_argument("--epochs_baseline", type=int, default=100)
    parser.add_argument("--use_cldice", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--skip_patch_ablation", action="store_true")
    parser.add_argument("--skip_pretrained_ablation", action="store_true")
    parser.add_argument("--skip_threshold_ablation", action="store_true")
    return parser.parse_args()

def setup_directories(out_dir):
    dirs = {
        "checkpoints": os.path.join(out_dir, "checkpoints"),
        "figures": os.path.join(out_dir, "figures"),
        "logs": os.path.join(out_dir, "logs"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

def train_baselines(train, val, test, device, args, dirs):
    print("\nTraining Full Supervision Baseline")
    train_full = RetinalVesselDataset(
        train, transform=transform_images("training", args.img_size))
    val_full = RetinalVesselDataset(
        val, transform=transform_images("validation", args.img_size))
    test_full = RetinalVesselDataset(
        test, transform=transform_images("validation", args.img_size))
    
    loader_train_full = DataLoader(train_full, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    loader_val = DataLoader(val_full, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    loader_test = DataLoader(test_full, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    
    model_full, logger_full = run_full_supervision(loader_train_full, loader_val, device,
                                                   num_epochs=args.epochs_baseline, ckpt_dir=dirs["checkpoints"])
    
    criterion = create_loss("bce_dice")
    metrics_full = evaluate(model_full, loader_test, criterion, device)
    
    print("\nTraining Sparse Baseline")
    simulator = SparseAnnotationSimulator(train, patch_size=args.patch_size,patches_per_image=1, seed=args.seed)
    
    sparse_dataset = SparsePatchDataset(train, simulator.patch_info, patch_size=args.patch_size, transform=get_sparse_training_transforms())
    
    loader_sparse = DataLoader(sparse_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    
    model_sparse, logger_sparse = run_sparse_baseline(loader_sparse, loader_val, device, num_epochs=args.epochs_baseline + 50, ckpt_dir=dirs["checkpoints"])
    
    metrics_sparse = evaluate(model_sparse, loader_test, criterion, device)
    
    return model_full, model_sparse, metrics_full, metrics_sparse, simulator


def run_self_labelling(model_sparse, train, val, test, simulator, device, args, dirs):
    # Prepare expansion manager
    image_shapes = []
    initial_patches = []
    
    for info in simulator.patch_info:
        sample = train[info["sample_idx"]]
        img = cv2.imread(sample["image_path"])
        h, w = img.shape[:2]
        image_shapes.append((h, w))
        patches_with_size = [(r, c, args.patch_size) for (r, c) in info["patches"]]
        initial_patches.append(patches_with_size)
    
    manager = SpatialExpansionManager(image_shapes=image_shapes, initial_patches=initial_patches, expand_px=args.expand_px)
    
    # visualize initial state
    for i in range(min(3, len(train))):
        sample = train[i]
        img = cv2.imread(sample["image_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)
        
        visualize_expansion_process(img, mask, manager, i, save_path=os.path.join(dirs["figures"], "expansion_init_{i+1}.png"), show=False)
    
    # run self-labelling from sparse model
    model_sl = copy.deepcopy(model_sparse)
    
    labeller = IncrementalSelfLabeller(model=model_sl, train=train, val=val,
                                       expansion_manager=manager, device=device, img_size=args.img_size,
                                       confidence_threshold=args.confidence_threshold, finetune_epochs=args.finetune_epochs, 
                                       finetune_lr=5e-4, pseudo_weight=args.pseudo_weight, max_iterations=args.max_iterations,
                                       patience=args.selflabel_patience, use_cldice=args.use_cldice, ckpt_dir=os.path.join(dirs["checkpoints"], "selflabel"))
    best_model, iteration_log = labeller.run()
    
    # final evaluation
    val_dataset = RetinalVesselDataset(test, transform=transform_images("validation", args.img_size))
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    criterion = create_loss("bce_dice")
    metrics_sl = evaluate(best_model, val_loader, criterion, device)
    
    # visualize final expansion state
    for i in range(min(3, len(train))):
        sample = train[i]
        img = cv2.imread(sample["image_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)
        
        visualize_expansion_process(img, mask, manager, i, save_path=os.path.join(dirs["figures"], f"expansion_final_{i+1}.png"), show=False)
    
    return best_model, metrics_sl, iteration_log, manager


def main():
    args = parse_args()
    set_seed(args.seed)
    dirs = setup_directories(args.out_dir)
    
    device = torch.device("cpu")
    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"device: {device}")
    
    # data
    samples = feature_discovery(args.data_dir)
    train, val, test = dataset_splitting(
        samples, val_frac=0.15, test_frac=0.2, seed=args.seed)
    
    print("baselines test")
    
    model_full, model_sparse, metrics_full, metrics_sparse, simulator = \
        train_baselines(train, val, test, device, args, dirs)
    
    if args.mode in ("selflabel_only", "all"):
        print("self-labelling test")
        
        model_sl, metrics_sl, iteration_log, manager = run_self_labelling(model_sparse, train, val, test, simulator, device, args, dirs)
        
        plot_selflabel_progress(iteration_log, full_sup_dice=metrics_full["dice"], sparse_dice=metrics_sparse["dice"], 
                                save_path=os.path.join(dirs["figures"], "selflabel_progress.png"), show=False)
        
        print("three way comparison test")
        
        coverage = simulator.get_annotation_coverage()
        print(f"\nAnnotation coverage: {coverage*100:.2f}%")
        
        print(f"\n  {'Metric':<15} {'Full Sup.':<12} {'Sparse':<12} {'Self-Label':<12} {'SL Gain':<12}")
        for metric in ["dice", "iou", "sensitivity", "specificity"]:
            f_val = metrics_full[metric]
            s_val = metrics_sparse[metric]
            sl_val = metrics_sl[metric]
            gain = sl_val - s_val
            print(f"  {metric:<15} {f_val:<12.4f} {s_val:<12.4f} {sl_val:<12.4f} {gain:<12.4f}")
        
        print("\nPer-Dataset Results")
        for model_obj, name in [(model_full, "Full"), (model_sparse, "Sparse"), (model_sl, "Self-Label")]:
            print(f"\n  [{name}]")
            per_dataset_evaluation(model_obj, test, device, args.img_size)
        
        all_results = {
            "full_supervision": {k: float(v) for k, v in metrics_full.items()},
            "sparse_baseline": {k: float(v) for k, v in metrics_sparse.items()},
            "self_labelling": {k: float(v) for k, v in metrics_sl.items()},
            "config": { "patch_size": args.patch_size, "expansion_pixels": args.expand_px,
                       "confidence_threshold": args.confidence_threshold, "pseudo_weight": args.pseudo_weight,
                       "annotation_coverage": float(coverage), "n_iterations": len(iteration_log)}, "iteration_log": iteration_log}
        
        with open(os.path.join(dirs["logs"], "full_results.json"), "w") as f:
            json.dump(all_results, f, indent=2, default=str)
    
    if args.mode in ("ablations_only", "all"):
        print("ablations test")
        
        if not args.skip_patch_ablation:
            ablation_patch_size(train, test, device, patch_sizes=[32, 64, 96, 128], num_epochs=args.epochs_baseline, seed=args.seed, out_dir=args.out_dir)
        
        if not args.skip_pretrained_ablation:
            ablation_pretrained(train, test, device, patch_size=args.patch_size,
                                num_epochs=args.epochs_baseline, seed=args.seed,
                                out_dir=args.out_dir)
        
        if not args.skip_threshold_ablation and args.mode in ("all",):
            ablation_confidence_threshold(model=model_sparse, train=train, test=test,
                                          patch_info=simulator.patch_info, patch_size=args.patch_size,
                                          device=device, thresholds=[0.3, 0.5, 0.7], 
                                          expand_px=args.expand_px, finetune_epochs=min(15, args.finetune_epochs),
                                          img_size=args.img_size, seed=args.seed, out_dir=args.out_dir)
    
    print("All experiments completed")


if __name__ == "__main__":
    main()