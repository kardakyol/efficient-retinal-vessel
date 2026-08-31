import json
import os

import torch
from torch.utils.data import DataLoader

from retinal_selflabel.core.datasets import (
    RetinalVesselDataset,
    SparseAnnotationSimulator,
    SparsePatchDataset,
    get_sparse_training_transforms,
    transform_images,
)
from retinal_selflabel.core.models import create_loss, create_model
from retinal_selflabel.core.train import evaluate, train_model
from retinal_selflabel.core.utils import set_seed

def ablation_patch_size(train, test, device, patch_sizes = [32, 64, 96, 128, 256], 
                        img_size = 512, num_epochs = 100, seed = 42, out_dir = "./outputs"):
    print("Ablation Test No.1: Patch Size (How does patch size affect the sparse baseline?)")
    
    results = {}
    
    for ps in patch_sizes:
        print(f"\nPatch size: {ps}x{ps}")
        set_seed(seed)
        
        # sparse annotator
        simulator = SparseAnnotationSimulator(samples=train, patch_size=ps,
            patches_per_image=1, min_vessel_fraction=0.01, seed=seed)
        coverage = simulator.get_annotation_coverage()
        
        # training dataset
        sparse_dataset = SparsePatchDataset(samples=train, patch_info=simulator.patch_info, patch_size=ps, 
                                            transform=get_sparse_training_transforms())
        
        val_dataset = RetinalVesselDataset(test, transform=transform_images("validation", img_size))
        
        train_loader = DataLoader(sparse_dataset, batch_size=max(2, 32 // (ps // 32)), shuffle=True, 
                                  num_workers=2, pin_memory=True,
)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True,)
        
        # train
        model = create_model("unet", "resnet34", "imagenet", 3, 1)
        criterion = create_loss("bce_dice")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
        
        model, logger = train_model(model=model, train_loader=train_loader, val_loader=val_loader,
            criterion=criterion, optimizer=optimizer, scheduler=scheduler, device=device,
            num_epochs=num_epochs, patience=20, ckpt_dir=os.path.join(out_dir, "checkpoints"),
            experiment_name=f"sparse_patch{ps}")
        
        final_metrics = evaluate(model, val_loader, criterion, device)
        
        results[ps] = { "coverage": coverage, "metrics": {k: float(v) for k, v in final_metrics.items()}}
        
        print(f" Patch {ps}x{ps}: coverage={coverage*100:.2f}%, Dice={final_metrics['dice']:.4f}, IoU={final_metrics['iou']:.4f}")
    
    # Summary table
    print("Patch Size Ablation Summary")
    print(f"  {'Patch':<10} {'Coverage':<12} {'Dice':<10} {'IoU':<10} {'Sens.':<10}")
    for ps in sorted(results.keys()):
        r = results[ps]
        print(f"{ps:<10} {r['coverage']*100:<12.2f}, {r['metrics']['dice']:<10.4f}, {r['metrics']['iou']:<10.4f},{r['metrics']['sensitivity']:<10.4f}")
    
    save_path = os.path.join(out_dir, "logs", "ablation_patch_size.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # convert keys to strings for json
    results_json = {str(k): v for k, v in results.items()}
    with open(save_path, "w") as f:
        json.dump(results_json, f, indent=2)
    
    return results

def ablation_pretrained(train, test, device,
                         patch_size = 128, img_size = 512, num_epochs = 100,
                         seed = 42, out_dir = "./outputs"):

    print("Ablation Test No.2 : Pretrained encoder vs from-scratch")
    
    results = {}
    
    for encoder_weights, label in [("imagenet", "pretrained"), (None, "scratch")]:
        for mode, label2 in [("full", "full"), ("sparse", "sparse")]:
            exp_name = f"{label}_{label2}"
            print(f"{exp_name}")
            set_seed(seed)
            
            if mode == "full":
                train_dataset = RetinalVesselDataset(train, transform=transform_images("training", img_size))
                train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
            else:
                simulator = SparseAnnotationSimulator(samples=train, patch_size=patch_size, 
                                                      patches_per_image=1, seed=seed)
                sparse_dataset = SparsePatchDataset(samples=train, patch_info=simulator.patch_info,
                                                    patch_size=patch_size, transform=get_sparse_training_transforms())
                train_loader = DataLoader(sparse_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
            
            val_dataset = RetinalVesselDataset(test, transform=transform_images("validation", img_size))
            val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
            
            model = create_model("unet", "resnet34", encoder_weights, 3, 1)
            criterion = create_loss("bce_dice")
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
            
            model, logger = train_model(model=model, train_loader=train_loader, val_loader=val_loader,
                criterion=criterion, optimizer=optimizer, scheduler=scheduler, device=device, num_epochs=num_epochs,
                patience=20, ckpt_dir=os.path.join(out_dir, "checkpoints"), experiment_name=exp_name)
            
            final_metrics = evaluate(model, val_loader, criterion, device)
            results[exp_name] = {k: float(v) for k, v in final_metrics.items()}
            
            print(f"{exp_name}: Dice={final_metrics['dice']:.4f}")
    
    print("Pretrained vs From-Scratch Ablation Summary")

    if "pretrained_full" in results and "pretrained_sparse" in results:
        gap_pretrained = results["pretrained_full"]["dice"] - results["pretrained_sparse"]["dice"]
        print(f"Pretrained: Full={results['pretrained_full']['dice']:.4f}, Sparse={results['pretrained_sparse']['dice']:.4f}, Gap={gap_pretrained:.4f}")
    
    if "scratch_full" in results and "scratch_sparse" in results:
        gap_scratch = results["scratch_full"]["dice"] - results["scratch_sparse"]["dice"]
        print(f" Scratch: Full={results['scratch_full']['dice']:.4f}, Sparse={results['scratch_sparse']['dice']:.4f}, Gap={gap_scratch:.4f}")
    
    if "pretrained_full" in results and "scratch_full" in results:
        print(f"\nGap without pretrained: {gap_scratch:.4f}")
        print(f"Gap with pretrained: {gap_pretrained:.4f}")
        print(f"Pretrained reduces gap by {gap_scratch - gap_pretrained:.4f}")
    
    save_path = os.path.join(out_dir, "logs", "ablation_pretrained.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def per_dataset_evaluation(model, all_test_samples, device, img_size= 512):

    print("Ablation Test No.3 : Per-Dataset Evaluation")
    
    datasets = set(s["dataset"] for s in all_test_samples)
    results = {}
    criterion = create_loss("bce_dice")
    
    for ds in sorted(datasets):
        ds_samples = [s for s in all_test_samples if s["dataset"] == ds]
        if len(ds_samples) == 0:
            continue
        
        ds_dataset = RetinalVesselDataset(ds_samples, transform=transform_images("validation", img_size))
        ds_loader = DataLoader(ds_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
        
        metrics = evaluate(model, ds_loader, criterion, device)
        results[ds] = {k: float(v) for k, v in metrics.items()}
        
        print(f"{ds:<10}: Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}, Sens={metrics['sensitivity']:.4f}")
    return results

def ablation_confidence_threshold(model, train, test, patch_info, patch_size,
                                    device, thresholds = [0.3, 0.5, 0.6, 0.7, 0.8], expand_px = 32,
                                    finetune_epochs = 20, img_size = 512, seed = 42, out_dir = "./outputs"):
    import copy

    from retinal_selflabel.selflabel.self_labelling import (
        IncrementalSelfLabeller,
        SpatialExpansionManager,
)
    print("Ablation Test No. 3: Confidence Threshold (How does confidence threshold affect self-labelling?)")
    
    results = {}
    
    # prepare image shapes and initial patches
    import cv2
    image_shapes = []
    initial_patches = []
    
    for info in patch_info:
        sample = train[info["sample_idx"]]
        img = cv2.imread(sample["image_path"])
        h, w = img.shape[:2]
        image_shapes.append((h, w))
        patches = [(r, c, patch_size) for (r, c) in info["patches"]]
        initial_patches.append(patches)
    
    for theta in thresholds:
        print(f"\nThreshold: {theta}")
        set_seed(seed)
        
        # fresh copy of the sparse model
        model_copy = copy.deepcopy(model)
        
        # fresh expansion manager
        manager = SpatialExpansionManager(image_shapes=image_shapes, initial_patches=initial_patches, expand_px=expand_px)
        
        # run self-labelling
        labeller = IncrementalSelfLabeller(model=model_copy, train=train, val=test,
            expansion_manager=manager, device=device, img_size=img_size, confidence_threshold=theta,
            finetune_epochs=finetune_epochs, finetune_lr=5e-4, pseudo_weight=1.0,
            max_iterations=30, ckpt_dir=os.path.join(out_dir, "checkpoints", f"theta_{theta}"),
)
        
        best_model, iteration_log = labeller.run()
        
        # get best dice from log
        best_entry = max(iteration_log, key=lambda x: x["val_dice"])
        results[theta] = {
            "best_dice": best_entry["val_dice"], "best_iou": best_entry["val_iou"], 
            "best_iteration": best_entry["iteration"], "final_coverage": iteration_log[-1]["coverage"],
            "n_iterations": len(iteration_log),
        }
        
        print(f"theta={theta}: Best Dice={best_entry['val_dice']:.4f} at iter {best_entry['iteration']}")
    
    # Summary
    print("Confidence Threshold Ablation Summary")
    print(f"  {'Threshold':<12} {'Best Dice':<12} {'Best Iter':<12} {'Coverage':<12}")
    for theta in sorted(results.keys()):
        r = results[theta]
        print(f"{theta:<12.1f} {r['best_dice']:<12.4f} {r['best_iteration']:<12d} {r['final_coverage']*100:<12.1f}%")
    
    save_path = os.path.join(out_dir, "logs", "ablation_threshold.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results_json = {str(k): v for k, v in results.items()}
    with open(save_path, "w") as f:
        json.dump(results_json, f, indent=2)
    
    return results 