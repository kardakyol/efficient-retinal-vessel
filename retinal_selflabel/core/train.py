#training engine

import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from retinal_selflabel.core.models import create_loss, create_model
from retinal_selflabel.core.utils import (
    MetricLogger,
    compute_all_metrics,
    save_checkpoint,
    visualize_predictions,
)

# cores
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_metrics = {"dice": 0.0, "iou": 0.0, "sensitivity": 0.0, "specificity": 0.0}
    n_batches = 0
    
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            batch_metrics = compute_all_metrics(probs, masks)
            for k in all_metrics:
                all_metrics[k] += batch_metrics[k]
        
        n_batches += 1
    
    results = {"loss": running_loss / n_batches}
    for k in all_metrics:
        results[k] = all_metrics[k] / n_batches
    
    return results

#evaluation on a dataset
@torch.no_grad()
def evaluate(model, dataloader, criterion, device):

    running_loss = 0.0
    all_metrics = {"dice": 0.0, "iou": 0.0, "sensitivity": 0.0, "specificity": 0.0}
    n_batches = 0
    
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        
        logits = model(images)
        loss = criterion(logits, masks)
        
        running_loss += loss.item()
        
        probs = torch.sigmoid(logits)
        batch_metrics = compute_all_metrics(probs, masks)
        for k in all_metrics:
            all_metrics[k] += batch_metrics[k]
        
        n_batches += 1
    
    results = {"loss": running_loss / max(n_batches, 1)}
    for k in all_metrics:
        results[k] = all_metrics[k] / max(n_batches, 1)
    
    return results


# training pipeline
def train_model(model, train_loader, val_loader, criterion, optimizer,
                scheduler=None, device = torch.device("cpu"), num_epochs= 100,
                patience= 15, ckpt_dir = "./outputs/checkpoints", experiment_name = "experiment"):
    
    model = model.to(device)
    logger = MetricLogger()
    
    best_dice = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    
    ckpt_path = os.path.join(ckpt_dir, experiment_name)
    os.makedirs(ckpt_path, exist_ok=True)
    
    print(f"Training: {experiment_name}")
    print(f"Epochs: {num_epochs} | Patience: {patience} | Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        # train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        logger.update("train", epoch, train_metrics)
        
        # validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        logger.update("val", epoch, val_metrics)
        
        # scheduler
        if scheduler is not None:
            scheduler.step()
        
        # early stopping
        current_dice = val_metrics["dice"]
        if current_dice > best_dice:
            best_dice = current_dice
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            
            save_checkpoint(model, optimizer, epoch, val_metrics, 
                            os.path.join(ckpt_path, "best_model.pth"), is_best=True)
        else:
            epochs_without_improvement += 1
        
        if epoch % 5 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:3d}/{num_epochs} | Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val Dice: {val_metrics['dice']:.4f} | Val IoU: {val_metrics['iou']:.4f} | "
                  f"LR: {lr:.6f} | "
                  f"{'Best' if epochs_without_improvement == 0 else ''}")
        
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}, no improvement for {patience} epochs)")
            break
    
    elapsed = time.time() - start_time
    print(f"\n Training complete in {elapsed/60:.1f} minutes, best validation Dice: {best_dice:.4f}")
    
    # restore best weights
    model.load_state_dict(best_model_wts)
    return model, logger


# experiment runners
def run_full_supervision(train_loader, val_loader, device, num_epochs = 100, lr = 1e-3,
                          patience = 15, ckpt_dir = "./outputs/checkpoints"):
    model = create_model(architecture="unet", encoder="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
    
    criterion = create_loss("bce_dice")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    model, logger = train_model(model=model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler, device=device,
        num_epochs=num_epochs, patience=patience, ckpt_dir=ckpt_dir, experiment_name="full_supervision",)
    
    return model, logger


def run_sparse_baseline(train_loader, val_loader, device, num_epochs= 150, lr = 1e-3,
                         patience = 20, ckpt_dir = "./outputs/checkpoints"):

    model = create_model(architecture="unet", encoder="resnet34", encoder_weights="imagenet",
        in_channels=3, classes=1)
    
    criterion = create_loss("bce_dice")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    model, logger = train_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        device=device, num_epochs=num_epochs, patience=patience,
        ckpt_dir=ckpt_dir, experiment_name="sparse_baseline",
    )
    
    return model, logger


# prediction utils
@torch.no_grad()
def predict_single(model, image, device, threshold):
    model.eval()
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    image = image.to(device)
    logits = model(image)
    probs = torch.sigmoid(logits)
    pred = (probs > threshold).float()
    
    return pred.squeeze().cpu().numpy()


def generate_comparison_figures(model_full, model_sparse, test_dataset, device, 
                                save_dir  = "./outputs/figures", n_samples = 5):
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(min(n_samples, len(test_dataset))):
        image_tensor, mask_tensor = test_dataset[i]
        
        # raw image 
        raw_image, raw_mask = test_dataset.get_raw(i)
        # normalize 
        raw_image = raw_image.astype(np.float32) / 255.0
        raw_mask = raw_mask.astype(np.float32)
        
        # generate predictions
        full_pred = predict_single(model_full, image_tensor, device)
        sparse_pred = predict_single(model_sparse, image_tensor, device)
        
        # resize predictions to match raw image size if needed
        import cv2
        if full_pred.shape != raw_mask.shape:
            full_pred = cv2.resize(full_pred, (raw_mask.shape[1], raw_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            sparse_pred = cv2.resize(sparse_pred, (raw_mask.shape[1], raw_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        visualize_predictions(image=raw_image, gt_mask=raw_mask, full_pred=full_pred,
            sparse_pred=sparse_pred, title=f"Test Sample {i+1}", 
            save_path=os.path.join(save_dir, f"comparison_{i+1}.png"), show=False,
        )
    
    print(f"{min(n_samples, len(test_dataset))} comparison figures saved to {save_dir}")