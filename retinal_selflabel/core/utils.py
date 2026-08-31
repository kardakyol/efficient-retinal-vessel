import os
import random

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

# reproducibility
def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# metrics
def compute_dice(pred, target, smooth = 1e-6):
    pred = pred.float().view(pred.size(0), -1)
    target = target.float().view(target.size(0), -1)
    
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


def compute_iou(pred, target, smooth = 1e-6):

    pred = pred.float().view(pred.size(0), -1)
    target = target.float().view(target.size(0), -1)
    
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


def compute_sensitivity(pred, target, smooth= 1e-6):
    # true positive rate
    pred = pred.float().view(pred.size(0), -1)
    target = target.float().view(target.size(0), -1)
    
    tp = (pred * target).sum(dim=1)
    fn = ((1 - pred) * target).sum(dim=1)
    
    sensitivity = (tp + smooth) / (tp + fn + smooth)
    return sensitivity.mean().item()


def compute_specificity(pred, target, smooth= 1e-6):
    # true negative rate
    pred = pred.float().view(pred.size(0), -1)
    target = target.float().view(target.size(0), -1)
    
    tn = ((1 - pred) * (1 - target)).sum(dim=1)
    fp = (pred * (1 - target)).sum(dim=1)
    
    specificity = (tn + smooth) / (tn + fp + smooth)
    return specificity.mean().item()


def compute_all_metrics(pred, target, threshold = 0.5):
    pred_binary = (pred > threshold).float()
    
    metrics = {
        "dice": compute_dice(pred_binary, target),
        "iou": compute_iou(pred_binary, target),
        "sensitivity": compute_sensitivity(pred_binary, target),
        "specificity": compute_specificity(pred_binary, target),
    }
    
    # cldice 
    try:
        from retinal_selflabel.core.topology_losses import compute_cldice_metric
        metrics["cldice"] = compute_cldice_metric(pred, target)
    except ImportError:
        pass
    return metrics


# visualization
def visualize_sample(image, mask, title= "", save_path = None, show = True):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")
    
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")
    
    # mask on image
    overlay = image.copy()
    if len(overlay.shape) == 2:
        overlay = np.stack([overlay] * 3, axis=-1)
    vessel_mask = mask > 0.5
    overlay[vessel_mask, 0] = np.clip(overlay[vessel_mask, 0] + 0.3, 0, 1)
    overlay[vessel_mask, 1] = np.clip(overlay[vessel_mask, 1] - 0.1, 0, 1)
    overlay[vessel_mask, 2] = np.clip(overlay[vessel_mask, 2] - 0.1, 0, 1)
    
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

#patch selection on img
def visualize_patch_selection(image, mask, patch_coords, patch_size,
                               save_path = None, show = True):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(image)
    for (r, c) in patch_coords:
        rect = mpatches.Rectangle((c, r), patch_size, patch_size, linewidth=2, edgecolor="lime", facecolor="none", linestyle="--")
        axes[0].add_patch(rect)
    axes[0].set_title("Image with Patch Locations")
    axes[0].axis("off")
    
    # full mask with patch rectangles
    axes[1].imshow(mask, cmap="gray")
    for (r, c) in patch_coords:
        rect = mpatches.Rectangle((c, r), patch_size, patch_size, linewidth=2, edgecolor="lime", facecolor="none", linestyle="--")
        axes[1].add_patch(rect)
    axes[1].set_title("Mask with Patch Locations")
    axes[1].axis("off")
    
    # annotated region mask
    annotated_mask = np.zeros_like(mask)
    for (r, c) in patch_coords:
        annotated_mask[r:r + patch_size, c:c + patch_size] = \
            mask[r:r + patch_size, c:c + patch_size]
    axes[2].imshow(annotated_mask, cmap="gray")
    axes[2].set_title(f"Sparse Annotation Only ({patch_size}x{patch_size})")
    axes[2].axis("off")
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

# for gt vs full supervision vs sparse prediction.
def visualize_predictions(image, gt_mask, full_pred, sparse_pred, title= "", save_path= None, show= True):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[0].axis("off")
    
    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")
    
    axes[2].imshow(full_pred, cmap="gray")
    axes[2].set_title("Full Supervision")
    axes[2].axis("off")
    
    axes[3].imshow(sparse_pred, cmap="gray")
    axes[3].set_title("Sparse Baseline")
    axes[3].axis("off")
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


# checkpoints
def save_checkpoint(model, optimizer, epoch, metrics, path, is_best = False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = { "epoch": epoch, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None, "metrics": metrics,
    }
    torch.save(checkpoint, path)
    if is_best:
        best_path = path.replace(".pth", "_best.pth")
        torch.save(checkpoint, best_path)


def load_checkpoint(model, path, optimizer=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["metrics"]


class MetricLogger:
    def __init__(self):
        self.history = {}
    
    # for a phase and epoch
    def update(self, phase, epoch, metrics):
        if phase not in self.history:
            self.history[phase] = []
        self.history[phase].append({"epoch": epoch, **metrics})
    
    def get_best(self, phase, metric = "dice", mode = "max"):
        if phase not in self.history:
            return {}
        records = self.history[phase]
        if mode == "max":
            best = max(records, key=lambda x: x.get(metric, 0))
        else:
            best = min(records, key=lambda x: x.get(metric, float("inf")))
        return best
    
    # training loss/dice curves
    def plot_training_curves(self, save_path = None, show = True):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for phase in self.history:
            num_epochs = [r["epoch"] for r in self.history[phase]]
            
            if "loss" in self.history[phase][0]:
                losses = [r["loss"] for r in self.history[phase]]
                axes[0].plot(num_epochs, losses, label=phase)
            
            if "dice" in self.history[phase][0]:
                dices = [r["dice"] for r in self.history[phase]]
                axes[1].plot(num_epochs, dices, label=phase)
        
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Dice Score")
        axes[1].set_title("Validation Dice")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()