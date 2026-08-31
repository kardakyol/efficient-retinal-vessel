import copy
import math
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from retinal_selflabel.core.models import create_loss
from retinal_selflabel.core.utils import save_checkpoint

# expanding labelled region for each training image
class SpatialExpansionManager:
    def __init__(self, image_shapes,initial_patches, expand_px = 16):
        self.image_shapes = image_shapes
        self.expand_px = expand_px
        self.n_images = len(image_shapes)
        self.iteration = 0

        self.labelled_masks = []
        self.pseudo_labels = []
        self.is_real_gt = []

        for i, (height, weight) in enumerate(image_shapes):
            labelled = np.zeros((height, weight), dtype=np.uint8)
            pseudo = np.zeros((height, weight), dtype=np.float32)
            real_gt = np.zeros((height, weight), dtype=np.uint8)
            for (row, col, patch_size) in initial_patches[i]:
                r_end, c_end = min(row + patch_size, height), min(col + patch_size, weight)
                labelled[row:r_end, col:c_end] = 1
                real_gt[row:r_end, col:c_end] = 1
            self.labelled_masks.append(labelled)
            self.pseudo_labels.append(pseudo)
            self.is_real_gt.append(real_gt)

    def get_expansion_ring(self, image_idx):
        labelled = self.labelled_masks[image_idx]
        kernel_size = 2 * self.expand_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(labelled, kernel, iterations=1)
        return np.clip(dilated - labelled, 0, 1).astype(np.uint8)

    # accept pseudo-labels using logit-magnitude confidence
    def update_with_pseudo_labels(self, image_idx, ring_mask, probs, logits, logit_margin = 1.0, min_component_size = 10):
        ring_f = ring_mask.astype(np.float32)

        # confident positive = logit > +margin and in ring
        conf_pos = ((logits > logit_margin) * ring_f).astype(np.float32)
        # confident negative = logit < -margin and in ring
        conf_neg = ((logits < -logit_margin) * ring_f).astype(np.float32)

        # morphological cleanup of positive pseudo-labels
        if min_component_size > 1 and conf_pos.sum() > 0:
            pos_u8 = conf_pos.astype(np.uint8)
            n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(pos_u8, connectivity=8)
            for label_id in range(1, n_lab):
                if stats[label_id, cv2.CC_STAT_AREA] < min_component_size:
                    conf_pos[labels == label_id] = 0
        conf_mask = np.clip(conf_pos + conf_neg, 0, 1).astype(np.uint8)

        # update pseudo-labels for confident pixels only
        self.pseudo_labels[image_idx] = np.where(conf_pos > 0, 1.0, self.pseudo_labels[image_idx])
        self.pseudo_labels[image_idx] = np.where(conf_neg > 0, 0.0, self.pseudo_labels[image_idx])

        # only confident pixels become labelled
        self.labelled_masks[image_idx] = np.clip(self.labelled_masks[image_idx] + conf_mask, 0, 1).astype(np.uint8)
        ring_pixels = int(ring_mask.sum())
        confident_pixels = int(conf_mask.sum())
        return {"ring_pixels": ring_pixels, "confident_pixels": confident_pixels, "skipped_pixels": ring_pixels - confident_pixels, "confidence_rate": float(confident_pixels / max(ring_pixels, 1)), 
                "new_vessel_pixels": int(conf_pos.sum())}

    def get_coverage(self):
        labelled_pixels = sum(mask.sum() for mask in self.labelled_masks)
        total_pixels = sum(height * weight for height, weight in self.image_shapes)
        return float(labelled_pixels / max(total_pixels, 1))

    def is_complete(self):
        return all(self.labelled_masks[i].sum() >= height * weight for i, (height, weight) in enumerate(self.image_shapes))

    def get_combined_mask(self, idx, gt_mask):
        real_region = self.is_real_gt[idx]
        pseudo_region = np.clip(self.labelled_masks[idx] - real_region, 0, 1)
        out = np.zeros_like(gt_mask, dtype=np.float32)
        out = np.where(real_region > 0, gt_mask.astype(np.float32), out)
        out = np.where(pseudo_region > 0, self.pseudo_labels[idx], out)
        return out

    def get_label_weight_mask(self, idx, real_weight = 1.0, pseudo_weight = 1.0):
        weight_mask = np.zeros(self.image_shapes[idx], dtype=np.float32)
        weight_mask = np.where(self.is_real_gt[idx] > 0, real_weight, weight_mask)
        pseudo_region = np.clip(self.labelled_masks[idx] - self.is_real_gt[idx], 0, 1)
        weight_mask = np.where(pseudo_region > 0, pseudo_weight, weight_mask)
        return weight_mask


class SelfLabellingDataset(Dataset):
    def __init__(self, samples, manager, img_size = 512, pseudo_weight = 0.5):
        self.samples = samples
        self.manager = manager
        self.img_size = img_size
        self.pseudo_weight = pseudo_weight

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = cv2.imread(sample["image_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)
        gt = (gt > 127).astype(np.float32)

        cmask = self.manager.get_combined_mask(idx, gt)
        wmask = self.manager.get_label_weight_mask(idx, 1.0, self.pseudo_weight)

        img = cv2.resize(img, (self.img_size, self.img_size))
        cmask = cv2.resize(cmask, (self.img_size, self.img_size),
                           interpolation=cv2.INTER_NEAREST)
        wmask = cv2.resize(wmask, (self.img_size, self.img_size),
                           interpolation=cv2.INTER_NEAREST)

        img_t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        cmask_t = torch.from_numpy(cmask).float().unsqueeze(0)
        wmask_t = torch.from_numpy(wmask).float().unsqueeze(0)
        return img_t, cmask_t, wmask_t

# bce with per-pixel weights
class WeightedBCELoss(nn.Module):
    def forward(self, logits, targets, weights):
        valid = (weights > 0).float()
        n_valid = valid.sum().clamp(min=1)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return (bce * weights * valid).sum() / n_valid

# with logit confidence and best-model rollback
class IncrementalSelfLabeller:
    def __init__(self, model, train, val, expansion_manager, device, img_size, 
                 confidence_threshold = 0.7, finetune_epochs = 15, finetune_lr = 5e-4,
                 pseudo_weight = 0.5, max_iterations = 50, patience = 8, use_cldice = False, 
                 ckpt_dir = "./outputs/checkpoints/selflabel"):
        self.model = model.to(device)
        self.train = train
        self.val = val
        self.manager = expansion_manager
        self.device = device
        self.img_size = img_size
        self.finetune_epochs = finetune_epochs
        self.finetune_lr = finetune_lr
        self.pseudo_weight = pseudo_weight
        self.max_iterations = max_iterations
        self.patience = patience
        self.use_cldice = use_cldice
        self.ckpt_dir = ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)

        theta = max(min(confidence_threshold, 0.999), 0.501)
        self.logit_margin = math.log(theta / (1.0 - theta))
        self.confidence_threshold = confidence_threshold
        self.iteration_log = []
        self.best_model_wts = copy.deepcopy(model.state_dict())
        self.best_dice = 0.0

    def run(self):
        print("Incremental Self-Labelling (logit confidence + rollback)")
        print(f"Threshold: {self.confidence_threshold}, (logit margin: {self.logit_margin:.2f})")
        print(f"Expansion: {self.manager.expand_px}px | FT epochs: {self.finetune_epochs} | Pseudo wt: {self.pseudo_weight}")
        print(f"Patience: {self.patience} | Initial cov: {self.manager.get_coverage()*100:.2f}%")

        start_time = time.time()
        no_improve = 0

        for iteration in range(1, self.max_iterations + 1):
            self.manager.iteration = iteration

            # rollback to best
            self.model.load_state_dict(self.best_model_wts)

            # expand
            expansion_stats = self._expand_labels()
            cov = self.manager.get_coverage()

            # fine-tune
            train_loss = self._finetune()

            # evaluate
            val_metrics = self._evaluate()

            entry = {"iteration": iteration, "coverage": cov, "val_dice": val_metrics["dice"], "val_iou": val_metrics["iou"],
                     "val_sensitivity": val_metrics["sensitivity"], "val_specificity": val_metrics["specificity"],
                     "val_loss": val_metrics["loss"], "train_loss": train_loss, "expansion_stats": expansion_stats,
                     "per_dataset": val_metrics.get("per_dataset", {}),}
            self.iteration_log.append(entry)

            if val_metrics["dice"] > self.best_dice:
                self.best_dice = val_metrics["dice"]
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                save_checkpoint(self.model, None, iteration, val_metrics, os.path.join(self.ckpt_dir, "best.pth"))
                best_flag = "Best!"
                no_improve = 0
            else:
                best_flag = ""
                no_improve += 1

            print(f"  Iteration {iteration:3d} | Coverage: {cov*100:5.1f}% |  Dice: {val_metrics['dice']:.4f} | IoU: {val_metrics['iou']:.4f} | "
                  f"Conf: {expansion_stats['confidence_rate']:.2f} |  Skip: {expansion_stats['skipped_pixels']:7d} | {best_flag}")

            if self.manager.is_complete():
                print(f"\n Full coverage at iteration {iteration}")
                break
            if no_improve >= self.patience:
                print(f"\n  No improvement for {self.patience} iters so early stopping")
                break

        elapsed = time.time() - start_time
        print(f"\n Done in {elapsed/60:.1f} min | Best Dice: {self.best_dice:.4f}")
        self.model.load_state_dict(self.best_model_wts)
        return self.model, self.iteration_log

    def _expand_labels(self):
        self.model.eval()
        totals = {"ring_pixels": 0, "confident_pixels": 0, "skipped_pixels": 0, "new_vessel_pixels": 0}
        with torch.no_grad():
            for idx, sample in enumerate(self.train):
                ring = self.manager.get_expansion_ring(idx)
                if ring.sum() == 0:
                    continue
                img = cv2.imread(sample["image_path"])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                orig_height, orig_weight = img.shape[:2]
                img_r = cv2.resize(img, (self.img_size, self.img_size))
                img_tensor = torch.from_numpy(img_r.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)

                logits = self.model(img_tensor)
                log_np = logits.squeeze().cpu().numpy()
                prb_np = torch.sigmoid(logits).squeeze().cpu().numpy()

                log_o = cv2.resize(log_np, (orig_weight, orig_height), interpolation=cv2.INTER_LINEAR)
                prb_o = cv2.resize(prb_np, (orig_weight, orig_height), interpolation=cv2.INTER_LINEAR)

                stats = self.manager.update_with_pseudo_labels(idx, ring, prb_o, log_o, self.logit_margin)
                for key in totals:
                    totals[key] += stats.get(key, 0)

        totals["confidence_rate"] = float(totals["confident_pixels"] / max(totals["ring_pixels"], 1))
        return totals

    def _finetune(self):
        self.model.train()
        dataset = SelfLabellingDataset(self.train, self.manager, self.img_size, self.pseudo_weight)
        loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)

        # use topology-aware loss if enabled
        if self.use_cldice:
            from retinal_selflabel.core.topology_losses import WeightedBCEClDiceLoss
            criterion = WeightedBCEClDiceLoss(bce_weight=0.5, cldice_weight=0.5, warmup_epochs=5, num_iter=10)
        else:
            criterion = WeightedBCELoss()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.finetune_lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
        losses = []
        for epoch in range(self.finetune_epochs):
            if self.use_cldice:
                criterion.set_epoch(epoch)
            epoch_loss, n_batches = 0.0, 0
            for imgs, masks, wts in loader:
                imgs = imgs.to(self.device)
                masks = masks.to(self.device)
                wts = wts.to(self.device)
                optimizer.zero_grad()
                out = self.model(imgs)
                loss = criterion(out, masks, wts)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg = epoch_loss / max(n_batches, 1)
            losses.append(avg)
            scheduler.step(avg)
            if len(losses) > 5 and avg > losses[-5] * 2.0:
                break
        return float(np.mean(losses[-3:]) if losses else 0.0)

    def _evaluate(self):
      from retinal_selflabel.core.datasets import (
          RetinalVesselDataset,
          transform_images,
      )
      from retinal_selflabel.core.train import evaluate
  
      val_dataset = RetinalVesselDataset(self.val, transform=transform_images("validation", self.img_size))
      val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
      overall = evaluate(self.model, val_loader, create_loss("bce_dice"), self.device)
  
      # per-dataset breakdown
      per_dataset = {}
      datasets = sorted(set(s["dataset"] for s in self.val))
      for ds_name in datasets:
          ds_samps = [s for s in self.val if s["dataset"] == ds_name]
          if not ds_samps:
              continue
          ds_data = RetinalVesselDataset(ds_samps, transform=transform_images("validation", self.img_size))
          ds_loader = DataLoader(ds_data, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
          ds_metrics = evaluate(self.model, ds_loader, create_loss("bce_dice"), self.device)
          per_dataset[ds_name] = float(ds_metrics["dice"])
  
      overall["per_dataset"] = per_dataset
      return overall

# visualization
def plot_selflabel_progress(iteration_log, full_sup_dice=None, sparse_dice=None, save_path=None, show=True):
    import matplotlib.pyplot as plt
    iters = [entry["iteration"] for entry in iteration_log]
    dices = [entry["val_dice"] for entry in iteration_log]
    covs = [entry["coverage"] * 100 for entry in iteration_log]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(iters, dices, "b-o", ms=4, label="Self-Labelling")
    if full_sup_dice is not None:
        ax1.axhline(y=full_sup_dice, color="green", ls="--", lw=2, label=f"Full Sup ({full_sup_dice:.4f})")
    if sparse_dice is not None:
        ax1.axhline(y=sparse_dice, color="red", ls="--", lw=2, label=f"Sparse ({sparse_dice:.4f})")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Val Dice")
    ax1.set_title("Self-Labelling Progress")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2d = ax2
    ax2c = ax2.twinx()
    dice_line = ax2d.plot(iters, dices, "b-o", ms=4, label="Dice")
    cov_line = ax2c.plot(iters, covs, "r-s", ms=4, label="Coverage %")
    ax2d.set_xlabel("Iteration")
    ax2d.set_ylabel("Dice", color="blue")
    ax2c.set_ylabel("Coverage (%)", color="red")
    ax2.set_title("Dice & Coverage")
    ax2.legend(dice_line + cov_line, [line.get_label() for line in dice_line + cov_line])
    ax2d.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def visualize_expansion_process(image, gt_mask, manager, idx, save_path=None, show=True):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(image); axes[0].set_title("Input"); axes[0].axis("off")
    overlay = (image.astype(np.float32) / 255.0).copy()
    real_region = manager.is_real_gt[idx]
    pseudo_region = np.clip(manager.labelled_masks[idx] - real_region, 0, 1)
    overlay[real_region > 0, 1] = np.clip(overlay[real_region > 0, 1] + 0.3, 0, 1)
    overlay[pseudo_region > 0, 2] = np.clip(overlay[pseudo_region > 0, 2] + 0.3, 0, 1)
    axes[1].imshow(overlay); axes[1].set_title("GT(green) Pseudo(blue)"); axes[1].axis("off")
    axes[2].imshow(manager.get_combined_mask(idx, gt_mask), cmap="gray")
    axes[2].set_title("Combined Mask"); axes[2].axis("off")
    ring = manager.get_expansion_ring(idx)
    ring_overlay = overlay.copy()
    ring_overlay[ring > 0, 0] = np.clip(ring_overlay[ring > 0, 0] + 0.4, 0, 1)
    axes[3].imshow(ring_overlay); axes[3].set_title("Next Ring (red)"); axes[3].axis("off")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()