# full image sparse training

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from retinal_selflabel.core.random_box_sampler import BoxPlacement

# transforms
def get_full_image_sparse_transforms(img_size = 512, train = True):
    # image, mask and validity mask.
    if train:
        pipe = A.Compose(
            [
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
                ToTensorV2(),
            ],
            additional_targets={"validity": "mask"},
        )
    else:
        pipe = A.Compose(
            [
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
                ToTensorV2(),
            ],
            additional_targets={"validity": "mask"},
        )
    return pipe

# dataset
class FullImageSparseDataset(Dataset):
    # train per image with a per-sample validity mask
    def __init__(self, samples, placements_by_sample, img_size = 512, transform = None, include_uncovered = True):
        self.samples = samples
        self.img_size = img_size
        self.transform = transform or get_full_image_sparse_transforms(img_size, train=True)
        self.placements_by_sample = placements_by_sample

        if include_uncovered:
            self.active_indices = list(range(len(samples)))
        else:
            self.active_indices = sorted(placements_by_sample.keys())

    def __len__(self):
        return len(self.active_indices)

    def build_validity_mask(self, sample_idx, height, weight):
        m = np.zeros((height, weight), dtype=np.uint8)
        for b in self.placements_by_sample.get(sample_idx, []):
            # box coordinates
            r0 = max(0, b.row)
            c0 = max(0, b.col)
            r1 = min(height, b.row + b.size)
            c1 = min(weight, b.col + b.size)
            m[r0:r1, c0:c1] = 1
        return m

    def __getitem__(self, i):
        sample_idx = self.active_indices[i]
        s = self.samples[sample_idx]

        image = cv2.imread(s["image_path"])
        if image is None:
            raise FileNotFoundError(s["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gt = cv2.imread(s["mask_path"], cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise FileNotFoundError(s["mask_path"])
        gt = (gt > 127).astype(np.uint8)

        height, weight = image.shape[:2]
        validity = self.build_validity_mask(sample_idx, height, weight)

        transformed = self.transform(image=image, mask=gt, validity=validity)
        image_t = transformed["image"]
        gt_t = transformed["mask"].float().unsqueeze(0)
        vmask_t = transformed["validity"].float().unsqueeze(0)

        return image_t, gt_t, vmask_t

# masked loss
class MaskedBCEDiceLoss(torch.nn.Module):
    # bce + dice in a validity mask
    def __init__(self, bce_weight = 0.5, dice_weight = 0.5, dice_smooth = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice_smooth = dice_smooth
        self._bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets, validity):
        # masked bce
        bce_map = self._bce(logits, targets)
        v = validity
        total_valid = v.sum()
        if total_valid < 1.0:
            return logits.sum() * 0.0 # no supervised pixels in the batch
        bce_masked = (bce_map * v).sum() / total_valid

        # masked soft dice 
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        p = probs * v
        t = targets * v
        intersection = (p * t).sum(dim=dims)
        union = p.sum(dim=dims) + t.sum(dim=dims)
        has_valid = (v.sum(dim=dims) > 0).float()
        dice_per = (2.0 * intersection + self.dice_smooth) / (union + self.dice_smooth)
        # valid pixels
        denom = has_valid.sum().clamp(min=1.0)
        dice_loss = 1.0 - (dice_per * has_valid).sum() / denom

        return self.bce_weight * bce_masked + self.dice_weight * dice_loss
