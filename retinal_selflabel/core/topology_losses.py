import torch
import torch.nn as nn
import torch.nn.functional as F

# skeletonization

def soft_erode(img):
    if len(img.shape) == 4:
        p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
        p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
        return torch.min(p1, p2)
    else:
        raise ValueError(f"Expected 4D tensor got {len(img.shape)}D")

# morphological dilation
def soft_dilate(img):
    if len(img.shape) == 4:
        return F.max_pool2d(img, kernel_size=(3, 3), stride=1, padding=(1, 1))
    else:
        raise ValueError(f"Expected 4D tensor got {len(img.shape)}D")

def soft_open(img):
    return soft_dilate(soft_erode(img))

# differentiable soft skeletonization
def soft_skeleton(img, num_iter=10):
    img_orig = img.clone()
    skeleton = F.relu(img - soft_open(img))

    for _ in range(num_iter):
        img = soft_erode(img)
        delta = F.relu(img - soft_open(img))
        skeleton = skeleton + F.relu(delta - skeleton * delta)

    return skeleton

# topology precision and sensitivity
def soft_tprec(s_pred, v_gt, smooth=1.0):
    # fraction of predicted skeleton inside GT
    intersection = (s_pred * v_gt).sum(dim=(2, 3))
    skeleton_sum = s_pred.sum(dim=(2, 3))
    return (intersection + smooth) / (skeleton_sum + smooth)


def soft_tsens(s_gt, v_pred, smooth=1.0):
    # topology sensitivity
    intersection = (s_gt * v_pred).sum(dim=(2, 3))
    skeleton_sum = s_gt.sum(dim=(2, 3))
    return (intersection + smooth) / (skeleton_sum + smooth)


# soft cldiceloss
class SoftClDiceLoss(nn.Module):
    def __init__(self, num_iter = 10, smooth = 1.0):
        super().__init__()
        self.num_iter = num_iter
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        # compute soft skeletons
        s_pred = soft_skeleton(probs, self.num_iter)
        s_gt = soft_skeleton(targets, self.num_iter)

        # topology precision and sensitivity
        tprec = soft_tprec(s_pred, targets, self.smooth)
        tsens = soft_tsens(s_gt, probs, self.smooth)

        # clDice = harmonic mean of t-precision and t-sensitivity
        cl_dice = (2.0 * tprec * tsens) / (tprec + tsens + 1e-7)

        return (1.0 - cl_dice).mean()

# combined loss with warmup scheduler
class BCEClDiceLoss(nn.Module):
    def __init__(self, bce_weight = 0.5, cldice_weight = 0.5, warmup_epochs = 10, num_iter = 10):
        super().__init__()
        self.bce_weight = bce_weight
        self.cldice_weight = cldice_weight
        self.warmup_epochs = warmup_epochs
        self.cldice = SoftClDiceLoss(num_iter=num_iter)
        self.current_epoch = 0

    def set_epoch(self, epoch):
        # sets epochs
        self.current_epoch = epoch

    def get_alpha(self):
        # linear warmup
        if self.warmup_epochs <= 0:
            return 1.0
        return min(1.0, self.current_epoch / self.warmup_epochs)

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        alpha = self.get_alpha()

        if alpha > 0:
            cldice = self.cldice(logits, targets)
            total = self.bce_weight * bce + self.cldice_weight * alpha * cldice
        else:
            total = self.bce_weight * bce
            cldice = torch.tensor(0.0)

        return total

# per-pixel weighted bce + cldice for self labelling fine-tuning
class WeightedBCEClDiceLoss(nn.Module):
    def __init__(self, bce_weight= 0.5, cldice_weight = 0.5, warmup_epochs = 5, num_iter = 10):
        super().__init__()
        self.bce_weight = bce_weight
        self.cldice_weight = cldice_weight
        self.warmup_epochs = warmup_epochs
        self.cldice = SoftClDiceLoss(num_iter=num_iter)
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def get_alpha(self):
        if self.warmup_epochs <= 0:
            return 1.0
        return min(1.0, self.current_epoch / self.warmup_epochs)

    def forward(self, logits, targets, weights):
        # weighted bce
        valid = (weights > 0).float()
        n_valid = valid.sum().clamp(min=1)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        weighted_bce = (bce * weights * valid).sum() / n_valid

        # clDice on full prediction
        alpha = self.get_alpha()
        if alpha > 0:
            cldice = self.cldice(logits, targets)
            total = self.bce_weight * weighted_bce + self.cldice_weight * alpha * cldice
        else:
            total = self.bce_weight * weighted_bce

        return total

# cldice as evaluation metric
def compute_cldice_metric(pred, target, num_iter= 15):
    with torch.no_grad():
        pred_bin = (pred > 0.5).float()
        target_bin = (target > 0.5).float()

        if pred_bin.dim() == 3:
            pred_bin = pred_bin.unsqueeze(0)
            target_bin = target_bin.unsqueeze(0)
        if pred_bin.dim() == 2:
            pred_bin = pred_bin.unsqueeze(0).unsqueeze(0)
            target_bin = target_bin.unsqueeze(0).unsqueeze(0)

        s_pred = soft_skeleton(pred_bin, num_iter)
        s_gt = soft_skeleton(target_bin, num_iter)

        tprec = soft_tprec(s_pred, target_bin)
        tsens = soft_tsens(s_gt, pred_bin)

        cl_dice = (2.0 * tprec * tsens) / (tprec + tsens + 1e-7)
        return cl_dice.mean().item()