import os
import sys

import cv2
import numpy as np

from retinal_selflabel.core.datasets import feature_discovery, sample_splitting
from retinal_selflabel.core.frangi_density import (
    compute_density_map,
    resolve_config,
    sample_position_from_density,
)

DATASET_ORDER = ["DRIVE", "CHASE", "HRF"]
PRETTY = {"DRIVE": "DRIVE", "CHASE": "CHASE_DB1", "HRF": "HRF"}


# split
def get_split(root_dir, seed=42, test_frac=0.2):
    if not os.path.isdir(root_dir):
        sys.exit(f"data root '{root_dir}' not found.")
    samples = feature_discovery(root_dir)
    train, test = sample_splitting(samples, test_frac=test_frac, seed=seed)
    return train, test


# image io
def load_rgb(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_gt(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise FileNotFoundError(path)
    return g > 127


def center_square(arr):
    h, w = arr.shape[:2]
    s = min(h, w)
    r0, c0 = (h - s) // 2, (w - s) // 2
    return arr[r0:r0 + s, c0:c0 + s]


# representative training images
def by_dataset(samples):
    out = {}
    for s in samples:
        out.setdefault(s["dataset"], []).append(s)
    for k in out:
        out[k] = sorted(out[k], key=lambda d: d["id"])
    return out

def pick_demo(samples, dataset="CHASE", index=0):
    grp = by_dataset(samples).get(dataset, [])
    if not grp:
        grp = sorted(samples, key=lambda d: d["id"])
    return grp[index % len(grp)]


# real box placement
def density_for(sample):
    cfg = resolve_config(sample["dataset"])
    return compute_density_map(sample["image_path"], cfg)

# list of non-overlapping density-weighted boxes.
def place_boxes(density, n, seed=42, min_size=32, max_size=256, fixed_size=None):
    rng = np.random.default_rng(seed)
    h, w = density.shape
    forbidden = np.zeros((h, w), dtype=np.uint8)
    out = []
    attempts = 0
    while len(out) < n and attempts < n * 40:
        attempts += 1
        size = fixed_size if fixed_size else int(rng.integers(min_size, max_size + 1))
        size = min(size, h, w)
        pos = sample_position_from_density(density, size, rng, forbidden_mask=forbidden)
        if pos is None:
            continue
        r, c = pos
        forbidden[r:r + size, c:c + size] = 1
        out.append((r, c, size))
    return out

def union_mask(shape, boxes):
    m = np.zeros(shape, dtype=np.uint8)
    for r, c, s in boxes:
        m[r:r + s, c:c + s] = 1
    return m


# frangi
def frangi_response(sample):
    from skimage.filters import frangi
    cfg = resolve_config(sample["dataset"])
    bgr = cv2.imread(sample["image_path"])
    green = bgr[:, :, 1].astype(np.float32) / 255.0
    sigmas = np.linspace(cfg.sigma_min, cfg.sigma_max, cfg.n_scales)
    v = frangi(green, sigmas=sigmas, black_ridges=cfg.black_ridges).astype(np.float32)
    p99 = float(np.percentile(v, 99.0)) or 1.0
    return np.clip(v / p99, 0.0, 1.0)


# matrix trace of one draw.
def sampling_trace(density_small, placed, cand_size, seed=1):
    h, w = density_small.shape
    forbidden = np.zeros((h, w), dtype=np.uint8)
    for (r, c, s) in placed:
        forbidden[max(0, r):r + s, max(0, c):c + s] = 1

    k = max(1, int(cand_size))
    score = cv2.boxFilter(density_small.astype(np.float32), ddepth=-1, ksize=(k, k), anchor=(0, 0),
                        normalize=True, borderType=cv2.BORDER_ISOLATED)
    overlap = cv2.boxFilter(forbidden.astype(np.float32), ddepth=-1, ksize=(k, k), anchor=(0, 0),
                            normalize=False, borderType=cv2.BORDER_ISOLATED)
    nh, nw = h - k + 1, w - k + 1
    score = score[:nh, :nw].copy()
    overlap = overlap[:nh, :nw].copy()
    valid = score.copy()
    valid[overlap > 0.0] = 0.0

    rng = np.random.default_rng(seed)
    flat = valid.ravel()
    total = float(flat.sum())
    if total <= 0:
        chosen = None
    else:
        csum = np.cumsum(flat, dtype=np.float64)
        idx = int(np.searchsorted(csum, rng.random() * csum[-1], side="right"))
        idx = min(idx, flat.size - 1)
        chosen = (idx // nw, idx % nw)

    forbidden_after = forbidden.copy()
    if chosen is not None:
        r, c = chosen
        forbidden_after[r:r + k, c:c + k] = 1
    return {
        "forbidden_before": forbidden,
        "overlap_count": overlap,
        "valid_weights": valid,
        "chosen": chosen,
        "cand_size": k,
        "forbidden_after": forbidden_after,
    }


# real expansion mechanism, gt-proxy logits
def selflabel_snapshots(sample, init_boxes, iterations=(0, 1, 3, 6), expand_px=16, seed_box_size=128):
    import cv2 as _cv2

    from retinal_selflabel.selflabel.self_labelling import SpatialExpansionManager

    gt = load_gt(sample["mask_path"]).astype(np.uint8)
    h, w = gt.shape
    mgr = SpatialExpansionManager([(h, w)], [list(init_boxes)], expand_px=expand_px)

    # gt edge-distance proxy for logit margin
    gt_f = gt.astype(np.float32)
    din = _cv2.distanceTransform((gt_f > 0).astype(np.uint8), _cv2.DIST_L2, 3)
    dout = _cv2.distanceTransform((gt_f == 0).astype(np.uint8), _cv2.DIST_L2, 3)
    conf = np.maximum(din, dout) # high = confident
    conf_thresh = 1.5
    pseudo_logit = np.where(gt_f > 0, conf, -conf)
    margin = float(conf_thresh)

    snaps = {}
    max_it = max(iterations)
    for it in range(0, max_it + 1):
        if it in iterations:
            labelled = mgr.labelled_masks[0].copy()
            real_gt = mgr.is_real_gt[0].copy()
            pseudo = np.clip(labelled - real_gt, 0, 1)
            ring = mgr.get_expansion_ring(0)
            snaps[it] = {"labelled": labelled, "real_gt": real_gt, "pseudo": pseudo, 
                         "ring": ring, "coverage": mgr.get_coverage()}
        if it == max_it:
            break
        ring = mgr.get_expansion_ring(0)
        mgr.update_with_pseudo_labels(0, ring, (pseudo_logit > 0).astype(np.float32), pseudo_logit, margin)

    return snaps, gt
