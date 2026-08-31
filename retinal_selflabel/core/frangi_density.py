# image information density via frangi vesselness

import hashlib
import json
import os
from dataclasses import asdict, dataclass
import cv2
import numpy as np
from skimage.filters import frangi

# per-dataset frangi defaults based on vessel calibre distribution of each dataset
frangi_dataset_scale = {
    "DRIVE":(1.0, 4.0, 6), "CHASE":(1.5, 6.0, 6),
    "CHASE_DB1": (1.5, 6.0, 6), "HRF": (3.0, 12.0, 8),
}
# just in case if you try different dataset in future
frangi_default_scale = (1.0, 6.0, 6)

# configurations
@dataclass(frozen=True)
class frangi_config:
    sigma_min: float
    sigma_max: float
    n_scales: int
    black_ridges: bool = False # vessels are bright in green channel
    temperature: float = 1.0 # soft weighting exponent
    epsilon: float = 1e-3 # floor probability for background pixels
    smoothing_sigma: float = 2.0 # post-frangi gaussian

    def cache_key(self):
        s = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha1(s.encode()).hexdigest()[:12]


def resolve_config(dataset, override= None):
    if override is not None:
        return override
    key = dataset.upper().replace("-", "_")
    smin, smax, n = frangi_dataset_scale.get(key, frangi_default_scale)
    return frangi_config(sigma_min=smin, sigma_max=smax, n_scales=n)


# core
def green_channel(img_path):
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(img_path)
    g = bgr[:, :, 1].astype(np.float32) / 255.0
    return g

# green channel extraction, frangi vesselness, smoothing, normalisation, temperature and epsilon floor
def compute_density_map(img_path, config):
    g = green_channel(img_path)

    sigmas = np.linspace(config.sigma_min, config.sigma_max, config.n_scales)
    vesselness = frangi(g, sigmas=sigmas, black_ridges=config.black_ridges,).astype(np.float32)

    # post-frangi smoothing
    if config.smoothing_sigma > 0:
        ksize = int(2 * round(3 * config.smoothing_sigma) + 1)
        vesselness = cv2.GaussianBlur(vesselness, (ksize, ksize), config.smoothing_sigma)

    # normalise to [0, 1]
    p99 = float(np.percentile(vesselness, 99.0))
    if p99 > 1e-8:
        vesselness = np.clip(vesselness / p99, 0.0, 1.0)
    else:
        # degenerate image
        return np.full_like(vesselness, config.epsilon)

    # temperature
    if config.temperature != 1.0:
        vesselness = np.power(vesselness, config.temperature)

    # epsilon floor
    density = np.clip(vesselness, config.epsilon, 1.0).astype(np.float32)
    return density

# caching layer
class density_cache:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def cachekey(self, img_path, config):
        h = hashlib.sha1(img_path.encode()).hexdigest()[:12]
        return f"{h}_{config.cache_key()}.npz"

    def cache_path(self, img_path, config):
        return os.path.join(self.cache_dir, self.cachekey(img_path, config))

    def get_or_compute(self, img_path, config):
        path = self.cache_path(img_path, config)
        if os.path.exists(path):
            try:
                with np.load(path) as data:
                    return data["density"].astype(np.float32)
            except Exception:
                pass
        density = compute_density_map(img_path, config)
        np.savez_compressed(path, density=density)
        return density

# pre downsample density for sampling
def dns_for_samp(density, downsample_max_dim = 512,):
    h, w = density.shape
    scale = max(1.0, max(h, w) / float(downsample_max_dim))
    if scale <= 1.0:
        return density, 1.0
    new_w = max(1, int(round(w / scale)))
    new_h = max(1, int(round(h / scale)))
    density_w = cv2.resize(density, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return density_w, scale

# box filter
def samp_in_ds( density_w, forbidden_w, box_size_w, scale, orig_h, orig_w, orig_box_size, rng,):
    hw, ww = density_w.shape
    if hw < box_size_w or ww < box_size_w:
        return None

    mean_density = cv2.boxFilter(density_w, ddepth=-1, ksize=(box_size_w, box_size_w), anchor=(0, 0), 
                                 normalize=True, borderType=cv2.BORDER_ISOLATED)
    
    valid = mean_density[: hw - box_size_w + 1, : ww - box_size_w + 1].copy()

    if forbidden_w is not None:
        fm_sum = cv2.boxFilter(forbidden_w.astype(np.float32), ddepth=-1, 
            ksize=(box_size_w, box_size_w), anchor=(0, 0), 
            normalize=False, borderType=cv2.BORDER_ISOLATED)
        fm_sum = fm_sum[: hw - box_size_w + 1, : ww - box_size_w + 1]
        valid[fm_sum > 0.0] = 0.0

    total = float(valid.sum())
    if total <= 0.0:
        return None

    flat = valid.ravel()
    csum = np.cumsum(flat, dtype=np.float64)
    target = rng.random() * float(csum[-1])
    idx = int(np.searchsorted(csum, target, side="right"))
    idx = min(idx, flat.size - 1)
    ncols_valid = valid.shape[1]
    r_w = idx // ncols_valid
    c_w = idx % ncols_valid

    if scale > 1.0:
        r = int(round(r_w * scale))
        c = int(round(c_w * scale))
        r = max(0, min(r, orig_h - orig_box_size))
        c = max(0, min(c, orig_w - orig_box_size))
    else:
        r, c = r_w, c_w
    return int(r), int(c)

# sampling helper
def sample_position_from_density(density, box_size, rng,
    forbidden_mask = None, downsample_max_dim = 512):
    # sample row, col position from density map with box size
    height, weight = density.shape
    if height < box_size or weight < box_size:
        return None

    density_w, scale = dns_for_samp(density, downsample_max_dim)
    forbidden_w = None
    if forbidden_mask is not None:
        forbidden_w, _ = dns_for_samp(forbidden_mask.astype(np.float32), downsample_max_dim)
        forbidden_w = (forbidden_w > 0.0).astype(np.uint8)

    box_size_w = max(1, int(round(box_size / scale)))
    return samp_in_ds(density_w, forbidden_w, box_size_w, scale, height, weight, box_size, rng)

# just in case to call code, to use directly
sample_in_downsampled_frame = samp_in_ds
