# old one with budget sweep, non overlap, density, variable box

import json
import os
from dataclasses import asdict, dataclass

import cv2
import numpy as np

# dataclasses
@dataclass(frozen=True)
class BoxPlacement:
    # one sampled box on one training image
    sample_idx: int        
    dataset: str           
    image_id: str          
    image_h: int
    image_w: int
    row: int              
    col: int               
    size: int              

    def as_tuple(self):
        return (self.row, self.col, self.size)


# registry builder
def build_image_registry(samples):
    # pre-read to sample
    registry = []
    for idx, s in enumerate(samples):
        img = cv2.imread(s["image_path"])
        if img is None:
            print(f"File cannot read {s['image_path']} excluding from registry.")
            continue
        height, width = img.shape[:2]
        registry.append({
            "sample_idx": idx,
            "dataset": s["dataset"],
            "image_id": s["id"],
            "image_path": s["image_path"],
            "mask_path": s["mask_path"],
            "h": height,
            "w": width,
        })
    return registry

# dataset, image, position sampler
class RandomBoxSampler:

    def __init__(self, samples, seed = 42, dataset_weighting = "by_image_count",
        enforce_vessel_fraction = None, max_attempts_per_box= 50):
        self.samples = samples
        self.registry = build_image_registry(samples)
        if not self.registry:
            raise ValueError("No readable images.")

        self.rng = np.random.default_rng(seed)
        self.enforce_vessel_fraction = enforce_vessel_fraction
        self.max_attempts_per_box = max_attempts_per_box

        # group registry indices by dataset
        self.by_dataset= {}
        for i, entry in enumerate(self.registry):
            self.by_dataset.setdefault(entry["dataset"], []).append(i)

        self._datasets = sorted(self.by_dataset.keys())
        if dataset_weighting == "by_image_count":
            counts = np.array([len(self.by_dataset[d])
                               for d in self._datasets], dtype=np.float64)
            self.dataset_probs = counts / counts.sum()
        elif dataset_weighting == "uniform":
            n = len(self._datasets)
            self.dataset_probs = np.full(n, 1.0 / n)
        else:
            raise ValueError(f"Dataset weighting must be by image count or uniform {dataset_weighting!r}")

        self._mask_cache = {}

    # vessel-fraction helpers
    def _get_mask(self, reg_idx):
        if reg_idx in self._mask_cache:
            return self._mask_cache[reg_idx]
        entry = self.registry[reg_idx]
        m = cv2.imread(entry["mask_path"], cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise RuntimeError(f"Cannot read mask {entry['mask_path']}")
        m = (m > 127).astype(np.uint8)
        self._mask_cache[reg_idx] = m
        return m

    def _patch_vessel_fraction(self, reg_idx, row, col, size):
        m = self._get_mask(reg_idx)
        patch = m[row:row + size, col:col + size]
        return float(patch.mean()) if patch.size > 0 else 0.0

    # core sampling 
    def _draw_position(self, reg_idx, size):
        # uniform draw
        entry = self.registry[reg_idx]
        h, w = entry["h"], entry["w"]
        if h < size or w < size:
            return None
        # top-left corner ranges
        r = int(self.rng.integers(0, h - size + 1))
        c = int(self.rng.integers(0, w - size + 1))
        return r, c

    def _draw_one_box(self, size):
        # Full 3 level draw for a single box
        attempts = 0
        max_draws = self.max_attempts_per_box if self.enforce_vessel_fraction else 1

        while attempts < max_draws:
            attempts += 1
            # dataset
            ds = self._datasets[int(self.rng.choice(len(self._datasets), p=self.dataset_probs))]
            # image within dataset
            candidates = [i for i in self.by_dataset[ds]
                          if self.registry[i]["h"] >= size
                          and self.registry[i]["w"] >= size]
            if not candidates:
                continue
            reg_idx = int(self.rng.choice(candidates))
            # position within image
            pos = self._draw_position(reg_idx, size)
            if pos is None:
                continue
            row, col = pos
            # vessel-fraction guard
            if self.enforce_vessel_fraction is not None:
                vf = self._patch_vessel_fraction(reg_idx, row, col, size)
                if vf < self.enforce_vessel_fraction:
                    continue
            entry = self.registry[reg_idx]
            return BoxPlacement(
                sample_idx=entry["sample_idx"],
                dataset=entry["dataset"],
                image_id=entry["image_id"],
                image_h=entry["h"],
                image_w=entry["w"],
                row=row, col=col, size=size,
            )
        return None

    # draw boxes, each of side box size
    def sample_boxes(self, total_boxes, box_size = 128):
        placements= []
        for _ in range(total_boxes):
            p = self._draw_one_box(box_size)
            if p is not None:
                placements.append(p)

        if len(placements) < total_boxes:
            print(f"Requested {total_boxes} boxes but only {len(placements)} satisfied the vessel-fraction")
        return placements

    # convenience
    @staticmethod
    def group_by_sample(placements):
        # group placements
        out = {}
        for p in placements:
            out.setdefault(p.sample_idx, []).append(p)
        return out

    @staticmethod
    def coverage_fraction(placements, total_image_area):
        # compute coverage
        per_sample = RandomBoxSampler.group_by_sample(placements)
        total_annotated = 0
        for sample_idx, boxes in per_sample.items():
            h, w = boxes[0].image_h, boxes[0].image_w
            m = np.zeros((h, w), dtype=np.uint8)
            for b in boxes:
                r0 = max(0, b.row)
                c0 = max(0, b.col)
                r1 = min(h, b.row + b.size)
                c1 = min(w, b.col + b.size)
                m[r0:r1, c0:c1] = 1
            total_annotated += int(m.sum())
        return total_annotated / max(total_image_area, 1)

    def total_image_area(self):
        return sum(e["h"] * e["w"] for e in self.registry)


# serialisation
def save_placements(placements, path, metadata = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "metadata": metadata or {},
        "placements": [asdict(p) for p in placements],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_placements(path):
    with open(path) as f:
        payload = json.load(f)
    placements = [BoxPlacement(**p) for p in payload["placements"]]
    return placements, payload.get("metadata", {})
