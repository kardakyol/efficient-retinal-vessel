# new one with budget sweep, non overlap, density, variable box
import numpy as np

from retinal_selflabel.core.frangi_density import (
    density_cache,
    frangi_config,
    dns_for_samp,
    resolve_config,
    sample_in_downsampled_frame,
)

from retinal_selflabel.core.random_box_sampler import (
    BoxPlacement,
    build_image_registry,
)

class NewRandomizer:
    # fos sampler
    def __init__(self, samples, seed = 42, min_size = 32, max_size = 56,
        dataset_weighting = "by_image_count", cache_dir = "./outputs_new/cache/frangi",
        frangi_config_override = None, max_per_image_retries = 20, max_global_retries = 10000):

        if min_size < 1 or max_size < min_size:
            raise ValueError(f"Box size range is not valid[{min_size}, {max_size}]")

        self.samples = samples
        self.registry = build_image_registry(samples)
        if not self.registry:
            raise ValueError("There is no readable images")

        self.rng = np.random.default_rng(seed)
        self.min_size = int(min_size)
        self.max_size = int(max_size)
        self.max_per_image_retries = int(max_per_image_retries)
        self.max_global_retries = int(max_global_retries)

        # per dataset gruop registry
        self.by_dataset = {}
        for i, entry in enumerate(self.registry):
            self.by_dataset.setdefault(entry["dataset"], []).append(i)

        self._datasets = sorted(self.by_dataset.keys())
        if dataset_weighting == "by_image_count":
            counts = np.array([len(self.by_dataset[d]) for d in self._datasets], dtype=np.float64)
            self.dataset_probs = counts / counts.sum()
        elif dataset_weighting == "uniform":
            n = len(self._datasets)
            self.dataset_probs = np.full(n, 1.0 / n)
        else:
            raise ValueError(f"Dataset weighting must be by image count or uniform {dataset_weighting!r}")

        # frangi caches
        self._density_cache = density_cache(cache_dir)
        self._frangi_override = frangi_config_override
        # in memo cache
        self._density_mem_cache = {}
        self._density_w_cache = {}
        self.downsample_max_dim = 512

        # per-image accumulated forbidden mask (already-placed boxes)
        self.forbidden_w = {}

    # per-image state
    def reset_forbidden(self):
        self.forbidden_w = {}

    def density_and_scale(self, reg_idx):
        # returns for one image
        if reg_idx in self._density_w_cache:
            return self._density_w_cache[reg_idx]
        entry = self.registry[reg_idx]
        config = resolve_config(entry["dataset"], self._frangi_override)
        if reg_idx in self._density_mem_cache:
            density = self._density_mem_cache[reg_idx]
        else:
            density = self._density_cache.get_or_compute(
                entry["image_path"], config
            )
            self._density_mem_cache[reg_idx] = density
        density_w, scale = dns_for_samp(
            density, downsample_max_dim=self.downsample_max_dim
        )
        self._density_w_cache[reg_idx] = (density_w, scale)
        return density_w, scale

    def _get_forbidden_w(self, reg_idx):
        # forbidden mask
        if reg_idx not in self.forbidden_w:
            density_w, _ = self.density_and_scale(reg_idx)
            self.forbidden_w[reg_idx] = np.zeros(
                density_w.shape, dtype=np.uint8
            )
        return self.forbidden_w[reg_idx]

    def mark_forbidden(self, reg_idx, row, col, size):
        # original to downsampled via per image scale
        density_w, scale = self.density_and_scale(reg_idx)
        mask = self._get_forbidden_w(reg_idx)
        hw, ww = mask.shape
        if scale > 1.0:
            r0_w = int(np.floor(row / scale)) - 1
            c0_w = int(np.floor(col / scale)) - 1
            r1_w = int(np.ceil((row + size) / scale)) + 1
            c1_w = int(np.ceil((col + size) / scale)) + 1
        else:
            r0_w, c0_w, r1_w, c1_w = row, col, row + size, col + size
        r0_w, c0_w = max(0, r0_w), max(0, c0_w)
        r1_w, c1_w = min(hw, r1_w), min(ww, c1_w)
        mask[r0_w:r1_w, c0_w:c1_w] = 1

    def get_density(self, reg_idx):
        # Density map for an entry
        if reg_idx in self._density_mem_cache:
            return self._density_mem_cache[reg_idx]
        entry = self.registry[reg_idx]
        config = resolve_config(entry["dataset"], self._frangi_override)
        density = self._density_cache.get_or_compute(entry["image_path"], config)
        self._density_mem_cache[reg_idx] = density
        return density

    def draw_box_size(self, reg_idx):
        # draw a square box side length that fits the chosen image
        entry = self.registry[reg_idx]
        height, weight = entry["h"], entry["w"]
        upper = min(self.max_size, height, weight)
        lower = min(self.min_size, upper)
        if upper < lower:
            return upper
        size = int(self.rng.integers(lower, upper + 1))
        return size

    def _draw_one_box(self):
        # full round with dataset-image-size-overlap
        global_attempts = 0

        while global_attempts < self.max_global_retries:
            global_attempts += 1

            # dataset
            ds_index = int(
                self.rng.choice(len(self._datasets), p=self.dataset_probs)
            )
            ds = self._datasets[ds_index]

            # image within dataset
            candidate_image_indices = list(self.by_dataset[ds])
            self.rng.shuffle(candidate_image_indices)

            for reg_idx in candidate_image_indices:
                entry = self.registry[reg_idx]
                height, weight = entry["h"], entry["w"]
                size = self.draw_box_size(reg_idx)
                if height < size or weight < size:
                    continue

                # density + forbidden mask
                density_w, scale = self.density_and_scale(reg_idx)
                forbidden_w = self._get_forbidden_w(reg_idx)
                size_w = max(1, int(round(size / scale))) if scale > 1.0 else size

                pos = sample_in_downsampled_frame(
                    density_w, forbidden_w, size_w, scale,
                    height, weight, size, self.rng,
                )
                if pos is None:
                    continue  # no free space
                row, col = pos
                self.mark_forbidden(reg_idx, row, col, size)
                return BoxPlacement(sample_idx=entry["sample_idx"], dataset=entry["dataset"],
                    image_id=entry["image_id"], image_h=height, image_w=weight,row=row, col=col, size=size)
        return None

    # main function
    def sample_boxes(self, total_boxes):
        #Draw total boxes across the training area.
        self.reset_forbidden()
        placements = []
        for _ in range(total_boxes):
            p = self._draw_one_box()
            if p is None:
                print(f"Training set full after {len(placements)} of {total_boxes} requested boxes.")
                break
            placements.append(p)
        return placements

    # helpers
    @staticmethod
    def group_by_sample(placements):
        out = {}
        for p in placements:
            out.setdefault(p.sample_idx, []).append(p)
        return out

    @staticmethod
    def coverage_fraction(placements, total_image_area):
        # union of box coverage
        per_sample = {}
        for p in placements:
            per_sample.setdefault(p.sample_idx, []).append(p)

        total_annotated = 0
        for boxes in per_sample.values():
            height, weight = boxes[0].image_h, boxes[0].image_w
            m = np.zeros((height, weight), dtype=np.uint8)
            for b in boxes:
                r0 = max(0, b.row)
                c0 = max(0, b.col)
                r1 = min(height, b.row + b.size)
                c1 = min(weight, b.col + b.size)
                m[r0:r1, c0:c1] = 1
            total_annotated += int(m.sum())
        return total_annotated / max(total_image_area, 1)

    def total_image_area(self):
        return sum(e["h"] * e["w"] for e in self.registry)

# fos but with fixed size
class FixedSizeAdapter:
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("min_box_size", 128)
        kwargs.setdefault("max_box_size", 128)
        self._inner = NewRandomizer(*args, **kwargs)

    def sample_boxes(self, total_boxes, box_size= 128):
        self._inner.min_size = box_size
        self._inner.max_size = box_size
        return self._inner.sample_boxes(total_boxes)

    def total_image_area(self):
        return self._inner.total_image_area()

    @staticmethod
    def group_by_sample(*args, **kwargs):
        return NewRandomizer.group_by_sample(*args, **kwargs)

    @staticmethod
    def coverage_fraction(*args, **kwargs):
        return NewRandomizer.coverage_fraction(*args, **kwargs)
