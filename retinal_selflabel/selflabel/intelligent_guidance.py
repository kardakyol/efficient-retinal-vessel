
import math

import cv2
import numpy as np

# returns the mean of field over the box 
def box_mean_field(field, box_size):
    h, w = field.shape
    if h < box_size or w < box_size:
        return np.empty((0, 0), dtype=np.float32)
    box_means = cv2.boxFilter(
        field.astype(np.float32), ddepth=-1, ksize=(box_size, box_size),
        anchor=(0, 0), normalize=True, borderType=cv2.BORDER_ISOLATED,
    )
    return box_means[: h - box_size + 1, : w - box_size + 1]


def _argmax2d(arr):
    index = int(np.argmax(arr))
    return index // arr.shape[1], index % arr.shape[1]


# all training area density-histogram seeding
class CorpusDensityHistogramSampler:
    def __init__(self, n_seeds, box_size=128, seed=42, min_density_quantile=0.50):
        self.n_seeds = n_seeds
        self.box_size = box_size
        self.seed = seed
        self.min_density_quantile = min_density_quantile

    def plan(self, n_images, density_provider, image_shapes):
        rng = np.random.default_rng(self.seed)

        # collecting candidates
        candidates = []
        for i in range(n_images):
            density = density_provider(i)
            box_means = box_mean_field(density, self.box_size)
            if box_means.size == 0:
                continue
            threshold = float(np.quantile(box_means, self.min_density_quantile))
            stride = max(1, self.box_size // 2)
            for r in range(0, box_means.shape[0], stride):
                for c in range(0, box_means.shape[1], stride):
                    value = float(box_means[r, c])
                    if value >= threshold:
                        candidates.append((i, r, c, value))
        if not candidates:
            raise RuntimeError("No candidate boxes")

        vals = np.array([c[3] for c in candidates], dtype=np.float64)
        lo, hi = float(vals.min()), float(vals.max())

        # target density levels
        if hi - lo < 1e-9:
            targets = np.full(self.n_seeds, lo)
        else:
            targets = np.linspace(lo, hi, self.n_seeds)

        # match one candidate to each target level
        plan = {i: [] for i in range(n_images)}
        used = np.zeros(len(candidates), dtype=bool)
        for target_level in targets:
            dist = np.abs(vals - target_level)
            dist[used] = np.inf
            j = int(np.argmin(dist))
            if not np.isfinite(dist[j]):   # if all used, allow reuse
                j = int(np.argmin(np.abs(vals - target_level)))
            else:
                used[j] = True
            img, r, c, _ = candidates[j]
            plan[img].append((int(r), int(c), int(self.box_size)))

        _ = rng  # reserved for future stochastic tie-breaking
        return [plan[i] for i in range(n_images)]


# one representative box per image, spanning the all training area density axis.
def per_image_density_histogram_seeds(n_images, density_provider, box_size = 128, min_density_quantile = 0.50, seed = 42):
    rng = np.random.default_rng(seed)
    cand_by_image = {}
    image_peak = []

    for i in range(n_images):
        density = density_provider(i)
        box_means = box_mean_field(density, box_size)
        if box_means.size == 0:
            cand_by_image[i] = []
            image_peak.append(0.0)
            continue
        threshold = float(np.quantile(box_means, min_density_quantile))
        stride = max(1, box_size // 2)
        candidates = [
            (float(box_means[r, c]), r, c)
            for r in range(0, box_means.shape[0], stride)
            for c in range(0, box_means.shape[1], stride)
            if float(box_means[r, c]) >= threshold
        ]
        cand_by_image[i] = candidates
        image_peak.append(float(box_means.max()))

    all_vals = [v for i in range(n_images) for (v, _, _) in cand_by_image[i]]
    if not all_vals:
        raise RuntimeError("No candidate boxes")
    lo, hi = float(min(all_vals)), float(max(all_vals))

    # order images by how dense they can get
    order = sorted(range(n_images), key=lambda i: image_peak[i])
    targets = (
        np.full(n_images, lo) if hi - lo < 1e-9 else np.linspace(lo, hi, n_images)
    )

    plan = [[] for _ in range(n_images)]
    for rank, img in enumerate(order):
        candidates = cand_by_image[img]
        if not candidates:
            continue
        target_level = float(targets[rank])
        _, r, c = min(candidates, key=lambda cand: abs(cand[0] - target_level))
        plan[img] = [(int(r), int(c), int(box_size))]

    _ = rng
    return plan

# morphological granulometry
class GranulometryProfiler:
    def __init__(self, radii = (1, 2, 3, 5, 8, 12, 16, 24)):
        self.radii = list(radii)

    @staticmethod
    def _disk(r):
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

    def spectrum(self, binary):
        binary_u8 = (binary > 0.5).astype(np.uint8)
        areas = []
        for radius in self.radii:
            opened = cv2.morphologyEx(binary_u8, cv2.MORPH_OPEN, self._disk(radius))
            areas.append(float(opened.sum()))
        areas.append(0.0) # open at r=inf is 0
        pattern_spectrum = np.array([areas[k] - areas[k + 1] for k in range(len(self.radii))])
        return np.clip(pattern_spectrum, 0.0, None).astype(np.float64)

    def size_label_map(self, binary):
        # per-pixel dominant-scale index
        binary_u8 = (binary > 0.5).astype(np.uint8)
        label = np.full(binary_u8.shape, -1, dtype=np.int16)
        for k, radius in enumerate(self.radii):
            opened = cv2.morphologyEx(binary_u8, cv2.MORPH_OPEN, self._disk(radius))
            label[opened > 0] = k
        return label

    def corpus_spectrum(self, n_images, binary_provider):
        total = np.zeros(len(self.radii), dtype=np.float64)
        for i in range(n_images):
            total += self.spectrum(binary_provider(i))
        total_sum = total.sum()
        return total / total_sum if total_sum > 0 else total


def size_balanced_seeds(n_images, binary_provider, image_shapes, n_seeds, box_size = 128, profiler = None, seed = 42):
    # pick seeds that flatten the training area object-size distribution
    profiler = profiler or GranulometryProfiler()
    rng = np.random.default_rng(seed)

    corpus = profiler.corpus_spectrum(n_images, binary_provider)
    eps = 1e-3
    target = np.full_like(corpus, 1.0 / len(corpus))  # uniform target
    deficit = np.clip(target - corpus, 0.0, None) + eps 

    # place per_image boxes 
    base, extra = divmod(n_seeds, max(1, n_images))
    quota = [max(1, base + (1 if i < extra else 0)) for i in range(n_images)]
    plan = {i: [] for i in range(n_images)}

    for i in range(n_images):
        binary = (binary_provider(i) > 0.5).astype(np.float32)
        size_labels = profiler.size_label_map(binary)
        size_field = np.zeros(size_labels.shape, dtype=np.float32)
        for k in range(len(profiler.radii)):
            size_field += (size_labels == k).astype(np.float32) * float(deficit[k])

        size_score = box_mean_field(size_field, box_size) # size-balance score
        vessel_density = box_mean_field(binary, box_size) # vessel density
        if size_score.size == 0:
            continue

        floor = max(0.02, 0.25 * float(vessel_density.max()))
        valid = vessel_density >= floor
        if not valid.any():
            valid = vessel_density >= max(1e-6, float(vessel_density.max()))

        # rank by size-deficit score
        combined = size_score + 1e-3 * vessel_density
        stride = max(1, box_size // 2)
        candidates = [(float(combined[r, c]), float(vessel_density[r, c]), r, c) for r in range(0, size_score.shape[0], stride) 
                 for c in range(0, size_score.shape[1], stride) if valid[r, c]]
        if not candidates:
            best_row, best_col = _argmax2d(vessel_density)
            plan[i] = [(int(best_row), int(best_col), int(box_size))]
            continue
        candidates.sort(key=lambda cand: (cand[0], cand[1]), reverse=True)

        picked = []
        for _sc, _d, r, c in candidates:
            if len(picked) >= quota[i]:
                break
            if all(abs(r - prev_row) >= box_size or abs(c - prev_col) >= box_size
                   for prev_row, prev_col in picked):
                picked.append((r, c))
        plan[i] = [(int(r), int(c), int(box_size)) for r, c in picked]

    _ = rng
    return [plan[i] for i in range(n_images)]


# confidence feedback loop
def binary_entropy(prob):
    # pixelwise binary entropy of a probability map, normalised to 0,1
    p = np.clip(prob.astype(np.float64), 1e-6, 1 - 1e-6)
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return (entropy / math.log(2.0)).astype(np.float32)


class ConfidenceFeedbackSelector:
    # model-in-the-loop selection of the next annotation regions
    def __init__(self, box_size = 128, entropy_weight = 1.0, prior_weight = 0.3, seed = 42):
        self.box_size = box_size
        self.entropy_weight = entropy_weight
        self.prior_weight = prior_weight
        self.rng = np.random.default_rng(seed)

    def priority_map(self, prob, prior_density= None):
        entropy = binary_entropy(prob)
        priority = self.entropy_weight * entropy
        if prior_density is not None:
            prior = prior_density.astype(np.float32)
            if prior.shape != entropy.shape:
                prior = cv2.resize(prior, (entropy.shape[1], entropy.shape[0]))
            priority = priority + self.prior_weight * prior
        return priority

    def select_seeds_for_image(self, prob, k, prior_density = None, forbidden = None):
        # Return up to k seed boxes 
        priority = self.priority_map(prob, prior_density)
        box_means = box_mean_field(priority, self.box_size)
        if box_means.size == 0:
            return []
        forbidden_mask = np.zeros(box_means.shape, dtype=bool)
        if forbidden is not None:
            forbidden_means = box_mean_field(forbidden.astype(np.float32), self.box_size)
            forbidden_mask |= forbidden_means > 0
        out = []
        work = box_means.copy()
        work[forbidden_mask] = -np.inf
        for _ in range(k):
            if not np.isfinite(work).any() or work.max() <= -np.inf:
                break
            r, c = _argmax2d(work)
            if not np.isfinite(work[r, c]):
                break
            out.append((int(r), int(c), int(self.box_size)))
            # paint a no-overlap window around the chosen corner
            r0, r1 = max(0, r - self.box_size + 1), r + self.box_size
            c0, c1 = max(0, c - self.box_size + 1), c + self.box_size
            work[r0:r1, c0:c1] = -np.inf
        return out


# real-pipeline glue
def make_frangi_provider(samples, cache, resolve_config_fn, img_size=None):
    # density_provider with last frangi cache
    def provider(i):
        sample = samples[i]
        cfg = resolve_config_fn(sample["dataset"])
        density = cache.get_or_compute(sample["image_path"], cfg)
        if img_size is not None:
            density = cv2.resize(density, (img_size, img_size))
        return density.astype(np.float32)
    return provider


# smoke test
def _synthetic_density(h=200, w=200, n_blobs=6, rng=None):
    rng = rng or np.random.default_rng(0)
    density = np.zeros((h, w), dtype=np.float32)
    ys, xs = np.mgrid[0:h, 0:w]
    for _ in range(n_blobs):
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        sigma = rng.integers(8, 40)
        density += np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * sigma * sigma))
    density += 1e-3
    return (density / density.max()).astype(np.float32)


if __name__ == "__main__":
    print("intelligent_guidance smoke test")
    rng = np.random.default_rng(7)
    SHAPES = [(200, 200)] * 10
    maps = [_synthetic_density(rng=np.random.default_rng(i)) for i in range(10)]
    dens = lambda i: maps[i]
    bins = lambda i: (maps[i] > 0.4).astype(np.float32)

    # density-histogram seeding
    seeds_hist = CorpusDensityHistogramSampler(n_seeds=12, box_size=64).plan(10, dens, SHAPES)
    tot1 = sum(len(x) for x in seeds_hist)
    print(f"density-histogram - {tot1} seeds over {sum(bool(x) for x in seeds_hist)} images")
    assert tot1 == 12

    # granulometry size balancing
    prof = GranulometryProfiler(radii=(1, 2, 4, 8, 16))
    cspec = prof.corpus_spectrum(10, bins)
    print(f"training area size spectrum is {np.round(cspec, 3)}")
    seeds_size = size_balanced_seeds(10, bins, SHAPES, n_seeds=12, box_size=64,
                             profiler=prof)
    tot2 = sum(len(x) for x in seeds_size)
    print(f"size-balanced - {tot2} seeds")
    assert tot2 == 12

    # confidence feedback
    sel = ConfidenceFeedbackSelector(box_size=64)
    prob = np.clip(maps[0] + rng.normal(0, 0.15, maps[0].shape), 0, 1)
    seeds3 = sel.select_seeds_for_image(prob, k=3, prior_density=maps[0])
    print(f"feedback (1 image) - {len(seeds3)} non-overlapping seeds at {seeds3}")
    assert 1 <= len(seeds3) <= 3

    print("\nall smoke tests are passed")
