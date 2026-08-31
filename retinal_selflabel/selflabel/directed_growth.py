from dataclasses import dataclass, field

import cv2
import numpy as np

from retinal_selflabel.selflabel.self_labelling import SpatialExpansionManager

# growth rate schedular
@dataclass
class GrowthScheduler:
    base: int = 16
    mode: str = "constant"
    slope: float = 0.0
    ratio: float = 1.0
    decay: float = 1.0
    min_r: int = 2
    max_r: int = 64
    fn: object = None

    def radius(self, k):
        if self.mode == "custom" and self.fn is not None:
            r = self.fn(k)
        elif self.mode == "linear":
            r = self.base + self.slope * (k - 1)
        elif self.mode == "geometric":
            r = self.base * (self.ratio ** (k - 1))
        elif self.mode == "decelerate":
            r = self.base * (self.decay ** (k - 1))
        else:
            r = self.base
        return int(max(self.min_r, min(self.max_r, round(r))))

    # convenient named presets matching the meeting
    @classmethod
    def fast(cls):
        return cls(base=32, mode="constant")

    @classmethod
    def medium(cls):
        return cls(base=16, mode="constant")

    @classmethod
    def slow(cls):
        return cls(base=8, mode="constant")

    @classmethod
    def accelerating(cls):
        return cls(base=6, mode="geometric", ratio=1.25, max_r=48)


# directed expansion ring
def directed_ring(labelled, guidance, expand_px, keep_fraction = 0.5, min_keep = 1):
    lab = (labelled > 0).astype(np.uint8)
    ks = 2 * int(expand_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    dilated = cv2.dilate(lab, kernel, iterations=1)
    ring = np.clip(dilated.astype(np.int16) - lab.astype(np.int16), 0, 1).astype(np.uint8)
    if ring.sum() == 0 or keep_fraction >= 1.0:
        return ring

    if guidance.shape != ring.shape:
        guidance = cv2.resize(guidance.astype(np.float32), (ring.shape[1], ring.shape[0]))
    ring_vals = guidance[ring > 0]
    if ring_vals.size == 0:
        return ring
    n_keep = max(min_keep, int(round(keep_fraction * ring_vals.size)))
    if n_keep >= ring_vals.size:
        return ring
    thr = np.partition(ring_vals, ring_vals.size - n_keep)[ring_vals.size - n_keep]
    directed = np.where((ring > 0) & (guidance >= thr), 1, 0).astype(np.uint8)
    return directed

class DirectedExpansionManager(SpatialExpansionManager):
    def __init__(self, image_shapes, initial_patches, guidance_maps, scheduler = None, keep_fraction = 0.5, expand_px = 16):
        super().__init__(image_shapes, initial_patches, expand_px)
        self.guidance_maps = [g.astype(np.float32) for g in guidance_maps]
        self.scheduler = scheduler or GrowthScheduler.medium()
        self.keep_fraction = float(keep_fraction)

    # replace one image's guidance map
    def set_guidance(self, image_idx, guidance):
        self.guidance_maps[image_idx] = guidance.astype(np.float32)

    # iteration is set by the labeller loop on the manager each round
    def get_expansion_ring(self, image_idx):
        k = max(1, int(getattr(self, "iteration", 1)))
        r = self.scheduler.radius(k)
        return directed_ring(self.labelled_masks[image_idx], self.guidance_maps[image_idx], expand_px=r, keep_fraction=self.keep_fraction)


# Speed-accuracy frontier harness
@dataclass
class FrontierPoint:
    schedule: str
    iteration: int
    coverage: float
    dice: float

@dataclass
class GrowthScheduleSweep:
    schedules: dict = field(default_factory=lambda: {"fast": GrowthScheduler.fast(), "medium": GrowthScheduler.medium(), "slow": GrowthScheduler.slow()})
    max_iterations: int = 20
    dry_run: bool = True
    labeller_factory: object = None

    def run(self):
        if not self.dry_run and self.labeller_factory is None:
            raise ValueError("provide labeller_factory for a real run.")
        out= []
        for name, sched in self.schedules.items():
            if self.dry_run:
                out.extend(self._simulate(name, sched))
            else:
                lab = self.labeller_factory(sched)     
                _, log = lab.run()
                for e in log:
                    out.append(FrontierPoint(name, e["iteration"], e["coverage"], e["val_dice"]))
        return out

    def _simulate(self, name, sched):
        # illustrative monotone-saturating dice vs coverage
        pts = []
        cov = 0.004
        ceiling = {"fast": 0.705, "medium": 0.717, "slow": 0.724}.get(name, 0.71)
        for k in range(1, self.max_iterations + 1):
            r = sched.radius(k)
            cov = min(1.0, cov + r / 600.0)
            dice = ceiling * (1.0 - np.exp(-6.0 * cov)) + 0.69 * np.exp(-6.0 * cov)
            pts.append(FrontierPoint(name, k, float(cov), float(dice)))
        return pts

if __name__ == "__main__":
    print("directed_growth smoke test")

    # schedules
    for name, s in [("fast", GrowthScheduler.fast()), ("medium", GrowthScheduler.medium()), ("slow", GrowthScheduler.slow()), ("accel", GrowthScheduler.accelerating())]:
        radii = [s.radius(k) for k in range(1, 6)]
        print(f" schedule {name:7s} radii = {radii}")

    # directed ring vs isotropic ring
    lab = np.zeros((120, 120), np.uint8)
    lab[50:70, 50:70] = 1 # a seed box
    g = np.zeros((120, 120), np.float32)
    g[:, 60:] = 1.0   # guidance favours the right side
    iso = directed_ring(lab, g, expand_px=10, keep_fraction=1.0)
    dirr = directed_ring(lab, g, expand_px=10, keep_fraction=0.5)
    print(f"\nisotropic ring pixels : {int(iso.sum())}, directed ring pixels : {int(dirr.sum())}")
    # directedness check
    cols = np.where(dirr > 0)[1]
    print(f"directed ring mean col: {cols.mean():.1f}")
    assert int(dirr.sum()) < int(iso.sum())
    assert cols.mean() >= 60

    # frontier harness
    sweep = GrowthScheduleSweep(max_iterations=12, dry_run=True)
    pts = sweep.run()
    finals = {}
    for p in pts:
        finals[p.schedule] = (p.coverage, p.dice)
    print("\n speed-accuracy frontier (Smoke test points):")
    for k, (cov, dice) in finals.items():
        print(f" {k:7s}: coverage={cov*100:5.1f}%  dice={dice:.4f}")
    print("\nall smoke tests have passed")
