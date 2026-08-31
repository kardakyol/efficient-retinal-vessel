# random box sampler with frangi, overlap, size
from dataclasses import dataclass
import numpy as np

from retinal_selflabel.core.frangi_density import frangi_config, dns_for_samp
from retinal_selflabel.core.random_box_sampler_v2 import NewRandomizer

@dataclass(frozen=True)
class factorial_cell:
    # 2x2x2
    frangi: bool
    non_overlap: bool
    variable_size: bool

    @property
    def label(self):
        index = int(self.frangi) * 4 + int(self.non_overlap) * 2 + int(self.variable_size)
        names = [
            "v1_baseline", "size_only", "overlap_only", "size_and_overlap", 
            "frangi_only", "frangi_size", "frangi_overlap", "v2_full"
        ]
        return names[index]

    @property
    def short(self):
        f = "F" if self.frangi else "f"
        o = "O" if self.non_overlap else "o"
        s = "S" if self.variable_size else "s"
        return f + o + s

    def as_dict(self):
        return {
            "frangi": self.frangi, "non_overlap": self.non_overlap, "variable_size": self.variable_size,
            "label": self.label, "short": self.short
        }


fos_cells = [
    factorial_cell(frangi=False, non_overlap=False, variable_size=False),  
    factorial_cell(frangi=False, non_overlap=False, variable_size=True),   
    factorial_cell(frangi=False, non_overlap=True,  variable_size=False),  
    factorial_cell(frangi=False, non_overlap=True,  variable_size=True),   
    factorial_cell(frangi=True,  non_overlap=False, variable_size=False),  
    factorial_cell(frangi=True,  non_overlap=False, variable_size=True),   
    factorial_cell(frangi=True,  non_overlap=True,  variable_size=False), 
    factorial_cell(frangi=True,  non_overlap=True,  variable_size=True), 
]

# configurable sampler 
class config_sampler(NewRandomizer):
    def __init__(self, samples, cell, seed = 42,
        fixed_size = 128, min_size = 32, max_size = 256,
        dataset_weighting = "by_image_count", cache_dir = "./outputs_new/cache/frangi",
        frangi_config_override = None, max_per_image_retries = 20,
        max_global_retries= 10000,):

        super().__init__(
            samples=samples, seed=seed, min_size=min_size, 
            max_size=max_size, dataset_weighting=dataset_weighting, cache_dir=cache_dir,
            frangi_config_override=frangi_config_override, max_per_image_retries=max_per_image_retries,
            max_global_retries=max_global_retries,
        )
        self.cell = cell
        self.fixed_size = fixed_size

        # cache for the uniform density case
        self.density_cache = {}

    # size policy
    def draw_box_size(self, reg_index):
        # either fixed size or uniform draw
        entry = self.registry[reg_index]
        height, weight = entry["h"], entry["w"]
        if not self.cell.variable_size:
            return min(self.fixed_size, height, weight)
        upper = min(self.max_size, height, weight)
        lower = min(self.min_size, upper)
        if upper < lower:
            return upper
        return int(self.rng.integers(lower, upper + 1))

    # density policy
    def density_and_scale(self, reg_index):
        # if frangi off, flat unit
        # else real density
        if self.cell.frangi:
            return super().density_and_scale(reg_index)
        if reg_index in self.density_cache:
            return self.density_cache[reg_index]
        
        entry = self.registry[reg_index]
        density = np.ones((entry["h"], entry["w"]), dtype=np.float32)
        density_w, scale = dns_for_samp(density, downsample_max_dim=self.downsample_max_dim)
        self.density_cache[reg_index] = (density_w, scale)
        return density_w, scale

    # overlapping
    def mark_forbidden(self, reg_index, row, col, size):
        # if non-overlap is off, do nothing
        if not self.cell.non_overlap:
            return
        super().mark_forbidden(reg_index, row, col, size)
    # introspection
    def as_metadata(self):
        return {"sampler": "config_sampler", "cell": self.cell.as_dict(), 
            "fixed_box_size": self.fixed_size, "min_box_size": self.min_size,
            "max_box_size": self.max_size,}
