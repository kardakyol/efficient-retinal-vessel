# RGB to Frangi sampling density
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

import retinal_selflabel.figures.fig_common as fc
import retinal_selflabel.figures.figure_utils as fu
from retinal_selflabel.core.frangi_density import frangi_dataset_scale

fu.apply_ieee_style(8.5)
plt.rcParams.update({"svg.fonttype": "none", "font.family": "serif"})
INK, MUTED = fu.C_TEXT, fu.C_MUTED
C_DRIVE, C_CHASE, C_HRF = "#1971c2", "#2b8a3e", "#e67700"


def save3(fig, out, stem):
    os.makedirs(out, exist_ok=True); p = os.path.join(out, stem)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{p}.{ext}", bbox_inches="tight", pad_inches=0.05,
                    dpi=300 if ext == "png" else None)
    plt.close(fig); print("  saved", p + ".{svg,pdf,png}")


def arrow(ax, p0, p1, color=INK, lw=1.2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10,
                 lw=lw, color=color, shrinkA=2, shrinkB=2, zorder=6))


def tile(ax, img, cx, cy, s, cmap=None, title=None, vmin=None, vmax=None, ec=fu.C_BORDER):
    ax.imshow(img, extent=(cx - s / 2, cx + s / 2, cy - s / 2, cy + s / 2),
              aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
              interpolation="bilinear", zorder=2)
    ax.add_patch(Rectangle((cx - s / 2, cy - s / 2), s, s, fill=False, ec=ec, lw=1.0, zorder=3))
    if title:
        ax.text(cx, cy + s / 2 + 0.10, title, ha="center", va="bottom", fontsize=7.6, fontweight="bold")


def main(data_dir, out_dir):
    train, _ = fc.get_split(data_dir)
    grp = fc.by_dataset(train)

    fig = plt.figure(figsize=(13, 8.8))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 13); ax.set_ylim(0, 8.8)
    ax.set_aspect("equal"); ax.axis("off")
    ax.text(0.25, 8.55, "From RGB to a sampling density: dataset-adaptive multiscale Frangi vesselness",
            fontsize=11, fontweight="bold", ha="left")

    # Frangi recipe on one demo image
    ax.text(0.25, 8.15, "The recipe (frangi_density.compute_density_map), shown on one CHASE_DB1 image:",
            fontsize=8.6, fontweight="bold", ha="left")
    demo = fc.pick_demo(train, "CHASE", 0)
    rgb = fc.center_square(fc.load_rgb(demo["image_path"]))
    green = rgb[:, :, 1]
    raw = fc.center_square(fc.frangi_response(demo))     # multiscale frangi
    dm = fc.center_square(fc.density_for(demo))          # full pipeline output
    s = 1.45; cy = 6.95
    xs = [1.15, 3.35, 5.55, 7.75, 9.95]
    tile(ax, rgb, xs[0], cy, s, title="RGB")
    tile(ax, green, xs[1], cy, s, cmap="Greens", title="green channel")
    tile(ax, raw, xs[2], cy, s, cmap="magma", title="multiscale Frangi")
    tile(ax, dm, xs[3], cy, s, cmap="magma", title="density map  DM")
    tile(ax, dm / max(dm.sum(), 1e-9), xs[4], cy, s, cmap="magma", title="sampling distribution")
    labels = ["take green ch.", r"frangi($\sigma_1..\sigma_n$)", "smooth + p99 norm", "/ \u03a3 \u2192 prob."]
    for i in range(4):
        arrow(ax, (xs[i] + s / 2 + 0.03, cy), (xs[i + 1] - s / 2 - 0.03, cy))
        ax.text((xs[i] + xs[i + 1]) / 2, cy + 0.12, labels[i], fontsize=6.5, color=MUTED, ha="center")
    ax.text(xs[4] + s / 2 + 0.25, cy, "+ \u03b5 floor so every\npixel keeps a small\nchance (non-overlap\nsafety)", fontsize=7,
            color=MUTED, ha="left", va="center")

    # all three datasets with their own sigma
    ax.text(0.25, 5.75, "Dataset-adaptive scale: each dataset runs its OWN \u03c3 range (default_frangi), "
            "matched to vessel calibre \u2014 one density map per training image",
            fontsize=8.6, fontweight="bold", ha="left")

    rows = [("DRIVE", "DRIVE", C_DRIVE, "~1\u20134 px vessels"),
            ("CHASE", "CHASE_DB1", C_CHASE, "~2\u20136 px vessels"),
            ("HRF", "HRF", C_HRF, "~3\u201312 px vessels")]
    rs = 1.4
    row_y = [4.55, 2.95, 1.35]
    col_x = [2.4, 4.6, 6.8, 9.0]
    col_titles = ["RGB", "green channel", "multiscale Frangi", "density map  DM"]
    for cx, t in zip(col_x, col_titles):
        ax.text(cx, row_y[0] + rs / 2 + 0.14, t, ha="center", va="bottom", fontsize=7.8, fontweight="bold")

    for (key, pretty, col, calibre), yy in zip(rows, row_y):
        sample = grp.get(key, [None])[0]
        if sample is None:
            continue
        r = fc.center_square(fc.load_rgb(sample["image_path"]))
        g = r[:, :, 1]
        rawf = fc.center_square(fc.frangi_response(sample))
        d = fc.center_square(fc.density_for(sample))
        smin, smax, nsc = frangi_dataset_scale[key]
        ax.add_patch(Rectangle((0.35, yy - rs / 2), 0.16, rs, fc=col, ec="none", zorder=3))
        ax.text(0.6, yy + 0.30, pretty, fontsize=8.5, fontweight="bold", color=col, ha="left", va="center")
        ax.text(0.6, yy - 0.16, f"$\\sigma \\in$ [{smin}, {smax}]\n{nsc} scales\n{calibre}",
                fontsize=6.8, color=MUTED, ha="left", va="center", linespacing=1.35)
        tile(ax, r, col_x[0], yy, rs, ec=col)
        tile(ax, g, col_x[1], yy, rs, cmap="Greens", ec=col)
        tile(ax, rawf, col_x[2], yy, rs, cmap="magma", ec=col)
        tile(ax, d, col_x[3], yy, rs, cmap="magma", ec=col)
        for i in range(3):
            arrow(ax, (col_x[i] + rs / 2 + 0.02, yy), (col_x[i + 1] - rs / 2 - 0.02, yy), color=col, lw=1.0)

    ax.text(10.0, 2.95,
            "Same five operations for every\ndataset; only the \u03c3 range changes.\n\n"
            "Maps are disk-cached (keyed by\nimage path + Frangi config) and\nreused across every sweep point\n"
            "and seed, so the Frangi cost is\npaid once.",
            fontsize=7.4, color=INK, ha="left", va="center", linespacing=1.4)

    save3(fig, out_dir, "fig4_frangi_density")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data"); ap.add_argument("--out", default="./figures_new")
    a = ap.parse_args(); main(a.data, a.out)
