#Four-level random-box sampler diagram
import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

import retinal_selflabel.figures.fig_common as fc
import retinal_selflabel.figures.figure_utils as fu

fu.apply_ieee_style(8.5)
plt.rcParams.update({"svg.fonttype": "none", "font.family": "serif"})

C_DRIVE, C_CHASE, C_HRF = "#1971c2", "#2b8a3e", "#e67700"
INK, MUTED = fu.C_TEXT, fu.C_MUTED
BOX = "#e8590c" 
SKY = "#1971c2"


def save3(fig, out, stem):
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, stem)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{p}.{ext}", bbox_inches="tight", pad_inches=0.05, dpi=300 if ext == "png" 
                    else None)
    plt.close(fig)
    print("saved", p + ".{svg,pdf,png}")


def tile(ax, img, cx, cy, s, label=None, lc=INK, border=fu.C_BORDER, lw=1.0):
    ax.imshow(img, extent=(cx - s / 2, cx + s / 2, cy - s / 2, cy + s / 2),
              aspect="auto", zorder=2, interpolation="lanczos")
    ax.add_patch(Rectangle((cx - s / 2, cy - s / 2), s, s, fill=False,
                 edgecolor=border, lw=lw, zorder=3))
    if label:
        ax.text(cx, cy - s / 2 - 0.10, label, ha="center", va="top", fontsize=7.5, color=lc, 
                fontweight="bold")


def arrow(ax, p0, p1, color=fu.C_BORDER, lw=1.3, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12, lw=lw, color=color, 
                                 shrinkA=1, shrinkB=1, connectionstyle=f"arc3,rad={rad}", zorder=4))


def panel_title(ax, x, y, tag, title, sub):
    ax.text(x, y, tag, fontsize=9.5, fontweight="bold", color=INK, ha="left", va="bottom")
    ax.text(x + 0.30, y, title, fontsize=8.8, fontweight="bold", color=INK, ha="left", va="bottom")
    ax.text(x + 0.30, y - 0.20, sub, fontsize=7.2, color=MUTED, ha="left", va="bottom", style="italic")


def main(data_dir, out_dir):
    train, _ = fc.get_split(data_dir)
    grp = fc.by_dataset(train)

    # representative images
    demo = fc.pick_demo(train, "CHASE", 0)
    rgb = fc.center_square(fc.load_rgb(demo["image_path"]))
    density = fc.density_for(demo)
    density_sq = fc.center_square(density)
    boxes = fc.place_boxes(density_sq, n=6, seed=42)

    fig = plt.figure(figsize=(13, 7.0))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 13); ax.set_ylim(0, 7.0)
    ax.set_aspect("equal"); ax.axis("off")

    ax.text(0.25, 6.78, "The randomizer: a four-level random-box sampler "
            "(dataset \u2192 image \u2192 size \u2192 Frangi-weighted position, strict non-overlap)",
            fontsize=11, fontweight="bold", ha="left")

    # for-loop bracket
    ax.add_patch(FancyBboxPatch((0.25, 3.55), 12.5, 2.95,
                 boxstyle="round,pad=0.02,rounding_size=0.08",
                 fc="#fbfbf9", ec="#d9d4cc", lw=1.1, zorder=1))
    ax.text(0.45, 6.30, "for n = 1 \u2026 N      (N = k\u00b2, k = 1\u202635, swept up to N = 729)",
            fontsize=8.5, fontweight="bold", ha="left")

    cxs = [2.05, 5.0, 8.0, 11.0]

    # sample dataset, p proportional to image count
    panel_title(ax, 0.45, 6.05, "(a)", "sample Dataset", r"p $\propto$ image count")
    cnts = [("DRIVE", len(grp.get("DRIVE", [])) or 16, C_DRIVE),
            ("CHASE", len(grp.get("CHASE", [])) or 22, C_CHASE),
            ("HRF",   len(grp.get("HRF",   [])) or 37, C_HRF)]
    bx = 1.05; base = 4.05; maxc = max(c for _, c, _ in cnts)
    for i, (nm, c, col) in enumerate(cnts):
        bh = 1.15 * c / maxc
        ax.add_patch(Rectangle((bx + i * 0.62, base), 0.46, bh, fc=col, ec="none", alpha=.9, zorder=3))
        ax.text(bx + i * 0.62 + 0.23, base - 0.14, nm, fontsize=7, color=col, fontweight="bold", ha="center")
        ax.text(bx + i * 0.62 + 0.23, base + bh + 0.10, str(c), fontsize=7, color=MUTED, ha="center")
    arrow(ax, (bx + 2 * 0.62 + 0.23, 5.55), (bx + 2 * 0.62 + 0.23, base + 1.15 * cnts[2][1] / maxc + 0.18), color=INK)

    # Sample image, uniform within dataset
    panel_title(ax, 3.55, 6.05, "(b)", "sample image", "uniform within dataset")
    thumbs = grp.get("CHASE", train)[:4]
    tsz = 0.66
    for j, s in enumerate(thumbs):
        try:
            th = fc.center_square(fc.load_rgb(s["image_path"]))
        except Exception:
            th = np.full((10, 10, 3), 200, np.uint8)
        cx = 3.75 + j * 0.78; cy = 4.55
        chosen = (j == 1)
        tile(ax, th, cx, cy, tsz, border=(BOX if chosen else "#c7ccd2"),
             lw=(2.0 if chosen else 0.8))

    # sample boxsize
    panel_title(ax, 6.7, 6.05, "(c)", "sample boxsize", r"s $\sim$ Uniform[32, 256] px")
    cx, cy, s = 7.85, 4.7, 1.5
    tile(ax, rgb, cx, cy, s)
    for frac, a in [(0.95, .35), (0.62, .6), (0.38, .9)]:
        sd = s * frac
        ax.add_patch(Rectangle((cx - sd / 2, cy - sd / 2), sd, sd, fill=False, edgecolor=BOX, 
                               lw=1.5, alpha=a, zorder=5))
    ry = 3.78
    arrow(ax, (cx - 0.85, ry), (cx + 0.85, ry), color=MUTED, lw=1.0)
    for v, fx in [(32, -0.85), (256, 0.85)]:
        ax.plot([cx + fx, cx + fx], [ry - 0.05, ry + 0.05], color=MUTED, lw=1.0)
        ax.text(cx + fx, ry - 0.16, str(v), fontsize=7, color=MUTED, ha="center")
    ax.text(cx, ry - 0.16, "side length (px)", fontsize=7, color=MUTED, ha="center")

    # sample position, Frangi density, non-overlap
    panel_title(ax, 9.7, 6.05, "(d)", "sample position", r"p $\propto$ Frangi density, non-overlap")
    cx, cy, s = 11.0, 4.7, 1.5
    tile(ax, rgb, cx, cy, s)
    # density heatmap overlay
    ax.imshow(density_sq, extent=(cx - s / 2, cx + s / 2, cy - s / 2, cy + s / 2),
              aspect="auto", cmap="magma", alpha=0.45, zorder=3, interpolation="bilinear")
    # non-overlapping placements
    H = density_sq.shape[0]
    for (r, c, bs) in boxes:
        x0 = cx - s / 2 + (c / H) * s
        y0 = cy + s / 2 - (r / H) * s - (bs / H) * s
        ax.add_patch(Rectangle((x0, y0), (bs / H) * s, (bs / H) * s, fill=False,
                     edgecolor=BOX, lw=1.6, zorder=6))

    for i in range(3):
        arrow(ax, (cxs[i] + 1.05, 4.7), (cxs[i + 1] - 1.05, 4.7), color=INK)

    # output of one full draw
    arrow(ax, (11.0, 3.55), (11.0, 3.05), color=INK)
    ax.text(11.18, 3.30, "repeat N times", fontsize=7.2, color=MUTED, ha="left")
    fu.draw_box(ax, 9.4, 2.35, 3.2, 0.62, "returns BoxPlacement(sample_idx, dataset,\nimage_id, row, col, size)",
                face="#eef3f7", edge=SKY, text_color="#0f3b57", fontsize=7.4)
    ax.text(1.5, 2.75, "Each draw is independent; the only human cost is later labelling the boxes' interiors.",
            fontsize=7.8, color=INK, ha="left", fontweight="bold")
    ax.text(1.5, 2.45, "Levels 1-2 set WHICH image; level 3 sets HOW BIG; level 4 sets WHERE (density-weighted, non-overlapping).",
            fontsize=7.4, color=MUTED, ha="left", style="italic")

    # innovations chips
    ax.text(0.25, 1.45, "Three innovations layered on the uniform baseline:",
            fontsize=8.5, fontweight="bold", ha="left")
    chips = [("Random", "free placement", "#eef0f2", "#c7ccd2"),
             ("+ Non-Overlap", "strict, same image", "#eaf6fb", SKY),
             ("+ Info Density", "Frangi-weighted", "#fff3df", BOX)]
    cx = 0.25
    for i, (t, s_, fcl, ec) in enumerate(chips):
        ax.add_patch(FancyBboxPatch((cx, 0.45), 2.7, 0.62,
                     boxstyle="round,pad=0.02,rounding_size=0.05", fc=fcl, ec=ec, lw=1.3, zorder=2))
        ax.text(cx + 1.35, 0.86, t, fontsize=8.5, fontweight="bold", ha="center", color=INK)
        ax.text(cx + 1.35, 0.60, s_, fontsize=7.2, color=MUTED, ha="center")
        if i < 2:
            arrow(ax, (cx + 2.7, 0.76), (cx + 3.0, 0.76), color=INK, lw=1.1)
        cx += 3.0
    ax.text(cx + 0.1, 0.86, r"constraints:  size $\in$ [32,256] px,", fontsize=7.6, ha="left", color=INK)
    ax.text(cx + 0.1, 0.62, "box \u2264 50% of min image side (\u2264 1 Mpx)", fontsize=7.6, ha="left", color=INK)
    ax.text(0.25, 0.18, "Sparse baseline = the N=75, fixed 128\u00d7128, uniform-position special case.",
            fontsize=7.2, color=MUTED, ha="left", style="italic")

    save3(fig, out_dir, "fig2a_randomizer")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="./figures_new")
    a = ap.parse_args()
    main(a.data, a.out)
