# forbidden-region non-overlap test.
import argparse
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

import retinal_selflabel.figures.fig_common as fc
import retinal_selflabel.figures.figure_utils as fu

fu.apply_ieee_style(8.5)
plt.rcParams.update({"svg.fonttype": "none", "font.family": "serif"})
INK, MUTED = fu.C_TEXT, fu.C_MUTED
BOX = "#e8590c"; SKY = "#1971c2"; GREEN = fu.C_GT; RED = fu.C_RING


def save3(fig, out, stem):
    os.makedirs(out, exist_ok=True); p = os.path.join(out, stem)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{p}.{ext}", bbox_inches="tight", pad_inches=0.05,
                    dpi=300 if ext == "png" else None)
    plt.close(fig); print("saved", p + ".{svg,pdf,png}")


def arrow(ax, p0, p1, color=INK, lw=1.3):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11, lw=lw, 
                                 color=color, shrinkA=2, shrinkB=2, zorder=8))


def panel(ax, M, cx, cy, s, cmap, title, sub=None, vmin=None, vmax=None, grid=False):
    ax.imshow(M, extent=(cx - s / 2, cx + s / 2, cy - s / 2, cy + s / 2), aspect="auto", 
              cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", zorder=2)
    ax.add_patch(Rectangle((cx - s / 2, cy - s / 2), s, s, fill=False, ec="#9aa0aa", lw=0.8, zorder=4))
    ax.text(cx, cy + s / 2 + 0.14, title, ha="center", va="bottom", fontsize=8.2, fontweight="bold")
    if sub:
        ax.text(cx, cy - s / 2 - 0.12, sub, ha="center", va="top", fontsize=7, color=MUTED)
    if grid and M.shape[0] <= 22:
        h, w = M.shape
        for gx in range(w + 1):
            ax.plot([cx - s / 2 + gx / w * s] * 2, [cy - s / 2, cy + s / 2], color="white", lw=0.3, zorder=3)
        for gy in range(h + 1):
            ax.plot([cx - s / 2, cx + s / 2], [cy - s / 2 + gy / h * s] * 2, color="white", lw=0.3, zorder=3)


def main(data_dir, out_dir):
    train, _ = fc.get_split(data_dir)
    demo = fc.pick_demo(train, "CHASE", 0)
    rgb = fc.center_square(fc.load_rgb(demo["image_path"]))
    density_full = fc.center_square(fc.density_for(demo))

    # coarse grid 
    Dsmall = cv2.resize(density_full, (64, 64), interpolation=cv2.INTER_AREA)
    placed = [(8, 10, 16), (34, 40, 18)]
    cand = 16
    tr = fc.sampling_trace(Dsmall, placed, cand, seed=4)

    fig = plt.figure(figsize=(13.2, 6.6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 13.2); ax.set_ylim(0, 6.6)
    ax.set_aspect("equal"); ax.axis("off")
    ax.text(0.25, 6.36, "Strict non-overlap is a 0/1 forbidden matrix + a box-filter overlap test "
            "(not pairwise interval comparison)", fontsize=11, fontweight="bold", ha="left")
    ax.text(0.25, 6.06, "exactly as in frangi_density.samp_in_ds: every candidate top-left"
            "corner is scored on one pass over the grid",
            fontsize=8, color=MUTED, ha="left", style="italic")

    s = 2.0; cy = 4.35
    cxs = [1.45, 4.0, 6.55, 9.1, 11.65]

    # density with already-placed boxes
    panel(ax, Dsmall, cxs[0], cy, s, "magma", "(a) density  D")
    Hs = Dsmall.shape[0]
    for (r, c, bs) in placed:
        x0 = cxs[0] - s / 2 + (c / Hs) * s
        y0 = cy + s / 2 - ((r + bs) / Hs) * s
        ax.add_patch(Rectangle((x0, y0), (bs / Hs) * s, (bs / Hs) * s, fill=False, ec="white", lw=1.3, zorder=6))
    ax.text(cxs[0], cy - s / 2 - 0.12, "2 boxes already accepted", ha="center", va="top", fontsize=7, color=MUTED)

    # forbidden matrix 
    panel(ax, tr["forbidden_before"], cxs[1], cy, s, "Greys", "(b) forbidden  F  (0/1)",
          "1 = covered by a placed box", vmin=0, vmax=1)

    # overlap count
    oc = tr["overlap_count"]
    panel(ax, oc, cxs[2], cy, s, "OrRd", "(c) overlap = boxFilter(F)",
          "px each candidate box would cover", vmin=0, vmax=max(1, oc.max()))

    # valid weights
    panel(ax, tr["valid_weights"], cxs[3], cy, s, "magma",
          "(d) valid = score \u00d7 [overlap=0]", "corners with overlap>0 set to 0")
    if tr["chosen"] is not None:
        r, c = tr["chosen"]; k = tr["cand_size"]
        vh, vw = tr["valid_weights"].shape
        dx = cxs[3] - s / 2 + ((c + 0.5) / vw) * s
        dy = cy + s / 2 - ((r + 0.5) / vh) * s
        ax.plot(dx, dy, "o", ms=5, mec="white", mfc="#39d353", mew=0.8, zorder=7)

    # forbidden after marking the sampled box
    panel(ax, tr["forbidden_after"], cxs[4], cy, s, "Greys", "(e) F after paint",
          "sampled box \u2192 1 (with 1-px buffer)", vmin=0, vmax=1)
    if tr["chosen"] is not None:
        r, c = tr["chosen"]; k = tr["cand_size"]; Gf = tr["forbidden_after"].shape[0]
        x0 = cxs[4] - s / 2 + (c / Gf) * s
        y0 = cy + s / 2 - ((r + k) / Gf) * s
        ax.add_patch(Rectangle((x0, y0), (k / Gf) * s, (k / Gf) * s, fill=False, ec="#39d353", lw=1.8, zorder=6))

    for i in range(4):
        arrow(ax, (cxs[i] + s / 2 + 0.04, cy), (cxs[i + 1] - s / 2 - 0.04, cy))
    ax.text((cxs[1] + cxs[2]) / 2, cy + 0.16, "boxFilter", fontsize=6.8, color=MUTED, ha="center")
    ax.text((cxs[2] + cxs[3]) / 2, cy + 0.16, "reject", fontsize=6.8, color=MUTED, ha="center")
    ax.text((cxs[3] + cxs[4]) / 2, cy + 0.16, "inverse-CDF\nsample", fontsize=6.8, color=MUTED, ha="center", va="bottom")

    # literal 0/1 grid + validity mask 
    ax.text(0.25, 2.78, "The same matrix, literally: a small patch of F printed as 0/1",
            fontsize=8.5, fontweight="bold", ha="left")
    sub = tr["forbidden_before"][4:14, 6:18]
    gx0, gy0, cw, ch = 0.45, 0.75, 0.26, 0.18
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            v = int(sub[i, j])
            ax.add_patch(Rectangle((gx0 + j * cw, gy0 + (sub.shape[0] - 1 - i) * ch), cw, ch,
                         fc=("#1f3a60" if v else "white"), ec="#c2c9d0", lw=0.4, zorder=3))
            ax.text(gx0 + j * cw + cw / 2, gy0 + (sub.shape[0] - 1 - i) * ch + ch / 2, str(v),
                    ha="center", va="center", fontsize=5.6, color=("white" if v else "#9aa0aa"), zorder=4)
    ax.text(gx0, gy0 - 0.16, "overlap test reduces to: does boxFilter(F) over this window exceed 0?",
            fontsize=7.4, color=MUTED, ha="left")

    # union of accepted boxes
    boxes_full = fc.place_boxes(density_full, n=5, seed=7)
    V = fc.union_mask(density_full.shape, boxes_full)
    vx, vy, vs = 9.6, 1.45, 2.2
    ax.text(vx, vy + vs / 2 + 0.16, r"validity mask  V = $\cup$ accepted boxes  (0/1)",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.imshow(V, extent=(vx - vs / 2, vx + vs / 2, vy - vs / 2, vy + vs / 2),
              aspect="auto", cmap="Blues", interpolation="nearest", zorder=2)
    ax.add_patch(Rectangle((vx - vs / 2, vy - vs / 2), vs, vs, fill=False, ec="#9aa0aa", lw=0.8, zorder=3))
    ax.text(vx, vy - vs / 2 - 0.14, f"the masked BCE+Dice loss is evaluated only on V's 1-pixels "
            f"({100*V.mean():.2f}% here)", ha="center", va="top", fontsize=7.2, color=MUTED)

    save3(fig, out_dir, "fig3_nonoverlap")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data"); ap.add_argument("--out", default="./figures_new")
    a = ap.parse_args(); main(a.data, a.out)
