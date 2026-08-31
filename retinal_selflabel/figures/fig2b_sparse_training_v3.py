# sparse-supervision pipeline diagram.
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
INK, MUTED = fu.C_TEXT, fu.C_MUTED
C_RGB, C_GT, C_PRED, BOX, SKY = "#1971c2", fu.C_GT, fu.C_RING, "#e8590c", "#1971c2"


def save3(fig, out, stem):
    os.makedirs(out, exist_ok=True); p = os.path.join(out, stem)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{p}.{ext}", bbox_inches="tight", pad_inches=0.05,
                    dpi=300 if ext == "png" else None)
    plt.close(fig); print("saved", p + ".{svg,pdf,png}")


def arrow(ax, p0, p1, color=INK, lw=1.5, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13,
                 lw=lw, color=color, shrinkA=2, shrinkB=2, linestyle=ls,
                 connectionstyle=f"arc3,rad={rad}", zorder=7))


def tile(ax, img, cx, cy, s, cmap=None, label=None, lc=INK, ec=fu.C_BORDER):
    ax.imshow(img, extent=(cx - s / 2, cx + s / 2, cy - s / 2, cy + s / 2),
              aspect="auto", cmap=cmap, interpolation="lanczos", zorder=2)
    ax.add_patch(Rectangle((cx - s / 2, cy - s / 2), s, s, fill=False, ec=ec, lw=1.0, zorder=3))
    if label:
        ax.text(cx, cy - s / 2 - 0.11, label, ha="center", va="top",
                fontsize=7.3, color=lc, fontweight="bold")

def try_predict(ckpt, rgb_u8):
    if not ckpt:
        return None
    try:
        import cv2
        import torch

        from retinal_selflabel.core.models import create_model
        m = create_model("unet", "resnet34", None, 3, 1)
        sd = torch.load(ckpt, map_location="cpu")
        m.load_state_dict(sd.get("model", sd), strict=False); m.eval()
        x = cv2.resize(rgb_u8, (512, 512)).astype(np.float32) / 255.0
        t = torch.from_numpy(x.transpose(2, 0, 1))[None]
        with torch.no_grad():
            p = torch.sigmoid(m(t))[0, 0].numpy()
        return cv2.resize(p, (rgb_u8.shape[1], rgb_u8.shape[0])) > 0.5
    except Exception as e:
        print("inference skipped", str(e)[:80]); return None


def main(data_dir, out_dir, ckpt):
    train, test = fc.get_split(data_dir)
    grp = fc.by_dataset(train)
    demo = fc.pick_demo(train, "CHASE", 0)
    rgb = fc.center_square(fc.load_rgb(demo["image_path"]))
    gt = fc.center_square(fc.load_gt(demo["mask_path"]))
    density = fc.center_square(fc.density_for(demo))
    boxes = fc.place_boxes(density, n=6, seed=42)
    V = fc.union_mask(density.shape, boxes).astype(bool)

    # masked RGB and masked GT (V applied)
    rgb_masked = (rgb.astype(np.float32) * V[..., None]).astype(np.uint8)
    gt_rgb = fu.overlay_mask_rgb(np.zeros((*gt.shape, 3), np.uint8), gt, fu.hex_to_rgb01(C_GT), alpha=1.0)
    gt_masked = (gt_rgb * V[..., None])
    test_demo = fc.pick_demo(test if test else train, "CHASE", 0)
    trgb = fc.center_square(fc.load_rgb(test_demo["image_path"]))
    tgt = fc.center_square(fc.load_gt(test_demo["mask_path"]))
    tgt_disp = fu.overlay_mask_rgb(np.zeros((*tgt.shape, 3), np.uint8), tgt, fu.hex_to_rgb01(C_GT), alpha=1.0)

    fig = plt.figure(figsize=(13.5, 6.6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 13.5); ax.set_ylim(0, 6.6)
    ax.set_aspect("equal"); ax.axis("off")
    ax.text(0.25, 6.36, "Sparse self-labelling \u2014 the training algorithm that uses the randomizer",
            fontsize=11, fontweight="bold", ha="left")

    yT, yB = 4.55, 1.55

    # data
    ax.text(0.95, 5.7, "Data (93)", fontsize=8, fontweight="bold", color=MUTED, ha="center")
    th = fc.center_square(fc.load_rgb(grp.get("DRIVE", train)[0]["image_path"]))
    tile(ax, th, 0.95, yT, 0.58, label="RGB", lc=C_RGB)
    gth = fc.center_square(fc.load_gt(grp.get("DRIVE", train)[0]["mask_path"]))
    tile(ax, gth.astype(float), 0.95, yB, 0.58, cmap="Greens", label="GT", lc=C_GT)
    arrow(ax, (1.3, yT), (2.0, yT)); arrow(ax, (1.3, yB), (2.0, yB))

    # split
    ax.text(2.6, 5.7, "Split (1)", fontsize=8, fontweight="bold", color=MUTED, ha="center")
    fu.draw_box(ax, 2.05, yT - 0.38, 1.1, 0.76, "Train\n63", face="#eaf3f7", edge=C_RGB, text_color="#1B3F6B", fontsize=7.5)
    fu.draw_box(ax, 2.05, yB - 0.38, 1.1, 0.76, "Test\n18", face="#fdf0e8", edge=C_PRED, text_color="#7A2D12", fontsize=7.5)

    # randomizer
    ax.text(4.55, 5.7, "Train (2)", fontsize=9, fontweight="bold", color=INK, ha="center")
    arrow(ax, (3.15, yT), (3.75, yT))
    ax.add_patch(FancyBboxPatch((3.8, yT - 0.55), 1.5, 1.1, boxstyle="round,pad=0.02,rounding_size=0.05", 
                                fc="#fff3df", ec=BOX, lw=1.4, zorder=2))
    ax.text(4.55, yT + 0.40, "randomizer", fontsize=8.2, fontweight="bold", color="#9a4a06", ha="center")
    ax.text(4.55, yT + 0.05, "4-level sampler\n(Fig 2a)\ndataset\u2192image\n\u2192size\u2192position", fontsize=6.6,
            color="#9a4a06", ha="center", va="center", linespacing=1.25)
    ax.text(4.55, yT - 0.42, "\u00d7 N draws", fontsize=7, color=BOX, ha="center", fontweight="bold")

    # union of boxes
    arrow(ax, (5.3, yT), (5.95, yT))
    tile(ax, V.astype(float), 6.5, yT, 1.0, cmap="Blues", label=r"V = $\cup$ boxes (0/1)", lc=SKY)

    # validity mask applied to rgb and gt
    arrow(ax, (7.05, yT), (7.65, yT))
    tile(ax, rgb_masked, 8.2, yT + 0.55, 0.9, label=None, ec=C_RGB)
    ax.text(8.2, yT + 1.05, r"RGB $\odot$ V", fontsize=6.8, color=C_RGB, ha="center", fontweight="bold")
    tile(ax, gt_masked, 8.2, yT - 0.55, 0.9, label=None, ec=C_GT)
    ax.text(8.2, yT - 1.04, r"GT $\odot$ V", fontsize=6.8, color=C_GT, ha="center", fontweight="bold")

    # u-net masked loss
    arrow(ax, (8.65, yT + 0.55), (9.25, yT + 0.15))
    arrow(ax, (8.65, yT - 0.55), (9.25, yT - 0.15))
    fu.draw_box(ax, 9.3, yT - 0.6, 1.85, 1.2, "U-Net\nFull image forward pass\nloss computed only on V\n(\u201cmask the loss,\nnot the image\u201d)",
                face=fu.C_UNET_F, edge=fu.C_UNET, text_color="#7A4708", fontsize=7.3)
    arrow(ax, (11.15, yT), (11.75, yT))
    fu.draw_box(ax, 11.8, yT - 0.3, 0.9, 0.6, "M*", face="#fff3bf", edge=fu.C_UNET, text_color="#7A4708", fontsize=9)

    # testing band
    ax.text(4.55, 2.15, "Testing (3)", fontsize=9, fontweight="bold", color=INK, ha="center")
    arrow(ax, (2.6, yB), (3.5, yB), color=C_PRED)
    tile(ax, trgb, 4.05, yB, 0.9, ec=C_PRED, label="test RGB", lc=C_PRED)
    tile(ax, tgt_disp, 5.95, yB, 0.9, label="test GT", lc=C_GT)
    # test rgb with m* inference
    arrow(ax, (4.5, yB + 0.5), (12.0, yT - 0.32), color=C_PRED, rad=0.16, ls=(0, (4, 2)))
    ax.text(6.6, 2.7, "test images \u2192 M* (inference, full-res, sigmoid \u2265 0.5)",
            fontsize=7, color=C_PRED, ha="left", style="italic")

    # prediction
    pred = try_predict(ckpt, trgb)
    if pred is None:
        disp = fu.overlay_mask_rgb(trgb.astype(np.uint8), tgt, fu.hex_to_rgb01(C_PRED), alpha=0.85)
        cap = "Prediction\n(target shown)"
    else:
        disp = fu.overlay_mask_rgb(trgb.astype(np.uint8), pred, fu.hex_to_rgb01(C_PRED), alpha=0.85)
        cap = "Prediction (model)"
    tile(ax, disp, 8.4, yB, 1.0, ec=C_PRED)
    ax.text(8.4, yB - 0.57, cap, fontsize=7, color=C_PRED, ha="center", va="top", fontweight="bold")
    arrow(ax, (12.25, yT - 0.32), (8.9, yB + 0.05), color=C_PRED, rad=0.2)

    # accuracy
    fu.draw_box(ax, 10.6, yB - 0.34, 2.5, 0.68,
                "Accuracy \u2014 held-out Dice\n0.6890 (1 box) \u2192 0.7148 (self-labelled)", face="#d6ead9", edge=C_GT, text_color="#0A4D1F", fontsize=7.4)
    arrow(ax, (5.95, yB - 0.52), (10.55, yB - 0.3), color=C_GT, rad=0.42)
    ax.text(8.3, 0.42, "test GT (Dice comparison)", fontsize=6.8, color=C_GT, ha="center", style="italic")
    save3(fig, out_dir, "fig2b_sparse_training")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data"); ap.add_argument("--out", default="./figures_new")
    ap.add_argument("--ckpt", default=None)
    a = ap.parse_args(); main(a.data, a.out, a.ckpt)
