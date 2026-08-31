# full-supervision protocol diagram.
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
C_RGB, C_GT, C_PRED = "#1971c2", fu.C_GT, fu.C_RING


def save3(fig, out, stem):
    os.makedirs(out, exist_ok=True); p = os.path.join(out, stem)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{p}.{ext}", bbox_inches="tight", pad_inches=0.05,
                    dpi=300 if ext == "png" else None)
    plt.close(fig); print("  saved", p + ".{svg,pdf,png}")


def arrow(ax, p0, p1, color=INK, lw=1.5, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13,
                 lw=lw, color=color, shrinkA=2, shrinkB=2, linestyle=ls,
                 connectionstyle=f"arc3,rad={rad}", zorder=7))


def tile(ax, img, cx, cy, s, cmap=None, label=None, lc=INK, ec=fu.C_BORDER):
    ax.imshow(img, extent=(cx - s / 2, cx + s / 2, cy - s / 2, cy + s / 2),
              aspect="auto", cmap=cmap, interpolation="lanczos", zorder=2)
    ax.add_patch(Rectangle((cx - s / 2, cy - s / 2), s, s, fill=False, ec=ec, lw=1.0, zorder=3))
    if label:
        ax.text(cx, cy - s / 2 - 0.12, label, ha="center", va="top",
                fontsize=7.5, color=lc, fontweight="bold")


def stack(ax, x, y, w, h, color, n=3, gap=0.045):
    for i in range(n - 1, -1, -1):
        ax.add_patch(FancyBboxPatch((x + i * gap, y + i * gap), w, h, boxstyle="round,pad=0.0,rounding_size=0.03",
                                    fc=color, ec="white", lw=0.8, alpha=0.95, zorder=3 + (n - i)))

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
    demo = fc.pick_demo(test if test else train, "CHASE", 0)
    rgb = fc.center_square(fc.load_rgb(demo["image_path"]))
    gt = fc.center_square(fc.load_gt(demo["mask_path"]))
    gt_disp = fu.overlay_mask_rgb(np.zeros((*gt.shape, 3), np.uint8), gt, fu.hex_to_rgb01(C_GT), alpha=1.0)

    fig = plt.figure(figsize=(13, 6.2))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 13); ax.set_ylim(0, 6.2)
    ax.set_aspect("equal"); ax.axis("off")
    ax.text(0.25, 5.98, "Full supervision \u2014 upper-bound reference (loss over every pixel, V \u2261 1)",
            fontsize=11, fontweight="bold", ha="left")

    yT, yB = 4.35, 1.5        # train band, test band

    # data
    ax.text(1.0, 5.55, "Data (93)", fontsize=8, fontweight="bold", color=MUTED, ha="center")
    stack(ax, 0.55, yT - 0.35, 0.9, 0.7, "#cfe0f5")
    th = fc.center_square(fc.load_rgb(grp.get("DRIVE", train)[0]["image_path"]))
    tile(ax, th, 1.0, yT, 0.6); ax.text(1.0, yT - 0.42, "RGB", fontsize=7, color=C_RGB, fontweight="bold", ha="center")
    stack(ax, 0.55, yB - 0.35, 0.9, 0.7, "#cdebd6")
    gth = fc.center_square(fc.load_gt(grp.get("DRIVE", train)[0]["mask_path"]))
    tile(ax, gth.astype(float), 1.0, yB, 0.6, cmap="Greens")
    ax.text(1.0, yB - 0.42, "GT", fontsize=7, color=C_GT, fontweight="bold", ha="center")

    # data to split arrow
    arrow(ax, (1.55, yT), (2.35, yT)); arrow(ax, (1.55, yB), (2.35, yB))

    # split
    ax.text(3.0, 5.55, "Stratified split", fontsize=8, fontweight="bold", color=MUTED, ha="center")
    fu.draw_box(ax, 2.4, yT - 0.4, 1.25, 0.8, "Train\n63 imgs\n(RGB+GT)", face="#eaf3f7", edge=C_RGB, text_color="#1B3F6B", fontsize=7.5)
    fu.draw_box(ax, 2.4, yB - 0.4, 1.25, 0.8, "Test\n18 held-out\n(RGB+GT)", face="#fdf0e8", edge=C_PRED, text_color="#7A2D12", fontsize=7.5)

    # train
    ax.text(5.55, 5.55, "Train (2)", fontsize=9, fontweight="bold", color=INK, ha="center")
    arrow(ax, (3.65, yT), (4.45, yT))
    fu.draw_box(ax, 4.5, yT - 0.55, 1.7, 1.1,
                "U-Net\nResNet-34 / ImageNet\nBCE + Dice\nloss over EVERY pixel",
                face=fu.C_UNET_F, edge=fu.C_UNET, text_color="#7A4708", fontsize=7.6)
    arrow(ax, (6.2, yT), (6.95, yT))
    fu.draw_box(ax, 7.0, yT - 0.32, 0.95, 0.64, "M*\ntrained", face="#fff3bf", edge=fu.C_UNET, text_color="#7A4708", fontsize=8.5)
    arrow(ax, (7.95, yT), (8.7, yT))

    pred = try_predict(ckpt, rgb)
    if pred is None:
        disp = fu.overlay_mask_rgb(rgb.astype(np.uint8), gt, fu.hex_to_rgb01(C_PRED), alpha=0.85)
        cap = "Prediction\n(target shown))"
    else:
        disp = fu.overlay_mask_rgb(rgb.astype(np.uint8), pred, fu.hex_to_rgb01(C_PRED), alpha=0.85)
        cap = "Prediction (model output)"
    tile(ax, disp, 9.35, yT, 1.15, ec=C_PRED)
    ax.text(9.35, yT - 0.66, cap, fontsize=7.2, color=C_PRED, fontweight="bold", ha="center", va="top")

    # testing
    ax.text(5.55, 2.05, "Testing (3)", fontsize=9, fontweight="bold", color=INK, ha="center")
    # test rgb
    tile(ax, rgb, 4.5, yB, 0.95, ec=C_PRED, label="test RGB")
    # test gt
    tile(ax, gt_disp, 6.4, yB, 0.95, label="test GT", lc=C_GT)

    # Test box to test RGB
    arrow(ax, (3.05, yB), (3.95, yB), color=C_PRED)
    # test RGB to up into M* 
    arrow(ax, (4.5, yB + 0.55), (7.1, yT - 0.36), color=C_PRED, rad=0.18, ls=(0, (4, 2)))
    ax.text(5.3, 2.95, "test images \u2192 M* (inference,\nfull-res forward, sigmoid \u2265 0.5)",
            fontsize=7, color=C_PRED, ha="left", style="italic")

    # accuracy
    fu.draw_box(ax, 10.7, yB - 0.32, 1.8, 0.64, "Accuracy\nDice = 0.7434 (ceiling)", face="#d6ead9", edge=C_GT, text_color="#0A4D1F", fontsize=8.5)
    # prediction to accuracy
    arrow(ax, (9.35, yT - 0.62), (11.0, yB + 0.33), color=C_PRED, rad=-0.18)
    ax.text(9.7, 2.75, "prediction", fontsize=6.8, color=C_PRED, ha="left", style="italic")
    # test gt to accuracy
    arrow(ax, (6.9, yB), (10.6, yB), color=C_GT, rad=-0.05)
    ax.text(8.4, yB + 0.12, "test GT (for the Dice comparison)", fontsize=6.8, color=C_GT, ha="center", style="italic")

    save3(fig, out_dir, "fig1_full_supervision")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data"); ap.add_argument("--out", default="./figures_new")
    ap.add_argument("--ckpt", default=None)
    a = ap.parse_args(); main(a.data, a.out, a.ckpt)
