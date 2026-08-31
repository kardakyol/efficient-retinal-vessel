import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

COL_SINGLE = 3.50  
COL_DOUBLE = 7.25

C_GT        = "#2b8a3e"   # real ground-truth
C_GT_FILL   = "#b2f2bb"   # gt
C_PSEUDO    = "#1971c2"   # accepted pseudo-label
C_PSEUDO_F  = "#a5d8ff"   # light pseudo fill
C_RING      = "#c92a2a"   # expansion ring
C_RING_F    = "#ffc9c9"

C_UNET      = "#e67700"   
C_UNET_F    = "#fff3bf"
C_LOSS      = "#5f3dc4"   
C_LOSS_F    = "#e5dbff"

C_TEXT      = "#1a1a2e"   
C_MUTED     = "#555577"   
C_BORDER    = "#2c2c2c"   
C_PANEL_BG  = "#f8f9fa"

C_FULL_LINE  = "#2b8a3e"  
C_SPARSE_L   = "#c92a2a" 
C_SL_LINE    = "#1971c2"  


def apply_ieee_style(base_fontsize = 8.5):
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset":   "stix",
        "font.size":          base_fontsize,
        "axes.titlesize":     base_fontsize + 0.5,
        "axes.labelsize":     base_fontsize,
        "xtick.labelsize":    base_fontsize - 1.0,
        "ytick.labelsize":    base_fontsize - 1.0,
        "legend.fontsize":    base_fontsize - 1.5,
        "figure.titlesize":   base_fontsize + 1.5,
        "axes.edgecolor":     C_BORDER,
        "axes.linewidth":     0.8,
        "axes.labelcolor":    C_TEXT,
        "axes.titlecolor":    C_TEXT,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "lines.linewidth":    1.4,
        "lines.markersize":   4.5,
        "xtick.color":        C_BORDER,
        "ytick.color":        C_BORDER,
        "xtick.major.width":  0.8,
        "ytick.major.width":  0.8,
        "xtick.major.size":   3.5,
        "ytick.major.size":   3.5,
        "axes.grid":          True,
        "grid.color":         "#e0e0e0",
        "grid.linewidth":     0.45,
        "grid.alpha":         0.7,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
        "figure.facecolor":   "white",
        "savefig.facecolor":  "white",
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
    })


# export
def save_figure(fig, base_path, also_pdf = True):
    stem, _ = os.path.splitext(base_path)
    png_path = f"{stem}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"png {png_path}")
    if also_pdf:
        pdf_path = f"{stem}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
        print(f"pdf {pdf_path}")
    plt.close(fig)


# drawing primitives
def draw_image_tile(ax, img, x, y, w, h, *, border = C_BORDER, border_lw = 1.0, label = None, 
                    label_fontsize = 7.5, label_color = C_TEXT, label_pad = 0.012, zorder = 2):
    ax.imshow(img, extent=(x, x + w, y, y + h), aspect="auto", zorder=zorder, interpolation="lanczos")
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=border, linewidth=border_lw, zorder=zorder + 1))
    if label:
        ax.text(x + w / 2, y + h + label_pad, label, ha="center", va="bottom", fontsize=label_fontsize, 
                color=label_color, fontweight="bold", zorder=zorder + 2)


def draw_box(ax, x, y, w, h, text, *, face = C_PANEL_BG, edge = C_BORDER,
             lw = 0.9, text_color = C_TEXT, fontsize = 7.5, bold = True, pad  = 0.008,
             rounding = 0.010):
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad},rounding_size={rounding}",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=2)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=text_color,
            fontsize=fontsize, fontweight="bold" if bold else "normal", zorder=3, multialignment="center")

def arrow(ax, x0, y0, x1, y1, *, color = C_BORDER, lw = 1.0, style = "-|>", mutation_scale = 11,
          connectionstyle = "arc3,rad=0", zorder = 3):
    ax.add_patch(FancyArrowPatch( (x0, y0), (x1, y1), arrowstyle=style, color=color, lw=lw, mutation_scale=mutation_scale,
        connectionstyle=connectionstyle, zorder=zorder))


# helpers
def overlay_mask_rgb(image_rgb_u8, mask_bool, color_rgb, alpha = 0.55):
    out = image_rgb_u8.astype(np.float32) / 255.0
    if mask_bool.any():
        c = np.array(color_rgb).reshape(1, 1, 3)
        out = np.where(mask_bool[..., None], out * (1 - alpha) + c * alpha, out)
    return np.clip(out, 0.0, 1.0)

def hex_to_rgb01(hex_str):
    h = hex_str.lstrip("#")
    return (int(h[0:2], 16) / 255.0,
            int(h[2:4], 16) / 255.0,
            int(h[4:6], 16) / 255.0)