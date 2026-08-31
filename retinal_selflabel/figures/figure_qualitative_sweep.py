import os
import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from retinal_selflabel.core.datasets import transform_images

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
})

# run the model on a single test image and upsample the prediction
@torch.no_grad()
def _predict_full_size(model, img_path, mask_shape, device, img_size = 512, threshold= 0.5):
    tf = transform_images("validation", img_size)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transformed = tf(image=img, mask=np.zeros(img.shape[:2], dtype=np.uint8))
    x = transformed["image"].unsqueeze(0).to(device)

    model.eval()
    logits = model(x)
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    pred = (probs >= threshold).astype(np.uint8)
    # resize to mask resolution
    pred = cv2.resize(pred, (mask_shape[1], mask_shape[0]), interpolation=cv2.INTER_NEAREST)
    return pred


def _dice(pred, gt):
    p = pred.astype(bool)
    g = gt.astype(bool)
    inter = int((p & g).sum())
    denom = int(p.sum() + g.sum())
    return 2.0 * inter / max(denom, 1)


def _error_map(pred, gt):
    h, w = pred.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    p = pred.astype(bool)
    g = gt.astype(bool)
    tp = p & g
    fp = p & ~g
    fn = ~p & g
    out[tp] = (46, 184, 100)    # green
    out[fp] = (208, 52, 44)     # red
    out[fn] = (36, 110, 197)    # blue
    return out


# core builders
def make_binary_mask_grid(test, models_by_label, label_order, save_path, device,
    img_size = 512, threshold = 0.5, dataset_tag_by_sample = None, column_headers = None):

    n_rows = len(test)
    n_cols = 1 + len(label_order)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.2 * n_cols, 2.4 * n_rows),
        squeeze=False,
    )

    headers = {"input": "Input RGB"}
    for lbl in label_order:
        default = lbl if column_headers is None else column_headers.get(lbl, lbl)
        headers[lbl] = default

    for r, sample in enumerate(test):
        img = cv2.imread(sample["image_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)
        gt = (gt > 127).astype(np.uint8)
        mask_shape = gt.shape

        # input RGB
        ax = axes[r, 0]
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(headers["input"])
        tag = (dataset_tag_by_sample or {}).get(sample.get("_idx", -1), sample.get("dataset", ""))
        ax.set_ylabel(tag, rotation=90, size=10)

        # GT or a prediction
        for c, lbl in enumerate(label_order, start=1):
            ax = axes[r, c]
            if lbl.upper() == "GT":
                ax.imshow(gt, cmap="gray")
                if r == 0:
                    ax.set_title("Ground truth")
            else:
                model = models_by_label[lbl]
                pred = _predict_full_size(model, sample["image_path"], mask_shape,device=device, 
                                          img_size=img_size, threshold=threshold)
                ax.imshow(pred, cmap="gray")
                d = _dice(pred, gt)
                ax.set_xlabel(f"Dice = {d:.3f}", fontsize=8.5)
                if r == 0:
                    ax.set_title(headers[lbl])
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        "Qualitative convergence: predictions vs annotation budget",
        fontsize=12, y=1.0,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# colour-coded error map grid (TP/FP/FN)
def make_error_map_grid(test, models_by_label, label_order, save_path, device, img_size = 512,
                        threshold = 0.5, column_headers = None):
    n_rows = len(test)
    n_cols = 1 + len(label_order)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.4 * n_rows), squeeze=False)

    for r, sample in enumerate(test):
        gt = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)
        gt = (gt > 127).astype(np.uint8)
        mask_shape = gt.shape

        # GT
        ax = axes[r, 0]
        ax.imshow(gt, cmap="gray")
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title("Ground truth")
        ax.set_ylabel(sample.get("dataset", ""), rotation=90, size=10)

        for c, lbl in enumerate(label_order, start=1):
            ax = axes[r, c]
            model = models_by_label[lbl]
            pred = _predict_full_size(model, sample["image_path"], mask_shape, device=device, img_size=img_size, 
                                      threshold=threshold)
            err = _error_map(pred, gt)
            ax.imshow(err)
            d = _dice(pred, gt)
            ax.set_xlabel(f"Dice = {d:.3f}", fontsize=8.5)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                pretty = (lbl if column_headers is None else column_headers.get(lbl, lbl))
                ax.set_title(pretty)

    legend = [
        mpatches.Patch(color=(46/255, 184/255, 100/255), label="TP (correct vessel)"),
        mpatches.Patch(color=(208/255, 52/255, 44/255), label="FP (over-segmentation)"),
        mpatches.Patch(color=(36/255, 110/255, 197/255), label="FN (missed vessel)"),
    ]
    fig.legend(handles=legend, loc="lower center",
               ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Error maps across the random sweep "
        "(green = TP, red = FP, blue = FN)",
        fontsize=12, y=1.0,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

 
# convenience, reconstruct a unet with resnet34 and load weights.
def load_model_from_checkpoint(checkpoint_path, device):
    from retinal_selflabel.core.models import create_model
    model = create_model(architecture="unet", encoder="resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# load test images and trained checkpoints
def run_qualitative_from_checkpoints(data, test_indices, checkpoint, orders, output, titles = None, img_size = 512, device = False, seed = 42):
    from retinal_selflabel.core.datasets import feature_discovery, sample_splitting

    dev = torch.device("cuda" if (torch.cuda.is_available() and not device) else "cpu")

    all_samples = feature_discovery(data)
    _, test_samples_all = sample_splitting(
        all_samples, test_frac=0.2, seed=seed
    )
    selected = [test_samples_all[i] for i in test_indices]

    models_by_label = {}
    for label in orders:
        if label.upper() == "GT":
            continue
        ckpt = checkpoint[label]
        models_by_label[label] = load_model_from_checkpoint(ckpt, dev)

    out_bin = os.path.join(output, "figures", "fig_qualitative_binary.png")
    out_err = os.path.join(output, "figures", "fig_qualitative_errormaps.png")
    make_binary_mask_grid(selected, models_by_label, orders, out_bin, dev,img_size=img_size,
                          column_headers=titles)
    err_order = [l for l in orders if l.upper() != "GT"]
    make_error_map_grid(selected, models_by_label, err_order, out_err, dev, img_size=img_size,
                        column_headers=titles)
    print(f"Figures written to:\n {out_bin}\n {out_err}")

def _cli():
    import argparse
    import json as _json
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint_json", required=True)
    p.add_argument("--orders", nargs="+", required=True)
    p.add_argument("--test_indices", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--titles_json", default=None)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", action="store_true")
    args = p.parse_args()

    with open(args.checkpoint_json) as f:
        ckpt_map = _json.load(f)
    headers = None
    if args.titles_json:
        with open(args.titles_json) as f:
            headers = _json.load(f)

    run_qualitative_from_checkpoints(
        data=args.data,
        test_indices=args.test_indices,
        checkpoint=ckpt_map,
        orders=args.orders,
        output=args.output,
        titles=headers,
        img_size=args.img_size,
        device=args.device,
        seed=args.seed,
    )

if __name__ == "__main__":
    _cli()
