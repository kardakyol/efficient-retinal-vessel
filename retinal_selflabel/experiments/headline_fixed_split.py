#re-run of the headline arms on a fixed partition

import argparse, copy, json, os, random, time
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tud
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from retinal_selflabel.core.datasets import (
    RetinalVesselDataset, SparseAnnotationSimulator, SparsePatchDataset,
    feature_discovery, get_sparse_training_transforms, transform_images, dataset_splitting,
)
from retinal_selflabel.core.full_image_sparse_dataset import (
    FullImageSparseDataset, get_full_image_sparse_transforms,
)
from retinal_selflabel.core.models import create_loss, create_model
from retinal_selflabel.core.random_box_sampler import BoxPlacement
from retinal_selflabel.core.train import (
    evaluate, run_full_supervision, run_sparse_baseline,
)
from retinal_selflabel.experiments.ablations import per_dataset_evaluation
from retinal_selflabel.experiments.run_selflabel import run_self_labelling

# determinism
def set_seed_strict(seed, deterministic_algos = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic_algos:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class SeededLoaders:
    def __init__(self, seed, enable=True):
        self.seed, self.enable = seed, enable
        self._orig, self._patched = None, []

    def __enter__(self):
        if not self.enable:
            return self
        self._orig = tud.DataLoader
        gen = torch.Generator()
        gen.manual_seed(self.seed)
        orig = self._orig

        class Seeded(orig):
            def __init__(s, *a, **kw):
                if kw.get("num_workers", 0) and "worker_init_fn" not in kw:
                    kw["worker_init_fn"] = _seed_worker
                if kw.get("shuffle", False) and "generator" not in kw:
                    kw["generator"] = gen
                super().__init__(*a, **kw)

        tud.DataLoader = Seeded
        import sys
        for mod in list(sys.modules.values()):
            if mod is not None and getattr(mod, "DataLoader", None) is orig:
                setattr(mod, "DataLoader", Seeded)
                self._patched.append(mod)
        return self

    def __exit__(self, *exc):
        if not self.enable:
            return False
        tud.DataLoader = self._orig
        for mod in self._patched:
            setattr(mod, "DataLoader", self._orig)
        return False


# loss-masked arm
def masked_bce_dice(logits, target, validity, eps=1e-6):
    v = validity
    den = v.sum().clamp_min(1.0)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * v).sum() / den
    p = torch.sigmoid(logits) * v
    t = target * v
    dice = 1.0 - (2.0 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)
    return bce + dice


def train_lossmasked(train_s, loader_val, loader_test, simulator, device,
                     img_size, num_epochs, patience, lr=1e-3):
    placements = {}
    for info in simulator.patch_info:
        sample_idx = info["sample_idx"]
        sample = train_s[sample_idx]
        height, weight = info.get("image_shape", (None, None))
        if height is None:
            mask = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)
            height, weight = mask.shape[:2]
        placements[sample_idx] = [BoxPlacement(sample_idx=sample_idx, dataset=sample["dataset"], 
                                               image_id=sample["id"], image_h=int(height), image_w=int(weight),
                                               row=int(r), col=int(c), size=128) for (r, c) in info["patches"]]
    dataset = FullImageSparseDataset(
        train_s, placements, img_size=img_size,
        transform=get_full_image_sparse_transforms(img_size, train=True))
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)

    model = create_model("unet", "resnet34", "imagenet", 3, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    criterion = create_loss("bce_dice")

    best, best_weights, stale = 0.0, copy.deepcopy(model.state_dict()), 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        for batch in loader:
            img, gt, val = (batch[0].to(device), batch[1].to(device), batch[2].to(device))
            if val.dim() == 3:
                val = val.unsqueeze(1)
            optimizer.zero_grad()
            loss = masked_bce_dice(model(img), gt, val)
            loss.backward()
            optimizer.step()
        scheduler.step()
        val_dice = evaluate(model, loader_val, criterion, device)["dice"]
        if val_dice > best:
            best, best_weights, stale = val_dice, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if epoch % 20 == 0 or epoch == 1:
            print(f"lossmask ep {epoch:3d}, val dice {val_dice:.4f}, (best {best:.4f})")
        if stale >= patience:
            print(f"lossmask early stop at epoch {epoch}")
            break
    model.load_state_dict(best_weights)
    return model, best


def make_args(seed, data_dir, out_dir, img_size):
    return SimpleNamespace(data_dir=data_dir, out_dir=out_dir, mode="all",
                           img_size=img_size, patch_size=128, expand_px=16, 
                           confidence_threshold=0.7, finetune_epochs=15, 
                           pseudo_weight=0.5, max_iterations=50, selflabel_patience=8,
                           epochs_baseline=100, use_cldice=False, seed=seed, 
                           no_cuda=False, skip_patch_ablation=True, 
                           skip_pretrained_ablation=True, skip_threshold_ablation=True)

ARM_KEY = {"full": "full_supervision", "sparse": "sparse_baseline",
           "lossmask": "loss_masked", "selflabel": "self_labelling"}

def run_one(seed, split_seed, arms, args_cli, repeat=0):
    tag = args_cli.tag or ("det" if args_cli.deterministic else "nondet")
    run_id = f"{tag}_split{split_seed}_seed{seed}_r{repeat}"
    out_path = os.path.join(args_cli.out_dir, f"{run_id}.json")

    prior = {}
    if os.path.exists(out_path) and not args_cli.force:
        prior = json.load(open(out_path)) or {}
        missing = [a for a in arms if ARM_KEY[a] not in prior]
        if not missing:
            print(f"skip, all requested arms present {run_id}")
            return prior
        print(f"[resume] {run_id}: {sorted(prior.get('arms', []))} on disk missing {missing} re-running it")

    start_time = time.time()
    dirs = {k: os.path.join(args_cli.out_dir, "work", run_id, k)
            for k in ("checkpoints", "figures", "logs")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    device = torch.device("cuda" if (torch.cuda.is_available() and not args_cli.no_cuda) else "cpu")
    set_seed_strict(seed, args_cli.deterministic)

    samples = feature_discovery(args_cli.data_dir)
    train, val, test = dataset_splitting(samples, val_frac=0.15, test_frac=0.2, seed=split_seed)
    print(f"# {run_id}")
    print(f"# partition seed {split_seed} (pinned)  |  model seed {seed}  |  repeat {repeat}")
    print(f"# train {len(train)}  val {len(val)}  test {len(test)}  |  arms {' '.join(arms)}  deterministic algorithms: {args_cli.deterministic}")

    sl_args = make_args(seed, args_cli.data_dir, args_cli.out_dir, args_cli.img_size)
    criterion = create_loss("bce_dice")
    out = dict(run_id=run_id, model_seed=seed, split_seed=split_seed, repeat=repeat,
               deterministic=bool(args_cli.deterministic), arms=list(arms),
               n_train=len(train), n_val=len(val), n_test=len(test),
               test_ids=sorted(f"{x['dataset']}/{x['id']}" for x in test))

    with SeededLoaders(seed, enable=True):
        make_eval_loader = lambda samples_subset: DataLoader(
            RetinalVesselDataset(samples_subset, transform=transform_images("validation", sl_args.img_size)),
            batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
        loader_val, loader_test = make_eval_loader(val), make_eval_loader(test)

        model_sparse = None
        simulator = None

        if "full" in arms:
            print("\n arm: full supervision")
            tr_loader = DataLoader(
                RetinalVesselDataset(train, transform=transform_images("training", sl_args.img_size)),
                batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
            model, _ = run_full_supervision(tr_loader, loader_val, device, num_epochs=sl_args.epochs_baseline, ckpt_dir=dirs["checkpoints"])
            out["full_supervision"] = {k: float(v) for k, v in evaluate(model, loader_test, criterion, device).items()}
            out["full_supervision_per_dataset"] = per_dataset_evaluation( model, test, device, sl_args.img_size)
            del model
            torch.cuda.empty_cache()

        if any(a in arms for a in ("sparse", "selflabel", "lossmask")):
            simulator = SparseAnnotationSimulator(train, patch_size=sl_args.patch_size, patches_per_image=1, seed=seed)
            out["annotation_coverage"] = float(simulator.get_annotation_coverage())

        if "sparse" in arms or "selflabel" in arms:
            print("\narm: sparse baseline")
            sp_loader = DataLoader(SparsePatchDataset(train, simulator.patch_info, sl_args.patch_size,transform=get_sparse_training_transforms()), batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
            model_sparse, _ = run_sparse_baseline(sp_loader, loader_val, device,num_epochs=sl_args.epochs_baseline + 50, ckpt_dir=dirs["checkpoints"])
            out["sparse_baseline"] = {k: float(v) for k, v in evaluate(model_sparse, loader_test, criterion, device).items()}
            out["sparse_baseline_per_dataset"] = per_dataset_evaluation(model_sparse, test, device, sl_args.img_size)

        if "lossmask" in arms:
            print("\arm: loss-masked full-image")
            model, best_val = train_lossmasked(train, loader_val, loader_test, simulator, device, sl_args.img_size, sl_args.epochs_baseline + 50, patience=20)
            out["loss_masked"] = {k: float(v) for k, v in evaluate(model, loader_test, criterion, device).items()}
            out["loss_masked_per_dataset"] = per_dataset_evaluation(model, test, device, sl_args.img_size)
            out["loss_masked_best_val"] = float(best_val)
            del model
            torch.cuda.empty_cache()

        if "selflabel" in arms:
            print("\n arm: incremental self-labelling")
            model, metrics, iteration_log, _ = run_self_labelling(model_sparse, train, val, test, simulator, device, sl_args, dirs)
            out["self_labelling"] = {k: float(v) for k, v in metrics.items()}
            out["self_labelling_per_dataset"] = per_dataset_evaluation(model, test, device, sl_args.img_size)
            out["iteration_log"] = iteration_log
            del model
            torch.cuda.empty_cache()

        del model_sparse
        torch.cuda.empty_cache()

    out["wall_clock_seconds"] = time.time() - start_time
    if prior:
        merged = dict(prior)
        merged.update(out)
        merged["arms"] = sorted(set(prior.get("arms", [])) | set(arms))
        out = merged
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n written to {out_path} and ({out['wall_clock_seconds'] / 60:.1f} min)")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--arms", nargs="+", default=["full", "sparse", "selflabel"], choices=["full", "sparse", "selflabel", "lossmask"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7, 2024, 31337])
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    runs = []
    for rep in range(args.repeats):
        for seed in args.seeds:
            runs.append(run_one(seed, args.split_seed, args.arms, args, repeat=rep))

    print(f"partition pinned at split_seed={args.split_seed}")
    keys = [("full_supervision", "full supervision"),
            ("sparse_baseline", "sparse baseline (crop)"),
            ("loss_masked", "loss-masked full image"),
            ("self_labelling", "self-labelling")]
    cols = {}
    for k, label in keys:
        dice_values = [r[k]["dice"] for r in runs if k in r]
        if not dice_values:
            continue
        cols[k] = dice_values
        sd = float(np.std(dice_values, ddof=1)) if len(dice_values) > 1 else float("nan")
        print(f"  {label:<26} n={len(dice_values)}  mean {np.mean(dice_values):.4f}  "
              f"SD(ddof=1) {sd:.4f}  min {min(dice_values):.4f}  max {max(dice_values):.4f}")

    if args.repeats > 1:
        print("\nWithin-seed spread (same seed, same partition, repeated):")
        for k, label in keys:
            for seed in args.seeds:
                dice_values = [r[k]["dice"] for r in runs
                     if k in r and r["model_seed"] == seed]
                if len(dice_values) < 2:
                    continue
                print(f"{label:<26} seed {seed:>5}: values {[round(x, 4) for x in dice_values]}  "
                      f"SD {np.std(dice_values, ddof=1):.4f}  range {max(dice_values) - min(dice_values):.4f}")

    if "self_labelling" in cols and "sparse_baseline" in cols:
        diff = np.array(cols["self_labelling"]) - np.array(cols["sparse_baseline"])
        print(f"\nPaired self-labelling minus sparse, n={len(diff)} per seed {[round(x, 4) for x in diff]}")
        print(f"mean {diff.mean():+.4f} SD(ddof=1) {np.std(diff, ddof=1):.4f}, {int((diff > 0).sum())}/{len(diff)} positive")
        try:
            from scipy import stats
            t_stat, p_value = stats.ttest_rel(cols["self_labelling"], cols["sparse_baseline"])
            print(f"paired t = {t_stat:.3f}, p = {p_value:.4f}")
        except Exception:
            pass

    if "loss_masked" in cols and "sparse_baseline" in cols:
        n = min(len(cols["loss_masked"]), len(cols["sparse_baseline"]))
        diff = np.array(cols["loss_masked"][:n]) - np.array(cols["sparse_baseline"][:n])
        print(f"\n Loss masking minus cropping, n={n} per seed {[round(x, 4) for x in diff]}")
        print(f"mean {diff.mean():+.4f} {int((diff > 0).sum())}/{n} in favour of masking")

    print(f"\nAll run files under {args.out_dir}")


if __name__ == "__main__":
    main()
