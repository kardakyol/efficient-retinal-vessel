import argparse
import copy
import json
import os

import cv2
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from retinal_selflabel.core.datasets import (
    RetinalVesselDataset,
    SparseAnnotationSimulator,
    feature_discovery,
    transform_images,
    sample_splitting,
)
from retinal_selflabel.core.frangi_density import (
    density_cache,
    resolve_config,
)
from retinal_selflabel.core.full_image_sparse_dataset import (
    FullImageSparseDataset,
    MaskedBCEDiceLoss,
    get_full_image_sparse_transforms,
)
from retinal_selflabel.core.models import create_loss, create_model
from retinal_selflabel.core.random_box_sampler import BoxPlacement
from retinal_selflabel.core.random_box_sampler_v2 import NewRandomizer
from retinal_selflabel.core.resume import cache_key, cached_json, resume_dir_for
from retinal_selflabel.core.train import evaluate
from retinal_selflabel.core.utils import set_seed
from retinal_selflabel.selflabel.directed_growth import (
    DirectedExpansionManager,
    GrowthScheduler,
)
from retinal_selflabel.selflabel.intelligent_guidance import (
    ConfidenceFeedbackSelector,
    CorpusDensityHistogramSampler,
    make_frangi_provider,
    per_image_density_histogram_seeds,
    size_balanced_seeds,
)
from retinal_selflabel.selflabel.self_labelling import (
    IncrementalSelfLabeller,
    SpatialExpansionManager,
)

_SCHEDULES = {
    "fast": GrowthScheduler.fast,
    "medium": GrowthScheduler.medium,
    "slow": GrowthScheduler.slow,
    "accel": GrowthScheduler.accelerating,
}


# Three-way split (the leakage fix)
def dataset_splitting(samples, val_frac = 0.15, test_frac = 0.2, seed = 42):
    train_pool, test = sample_splitting(samples, test_frac=test_frac, seed=seed)
    rel_val = val_frac / max(1e-9, (1.0 - test_frac))
    rel_val = min(max(rel_val, 0.0), 0.9)
    train, val = sample_splitting(train_pool, test_frac=rel_val, seed=seed + 1)
    print(f"Three-way split: {len(train)} train, {len(val)} val, {len(test)} test")
    return train, val, test

# helpers
def image_shapes_for(samples):
    shapes = []
    for s in samples:
        img = cv2.imread(s["image_path"])
        if img is None:
            raise FileNotFoundError(s["image_path"])
        shapes.append(img.shape[:2])
    return shapes

def _empty_seeds(n):
    return [[] for _ in range(n)]

# for granulometry
def make_binary_provider(samples, cache, quantile = 0.80):
    def provider(i):
        s = samples[i]
        cfg = resolve_config(s["dataset"])
        d = cache.get_or_compute(s["image_path"], cfg)
        thr = float(np.quantile(d, quantile))
        return (d >= thr).astype(np.uint8)
    return provider


# Seeding strategies
def build_initial_seeds(strategy, train, image_shapes, cache, n_seeds, box_size, min_box, max_box, seed):
    n = len(train)

    if strategy == "random":
        sim = SparseAnnotationSimulator(train, patch_size=box_size, patches_per_image=n_seeds, min_vessel_fraction=0.01, seed=seed)
        seeds = _empty_seeds(n)
        for info in sim.patch_info:
            i = info["sample_idx"]
            seeds[i] = [(r, c, box_size) for (r, c) in info["patches"]]
        return seeds

    if strategy == "frangi":
        sampler = NewRandomizer(train, seed=seed, min_size=min_box, 
                                max_size=max_box, cache_dir=cache.cache_dir)
        placements = sampler.sample_boxes(total_boxes=n_seeds * n)
        grouped = NewRandomizer.group_by_sample(placements)
        seeds = _empty_seeds(n)
        for i, boxes in grouped.items():
            seeds[i] = [(b.row, b.col, b.size) for b in boxes]
        return seeds

    if strategy == "density_hist":
        provider = make_frangi_provider(train, cache, resolve_config)
        sampler = CorpusDensityHistogramSampler(n_seeds=n_seeds * n, box_size=box_size, seed=seed)
        return sampler.plan(n, provider, image_shapes)

    if strategy == "density_per_image":
        provider = make_frangi_provider(train, cache, resolve_config)
        return per_image_density_histogram_seeds(n, provider, box_size=box_size, seed=seed)

    if strategy == "granulometry":
        binprov = make_binary_provider(train, cache)
        return size_balanced_seeds(n, binprov, image_shapes, n_seeds=n_seeds * n, box_size=box_size, seed=seed)
    raise ValueError(f"Unknown seeding strategy: {strategy}")


def add_confidence_seeds(seeds, model, train, cache, device, img_size, k_per_image, box_size):
    
    selector = ConfidenceFeedbackSelector(box_size=box_size)
    model.eval()
    out = [list(s) for s in seeds]
    with torch.no_grad():
        for i, s in enumerate(train):
            img = cv2.imread(s["image_path"])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ho, wo = img.shape[:2]
            img_r = cv2.resize(img, (img_size, img_size))
            t = torch.from_numpy(img_r.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(t)).squeeze().cpu().numpy()
            prob = cv2.resize(prob, (wo, ho), interpolation=cv2.INTER_LINEAR)

            cfg = resolve_config(s["dataset"])
            prior = cache.get_or_compute(s["image_path"], cfg)

            forbidden = np.zeros((ho, wo), dtype=np.uint8)
            for (r, c, sz) in out[i]:
                forbidden[r:r + sz, c:c + sz] = 1

            new = selector.select_seeds_for_image(
                prob, k_per_image, prior_density=prior, forbidden=forbidden
)
            out[i].extend(new)
    return out

def seeds_to_placements(seeds, train, image_shapes):
    out = {}
    for i, boxes in enumerate(seeds):
        if not boxes:
            continue
        h, w = image_shapes[i]
        s = train[i]
        out[i] = [BoxPlacement( sample_idx=i, dataset=s["dataset"], image_id=s["id"],
                image_h=h, image_w=w, row=r, col=c, size=sz) for (r, c, sz) in boxes]
    return out


# Base model training with masked loss
def train_base_model(train, placements, val, device, img_size, num_epochs, lr, patience, bs = 4, init_model = None):
    if init_model is None:
        model = create_model(architecture="unet", encoder="resnet34",
            encoder_weights="imagenet", in_channels=3, classes=1).to(device)
    else:
        model = init_model.to(device)

    ds = FullImageSparseDataset(train, placements, img_size=img_size, transform=get_full_image_sparse_transforms(img_size, train=True), include_uncovered=True)
    loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0)

    vds = RetinalVesselDataset(val, transform=transform_images("validation", img_size))
    vloader = DataLoader(vds, batch_size=4, shuffle=False, num_workers=0)

    crit = MaskedBCEDiceLoss()
    opt = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=num_epochs, eta_min=1e-6)

    best_dice = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    no_improve = 0

    for ep in range(1, num_epochs + 1):
        model.train()
        for img, gt, v in loader:
            img, gt, v = img.to(device), gt.to(device), v.to(device)
            opt.zero_grad()
            logits = model(img)
            loss = crit(logits, gt, v)
            loss.backward()
            opt.step()
        sched.step()

        vm = evaluate(model, vloader, create_loss("bce_dice"), device)
        if vm["dice"] > best_dice:
            best_dice = vm["dice"]
            best_wts = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_wts)
    print(f"Base model best VAL Dice: {best_dice:.4f}")
    return model, best_dice


# Manager factory
def build_manager(growth, image_shapes, seeds, guidance_maps, schedule,
                  expand_px, keep_fraction):
    
    if growth == "uniform":
        return SpatialExpansionManager(image_shapes=image_shapes, 
                                       initial_patches=seeds, expand_px=expand_px)
    if growth == "directed":
        if guidance_maps is None:
            raise ValueError("directed growth requires guidance_maps")
        return DirectedExpansionManager(image_shapes=image_shapes, initial_patches=seeds,
                                        guidance_maps=guidance_maps, scheduler=_SCHEDULES[schedule](),
                                        keep_fraction=keep_fraction, expand_px=expand_px)
    raise ValueError(f"Unknown growth mode: {growth}")

# One full run
def run_one(args, train, val, test, cache, device, seeding, growth):
    print(f"seeding={seeding}  growth={growth}  schedule={args.schedule}")

    shapes = image_shapes_for(train)
    seeds = build_initial_seeds(seeding, train, shapes, cache, n_seeds=args.n_seeds,
                                 box_size=args.box_size, min_box=args.min_box, max_box=args.max_box, seed=args.seed)
    placements = seeds_to_placements(seeds, train, shapes)

    # Base model selected on validation
    base_model, base_val = train_base_model(train, placements, val, device, img_size=args.img_size, num_epochs=args.base_epochs, lr=args.lr, patience=args.base_patience)

    if args.reseed_confidence > 0:
        seeds = add_confidence_seeds(seeds, base_model, train, cache, device, args.img_size, args.reseed_confidence, args.box_size)

    # Guidance maps for directed growth
    guidance_maps = None
    if growth == "directed":
        prov = make_frangi_provider(train, cache, resolve_config)
        guidance_maps = [prov(i) for i in range(len(train))]

    manager = build_manager(growth, shapes, seeds, guidance_maps, schedule=args.schedule, 
                            expand_px=args.expand_px, keep_fraction=args.keep_fraction)

    # Self-labelling
    labeller = IncrementalSelfLabeller(model=copy.deepcopy(base_model), train=train,
                                       val=val, expansion_manager=manager, device=device,
                                       img_size=args.img_size, confidence_threshold=args.confidence_threshold, 
                                       finetune_epochs=args.finetune_epochs, finetune_lr=5e-4,
                                       pseudo_weight=args.pseudo_weight, max_iterations=args.max_iterations,
                                       patience=args.selflabel_patience, ckpt_dir=os.path.join(args.out_dir, "ckpt", f"{seeding}_{growth}"),)
    best_model, it_log = labeller.run()

    tds = RetinalVesselDataset(test, transform=transform_images("validation", args.img_size))
    tloader = DataLoader(tds, batch_size=4, shuffle=False, num_workers=0)
    test_metrics = evaluate(best_model, tloader, create_loss("bce_dice"), device)

    print(f"Best Validation Dice: {labeller.best_dice:.4f} | "
          f"Test Dice: {test_metrics['dice']:.4f}")

    return {"seeding": seeding, "growth": growth, "schedule": args.schedule if growth == "directed" else None,
            "n_seeds_per_image": args.n_seeds, "box_size": args.box_size, "seed": args.seed,
            "base_val_dice": float(base_val), "best_val_dice": float(labeller.best_dice),
            "test_metrics": {k: float(v) for k, v in test_metrics.items()}, "final_coverage": float(manager.get_coverage()),
            "iteration_log_val": [{"iteration": e["iteration"], "coverage": e["coverage"],"val_dice": e["val_dice"]} for e in it_log]}


# Iterative confidence-feedback loop
def _confidence_new_boxes(model, train, seeds, priors, selector, args, device):
    model.eval()
    with torch.no_grad():
        for i, s in enumerate(train):
            img = cv2.cvtColor(cv2.imread(s["image_path"]), cv2.COLOR_BGR2RGB)
            ho, wo = img.shape[:2]
            t = torch.from_numpy(cv2.resize(img, (args.img_size, args.img_size)).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(t)).squeeze().cpu().numpy()
            prob = cv2.resize(prob, (wo, ho), interpolation=cv2.INTER_LINEAR)
            forbidden = np.zeros((ho, wo), dtype=np.uint8)
            for (r, c, sz) in seeds[i]:
                forbidden[r:r + sz, c:c + sz] = 1
            new = selector.select_seeds_for_image(
                prob, args.feedback_k, prior_density=priors[i], forbidden=forbidden)
            seeds[i].extend(new)


def _random_new_boxes(train, seeds, shapes, k, box_size, rng):
    for i in range(len(train)):
        ho, wo = shapes[i]
        if ho <= box_size or wo <= box_size:
            continue
        added = 0
        for _ in range(200):
            if added >= k:
                break
            r = int(rng.integers(0, ho - box_size + 1))
            c = int(rng.integers(0, wo - box_size + 1))
            if all(abs(r - pr) >= box_size or abs(c - pc) >= box_size
                   for (pr, pc, _sz) in seeds[i]):
                seeds[i].append((r, c, box_size))
                added += 1


def run_confidence_loop(args, train, val, test, cache, device):
    print(f"Confidence Feedback Loop x Random Control rounds={args.feedback_rounds} k/round/image={args.feedback_k}")

    shapes = image_shapes_for(train)
    init_seeds = build_initial_seeds("random", train, shapes, cache, n_seeds=1, 
                                     box_size=args.box_size, min_box=args.min_box, max_box=args.max_box, seed=args.seed)

    prov = make_frangi_provider(train, cache, resolve_config)
    priors = [prov(i) for i in range(len(train))]
    selector = ConfidenceFeedbackSelector(box_size=args.box_size)

    # Shared base model
    placements = seeds_to_placements(init_seeds, train, shapes)
    base_model, base_val = train_base_model(train, placements, val, device, args.img_size, args.base_epochs, args.lr, args.base_patience)
    print(f"Shared base | boxes={sum(len(s) for s in init_seeds):4d} | Validation Dice {base_val:.4f}")

    tds = RetinalVesselDataset(test, transform=transform_images("validation", args.img_size))
    tloader = DataLoader(tds, batch_size=4, shuffle=False, num_workers=0)
    criterion = create_loss("bce_dice")

    arms = {}
    for mode in ("confidence", "random"):
        seeds = [list(s) for s in init_seeds]
        model = copy.deepcopy(base_model)
        best_model = copy.deepcopy(base_model)
        best_val = base_val
        rng = np.random.default_rng(args.seed + 1)
        log = [{"round": 0, "boxes": sum(len(s) for s in seeds), "val_dice": float(base_val)}]
        print(f"\n{mode} arm]")
        for rd in range(1, args.feedback_rounds + 1):
            if mode == "confidence":
                _confidence_new_boxes(model, train, seeds, priors, selector, args, device)
            else:
                _random_new_boxes(train, seeds, shapes, args.feedback_k, args.box_size, rng)
            placements = seeds_to_placements(seeds, train, shapes)
            model, val_dice = train_base_model(train, placements, val, device, 
                                               args.img_size, args.feedback_epochs, args.lr,
                                               args.base_patience, init_model=copy.deepcopy(best_model))
            flag = ""
            if val_dice > best_val:
                best_val = val_dice
                best_model = copy.deepcopy(model)
                flag = "Best"
            log.append({"round": rd, "boxes": sum(len(s) for s in seeds), "val_dice": float(val_dice)})
            print(f"Round {rd} | boxes={log[-1]['boxes']:4d} | "
                  f"Validation Dice {val_dice:.4f} | {flag}")
        test_metrics = evaluate(best_model, tloader, criterion, device)
        arms[mode] = {"best_val_dice": float(best_val), "test_metrics": {k: float(v) for k, v in test_metrics.items()}, "rounds_log_val": log}
        print(f" {mode}: Best Validation {best_val:.4f} | "
              f"Test Dice {test_metrics['dice']:.4f}")

    delta = arms["confidence"]["test_metrics"]["dice"] - \
        arms["random"]["test_metrics"]["dice"]
    print("Confidence vs Random on matched budget")
    print(f" {'round':<8}{'confidence VAL':<18}{'random VAL':<18}")
    for cr, rr in zip(arms["confidence"]["rounds_log_val"], 
                      arms["random"]["rounds_log_val"]):
        print(f"{cr['round']:<8}{cr['val_dice']:<18.4f}{rr['val_dice']:<18.4f}")
    print(f"\n Testconfidence {arms['confidence']['test_metrics']['dice']:.4f} |  random {arms['random']['test_metrics']['dice']:.4f} | delta (conf - rand) {delta:+.4f}")

    return {
        "mode": "confidence_loop",
        "feedback_rounds": args.feedback_rounds,
        "feedback_k": args.feedback_k,
        "base_val_dice": float(base_val),
        "confidence": arms["confidence"],
        "random_control": arms["random"],
        "test_delta_conf_minus_rand": float(delta),
    }


# Speed-accuracy frontier

def run_frontier(args, train, val, test, cache, device, resume_dir=None):
    print(f"Speed-Accuracy Frontier seeding={args.seeding}")
    rows = []
    for sched in list(_SCHEDULES.keys()):
        args.schedule = sched

        def _compute(sched=sched):
            args.schedule = sched
            return run_one(args, train, val, test, cache, device, seeding=args.seeding, growth="directed")

        if resume_dir is not None:
            r, hit = cached_json(
                resume_dir,
                cache_key("frontier", args.seed, args.seeding, sched),
                _compute)
            if hit:
                print(f"frontier seed={args.seed} schedule={sched} (Test {r['test_metrics']['dice']:.4f})")
        else:
            r = _compute()
        rows.append(r)

    print("Frontier (measured)")
    print(f"{'schedule':<10}{'coverage%':<12}{'best VAL':<12}{'TEST dice':<12}")
    for r in rows:
        print(f"{r['schedule']:<10}{r['final_coverage']*100:<12.2f} {r['best_val_dice']:<12.4f}{r['test_metrics']['dice']:<12.4f}")
    return {"mode": "frontier", "results": rows}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./outputs_phase2")
    p.add_argument("--seeding", type=str, default="density_hist", choices=["random", "frangi", "density_hist", "density_per_image", "granulometry"])
    p.add_argument("--growth", type=str, default="uniform", choices=["uniform", "directed"])
    p.add_argument("--schedule", type=str, default="medium",choices=list(_SCHEDULES.keys()))
    p.add_argument("--compare", action="store_true")
    p.add_argument("--frontier", action="store_true")
    p.add_argument("--confidence_loop", action="store_true")
    p.add_argument("--feedback_rounds", type=int, default=5)
    p.add_argument("--feedback_k", type=int, default=1)
    p.add_argument("--feedback_epochs", type=int, default=40)
    p.add_argument("--n_seeds", type=int, default=1)
    p.add_argument("--box_size", type=int, default=128)
    p.add_argument("--min_box", type=int, default=64)
    p.add_argument("--max_box", type=int, default=192)
    p.add_argument("--reseed_confidence", type=int, default=0)
    p.add_argument("--expand_px", type=int, default=16)
    p.add_argument("--keep_fraction", type=float, default=0.5)
    p.add_argument("--confidence_threshold", type=float, default=0.7)
    p.add_argument("--pseudo_weight", type=float, default=0.3)
    p.add_argument("--finetune_epochs", type=int, default=15)
    p.add_argument("--max_iterations", type=int, default=25)
    p.add_argument("--selflabel_patience", type=int, default=8)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--base_epochs", type=int, default=120)
    p.add_argument("--base_patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--no_cuda", action="store_true")
    return p.parse_args()


def _set_deterministic():
    import random as _random
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    _random.seed(0)


def _mean_std(vals):
    n = len(vals)
    m = sum(vals) / n
    s = (sum((v - m) ** 2 for v in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return {"mean": float(m), "std": float(s), "n": n, "values": [float(v) for v in vals]}

def _aggregate(mode, per_seed):
    summary = {"mode": mode, "seeds_run": len(per_seed)}
    if mode == "compare":
        by = {}
        for out in per_seed:
            for r in out["results"]:
                by.setdefault(r["seeding"], []).append(r["test_metrics"]["dice"])
        summary["by_seeding_test_dice"] = {k: _mean_std(v) for k, v in by.items()}
    elif mode == "frontier":
        by = {}
        for out in per_seed:
            for r in out["results"]:
                by.setdefault(r["schedule"], []).append(r["test_metrics"]["dice"])
        summary["by_schedule_test_dice"] = {k: _mean_std(v) for k, v in by.items()}
    elif mode == "confidence_loop":
        summary["confidence_test_dice"] = _mean_std([o["confidence"]["test_metrics"]["dice"] for o in per_seed])
        summary["random_control_test_dice"] = _mean_std([o["random_control"]["test_metrics"]["dice"] for o in per_seed])
        summary["delta_conf_minus_rand"] = _mean_std([o["test_delta_conf_minus_rand"] for o in per_seed])
    else:  # single
        summary["test_dice"] = _mean_std([o["results"][0]["test_metrics"]["dice"] for o in per_seed])
    return summary


def _print_summary(summary):
    print(f"Aggregate over {summary['seeds_run']} seeds (mean+-std)")
    if summary["mode"] == "compare":
        print(f"  {'seeding':<18}{'Test dice (mean +- std)':<28}")
        for k, v in summary["by_seeding_test_dice"].items():
            print(f"  {k:<18}{v['mean']:.4f} +- {v['std']:.4f}")
    elif summary["mode"] == "frontier":
        print(f"  {'schedule':<12}{'Test dice (mean +- std)':<28}")
        for k, v in summary["by_schedule_test_dice"].items():
            print(f" {k:<12}{v['mean']:.4f} +- {v['std']:.4f}")
    elif summary["mode"] == "confidence_loop":
        c = summary["confidence_test_dice"]; r = summary["random_control_test_dice"]
        d = summary["delta_conf_minus_rand"]
        print(f" confidence test {c['mean']:.4f} +- {c['std']:.4f}")
        print(f" random control {r['mean']:.4f} +- {r['std']:.4f}")
        print(f" delta (conf-rand){d['mean']:+.4f} +- {d['std']:.4f}")
    else:
        v = summary["test_dice"]
        print(f"Test dice {v['mean']:.4f} +- {v['std']:.4f}")


def _dispatch(args, train, val, test, cache, device, resume_dir=None):
    sd = args.seed
    if args.frontier:
        return run_frontier(args, train, val, test, cache, device, resume_dir=resume_dir)
    if args.confidence_loop:
        def _compute():
            return run_confidence_loop(args, train, val, test, cache, device)
        if resume_dir is not None:
            out, hit = cached_json(resume_dir, cache_key("confidence", sd),
                                   _compute)
            if hit:
                print(f"confidence seed={sd} (delta {out.get('test_delta_conf_minus_rand', 0):+.4f})")
            return out
        return _compute()
    if args.compare:
        strategies = ["random", "frangi", "density_hist", "density_per_image", "granulometry"]
        results = []
        for strat in strategies:
            def _compute(strat=strat):
                return run_one(args, train, val, test, cache, device, seeding=strat, growth=args.growth)
            if resume_dir is not None:
                r, hit = cached_json(resume_dir, cache_key("compare", sd, strat, args.growth, args.n_seeds), _compute)
                if hit:
                    print(f"Compare seed={sd} seeding={strat} "
                          f"(Test {r['test_metrics']['dice']:.4f})")
            else:
                r = _compute()
            results.append(r)
        print(f"Matched-Budget Comparison (n_seeds/image={args.n_seeds}, "
              f"growth={args.growth}, seed={sd})")
        print(f"  {'seeding':<18}{'base VAL':<12}{'best VAL':<12} {'Test dice':<12}{'coverage':<10}")
        for r in results:
            print(f"{r['seeding']:<18}{r['base_val_dice']:<12.4f}"
                  f"{r['best_val_dice']:<12.4f}"
                  f"{r['test_metrics']['dice']:<12.4f}"
                  f"{r['final_coverage']*100:<10.2f}")
        return {"mode": "compare", "config": vars(args), "results": results}

    def _compute():
        return run_one(args, train, val, test, cache, device, seeding=args.seeding, growth=args.growth)
    if resume_dir is not None:
        r, _hit = cached_json(resume_dir, cache_key("single", sd, args.seeding, args.growth, args.schedule), _compute)
    else:
        r = _compute()
    return {"mode": "single", "config": vars(args), "results": [r]}


def main():
    args = parse_args()
    if args.deterministic:
        _set_deterministic()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cpu")
    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"Info Device: {device}")

    samples = feature_discovery(args.data_dir)
    cache = density_cache(os.path.join(args.out_dir, "cache", "frangi"))
    resume_dir = resume_dir_for(args.out_dir)
    seed_list = args.seeds if args.seeds else [args.seed]

    per_seed = []
    for sd in seed_list:
        if len(seed_list) > 1:
            print(f"Seed {sd}  ({seed_list.index(sd)+1}/{len(seed_list)})")
        args.seed = sd
        set_seed(sd)
        train, val, test = dataset_splitting(
            samples, val_frac=args.val_frac,
            test_frac=args.test_frac, seed=sd)
        out = _dispatch(args, train, val, test,
                        cache, device, resume_dir=resume_dir)
        per_seed.append(out)
        if len(seed_list) > 1:
            sdir = os.path.join(args.out_dir, f"seed_{sd}")
            os.makedirs(sdir, exist_ok=True)
            with open(os.path.join(sdir, "phase2_results.json"), "w") as f:
                json.dump(out, f, indent=2)

    if len(seed_list) > 1:
        summary = _aggregate(per_seed[0]["mode"], per_seed)
        _print_summary(summary)
        with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
            json.dump({"summary": summary, "per_seed": per_seed}, f, indent=2)
        print(f"\nSaved per-seed results and summary to {args.out_dir}")
    else:
        with open(os.path.join(args.out_dir, "phase2_results.json"), "w") as f:
            json.dump(per_seed[0], f, indent=2)
        print(f"\n Saved results to "
              f"{os.path.join(args.out_dir, 'phase2_results.json')}")


if __name__ == "__main__":
    main()
