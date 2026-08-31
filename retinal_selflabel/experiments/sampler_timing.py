# does the forbidden-region test actually make placement cheaper?

import argparse, json, os, time
import numpy as np

from retinal_selflabel.core.frangi_density import samp_in_ds

def naive_pairwise(dens, box, n_target, rng, max_tries=5000):
    hw, ww = dens.shape
    if hw < box or ww < box:
        return [], 0
    valid = dens[: hw - box + 1, : ww - box + 1]
    flat = valid.ravel().astype(np.float64)
    if flat.sum() <= 0:
        return [], 0
    csum = np.cumsum(flat)
    total = csum[-1]
    ncols = valid.shape[1]
    placed, tries = [], 0
    for _ in range(n_target):
        got = None
        for _ in range(max_tries):
            tries += 1
            idx = min(int(np.searchsorted(csum, rng.random() * total, side="right")), flat.size - 1)
            r, c = idx // ncols, idx % ncols
            ok = True
            for (pr, pc, ps) in placed:
                if not (r + box <= pr or pr + ps <= r or
                        c + box <= pc or pc + ps <= c):
                    ok = False
                    break
            if ok:
                got = (r, c)
                break
        if got is None:
            break
        placed.append((got[0], got[1], box))
    return placed, tries

# boxfilter over the forbidden mask 
def boxfilter_place(dens, box, n_target, rng):
    forb = np.zeros(dens.shape, np.float32)
    placed = []
    for _ in range(n_target):
        res = samp_in_ds(dens, forb, box, 1.0, dens.shape[0], dens.shape[1], box, rng)
        if res is None:
            break
        r, c = res
        forb[r: r + box, c: c + box] = 1.0
        placed.append((r, c, box))
    return placed, n_target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--grid", type=int, default=512)
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 9, 25, 100, 256, 441, 729])
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    CONFIGS = [(19, "HRF-like: 3504 px source, 128 px box"),
               (48, "CHASE_DB1-like: 999 px source, 128 px box"),
               (112, "DRIVE-like: 584 px source, 128 px box")]

    rows = []
    for box, label in CONFIGS:
        cap = (a.grid // box) ** 2
        print(f"{label}   grid {a.grid}x{a.grid}, box {box} px, non-overlapping capacity ~{cap}")
        print(f"{'N':>5}{'pairwise ms':>14}{'box filter ms':>15}, {'ratio':>9}{'pairwise placed':>17}{'bf placed':>12}")
        for N in a.budgets:
            tn, tb, pn, pb = [], [], [], []
            for rep in range(a.reps):
                rng = np.random.default_rng(rep)
                d = np.abs(rng.normal(size=(a.grid, a.grid))).astype(np.float32)
                d /= max(d.max(), 1e-9)

                rng = np.random.default_rng(rep)
                t0 = time.perf_counter()
                p, _ = naive_pairwise(d, box, N, rng)
                tn.append(time.perf_counter() - t0)
                pn.append(len(p))

                rng = np.random.default_rng(rep)
                t0 = time.perf_counter()
                p, _ = boxfilter_place(d, box, N, rng)
                tb.append(time.perf_counter() - t0)
                pb.append(len(p))

            ta, tbm = float(np.median(tn)), float(np.median(tb))
            row = dict(config=label, grid=a.grid, box=box, N=N, pairwise_s=ta, 
                       boxfilter_s=tbm, ratio_pairwise_over_boxfilter=ta / max(tbm, 1e-12),
                       pairwise_placed=float(np.median(pn)), boxfilter_placed=float(np.median(pb)))
            rows.append(row)
            print(f"{N:>5}{ta * 1e3:>14.2f}{tbm * 1e3:>15.2f}"
                  f"{row['ratio_pairwise_over_boxfilter']:>9.3f}"
                  f"{np.median(pn):>17.0f}{np.median(pb):>12.0f}")

    out = dict(rows=rows)
    p = os.path.join(a.out_dir, "sampler_timing.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwritten {p}")

if __name__ == "__main__":
    main()
