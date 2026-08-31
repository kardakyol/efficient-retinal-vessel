import argparse

import retinal_selflabel.figures.fig1_full_supervision_v3 as f1
import retinal_selflabel.figures.fig2a_randomizer_v3 as f2a
import retinal_selflabel.figures.fig2b_sparse_training_v3 as f2b
import retinal_selflabel.figures.fig3_nonoverlap_v3 as f3
import retinal_selflabel.figures.fig4_frangi_density_v3 as f4

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="./figures_redraw")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--results", default=None)
    a = ap.parse_args()
    print("Fig 1 - "); f1.main(a.data, a.out, a.ckpt)
    print("Fig 2a - "); f2a.main(a.data, a.out)
    print("Fig 2b - "); f2b.main(a.data, a.out, a.ckpt)
    print("Fig 3 - "); f3.main(a.data, a.out)
    print("Fig 4 - "); f4.main(a.data, a.out)
    print("Done", a.out)
