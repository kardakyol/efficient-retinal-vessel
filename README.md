# Annotation-Efficient Retinal Vessel Segmentation

## Dataset

https://www.kaggle.com/datasets/ipythonx/retinal-vessel-segmentation

## Repository layout

```
core/          shared model, data loading, training loop
experiments/   experiments
selflabel/     incremental self-labelling loop
figures/       plotting scripts that read JSON from results/ and emit base diagrams
results/       JSON produced by the experiment scripts
```

---

## Script -> output -> dissertation artefact

| Script | Output in results/ | Dissertation artefact |
|---|---|---|
| headline_fixed_split.py | nondet_split42_seed*.json, repeat_*.json | Tables 2, 7, 8 · §5 |
| experiments_random_sweep_v2.py | sweep_results_v2.json | Figure 1 · §4.2 |
| ablations_factorial_v2.py | factorial_results.json | Table 4 (upper) · Figures 18, 19 |
| comparative_baseline.py | comparative_results.json | Table 5 · §4.4 |
| phase2_guidance.py --compare | p2_compare_summary.json | Table 4 (lower) · Figure 20 |
| phase2_guidance.py --confidence_loop | p2_feedback_summary.json | Figures 24, 25 |
| phase2_guidance.py --frontier | p2_frontier_summary.json | Figure 26 · A.6 |
| sampler_timing.py | sampler_timing.json | Figure 16 · A.3 |
| frangi_sigma_robustness_v2.py | sigma_robustness_results.json | Figure 17 |
| coverage_matched_v2.py | coverage_matched_results.json | A.3 |
| experiments_incremental_annotation_v2.py | incremental_v2_results.json | Figure 21 · A.4 |
| ablations.py | seed*_ablation/ablation_{patch_size,pretrained,threshold}.json | A.2 · §3.4 |
| experiments_extended.py | lambda_ablation.json, entropy_error.json | A.6 · §4.6 |

---

## Running the experiments

| Result in results/ | Command |
|---|---|
| nondet_split42_seed*.json | python -m retinal_selflabel.experiments.headline_fixed_split --out_dir outputs --arms full sparse selflabel lossmask |
| repeat_nondet_split42_seed42_r*.json | python -m retinal_selflabel.experiments.headline_fixed_split --out_dir outputs --arms full sparse --seeds 42 --repeats 3 --tag repeat_nondet |
| repeat_det_split42_seed42_r*.json | as above, plus --deterministic --tag repeat_det |
| sweep_results_v2.json | python -m retinal_selflabel.experiments.experiments_random_sweep_v2 --seeds 42 123 7 |
| factorial_results.json | python -m retinal_selflabel.experiments.ablations_factorial_v2 |
| comparative_results.json | python -m retinal_selflabel.experiments.comparative_baseline |
| p2_compare_summary.json | python -m retinal_selflabel.experiments.phase2_guidance --compare |
| p2_feedback_summary.json | python -m retinal_selflabel.experiments.phase2_guidance --confidence_loop |
| p2_frontier_summary.json | python -m retinal_selflabel.experiments.phase2_guidance --frontier |
| sampler_timing.json | python -m retinal_selflabel.experiments.sampler_timing |
| sigma_robustness_results.json | python -m retinal_selflabel.experiments.frangi_sigma_robustness_v2 |
| coverage_matched_results.json | python -m retinal_selflabel.experiments.coverage_matched_v2 |
| incremental_v2_results.json | python -m retinal_selflabel.experiments.experiments_incremental_annotation_v2 |
| lambda_ablation.json, entropy_error.json | python -m retinal_selflabel.experiments.experiments_extended |
| seed*_ablation/ablation_*.json | no CLI - ablations.py is a module of functions, called directly from a driver |

Every script defaults to the seeds that were actually used, with one exception: experiments_random_sweep_v2 defaults to --seeds 42, while the reported sweep used 42 123 7. Pass them explicitly, as above.

Scripts write under --out_dir (./outputs_new for most, ./outputs for experiments_extended and run_selflabel, ./outputs_phase2 for phase2_guidance). They do not write into results/; the JSON committed there is the copy the dissertation refers to.

---

## Notes for anyone re-running this code

### Figures are hand-finished

The scripts in figures/ generate the **base** diagram from the JSON in results/. The figures printed in the dissertation are those diagrams after manual editing in SVG (labels, spacing, styling). Re-running a figure script reproduces the underlying numbers and the general layout, but not the exact image that appears in the text.

### Phase 2 uses a different split policy

phase2_guidance.py draws a **new partition for every seed**. Every other script uses the **fixed** partition with split_seed=42. This is deliberate and is stated in §3.1 of the dissertation. It matters when comparing Phase 2 numbers directly against fixed-split results.

### Code that runs but is not used

- **experiments_extended.py** contains five experiments. Only lambda_ablation() and entropy_error() feed the dissertation (A.6 · §4.6). The other three execute and write output, but nothing in the text depends on them.
- **run_selflabel.py**: main() is not used. headline_fixed_split.py imports run_self_labelling() from this module directly, and that is the entry point for the reported self-labelling runs.