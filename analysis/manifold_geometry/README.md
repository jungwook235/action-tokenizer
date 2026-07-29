# Manifold geometry of raw actions — dimension ratio and volume occupancy

Self-contained analysis directory. Nothing outside `analysis/manifold_geometry/`
is read for code or written to; the M5 scripts in
`Isaac-GR00T/experiments/analysis/latent_vs_raw_dexterous/` are the conceptual
ancestor but are **not** imported or modified.

## What this measures

The claim under test: *as DoF grows, the action manifold occupies a vanishing
fraction of its ambient space, so an unconstrained policy has more room to leave
it.* Two complementary halves.

### Measurement 1 — dimension ratio `d/D` and codimension `D − d`

`D` = nominal ambient dim (group DoF, times the chunk length at chunk
granularity). `d` estimated by three estimators so the conclusion does not rest
on one threshold:

| estimator | family | reads |
|---|---|---|
| `PR = (Σλ)²/Σλ²` | linear, threshold-free | covariance eigenspectrum |
| `n_pc90/95/99` | linear, thresholded | same, sensitivity analysis over 90/95/99 |
| TwoNN (Facco 2017) | nonlinear | ratio of 1st/2nd NN distances |
| MLE (Levina–Bickel 2004) | nonlinear, lower variance | k-NN distances, k=10..20 |

### Measurement 2 — ε-occupancy (volume collapse)

For a `d`-dim manifold in `D`-dim ambient space the ε-tube volume scales as
`ε^(D−d)`, so codimension enters as an *exponent*. We sample `M` random ambient
points, measure each one's distance to the nearest real action, and read off

* `occ@k·r_med` — the fraction landing within `k ×` the data's own median NN
  distance. This is the headline "how much of the space does the manifold
  actually occupy" number.
* `codim_slope` — the log–log slope of the lower tail of the ambient→data NN
  distance CDF, i.e. `d log P / d log ε ≈ D − d`. Cross-validates Measurement 1
  by an independent route.

Two ambient references, both z-score matched to the data so only correlation
structure distinguishes them: isotropic Gaussian `N(0, I_D)` (default) and
uniform over the per-dim data range.

## Honest limits (carry these into any writeup)

* NN estimators over-estimate `d` under noise ("shadow dimension") and
  under-estimate it once `d > log n`. The robustness argument is that the linear
  and nonlinear families **agree in trend**, not that either number is exact.
* `codim_slope` equals the true codimension only for ε below the manifold's
  reach. In high `D` the smallest resolvable ambient→data distance usually sits
  above the reach, so the fitted slope is a **lower bound** on codimension
  (equivalently, an upper bound on `d`).
* `occ@k·r_med = 0` is a censored observation, not a measured zero: it means
  "below the `1/M` resolution limit". Charts mark these hollow with a down
  arrow, tables print `<1e-05`.
* Chunk granularity inflates codimension partly through *temporal* correlation
  (16 consecutive steps are highly correlated), which is a different mechanism
  from cross-DoF synergy. Read `single` for the DoF claim and `chunk` for what
  a chunk-level tokenizer actually faces.

## Data

Raw `action` parquet columns only, **all episodes, no train/val split** (the M5
convention; the original M5 runs subsampled 400 episodes, this one does not).
Group slices per embodiment are restated in `mg_embodiments.py` from each
data-config's `action_keys` × `meta/modality.json`.

Granularity:
* `single` — sample = one timestep, `D = |group|`
* `chunk` — sample = 16 non-overlapping consecutive steps, `D = |group| × 16`

## Files

```
mg_embodiments.py   dataset paths + DoF group slices, per embodiment
mg_metrics.py       PR / PCA-k%, TwoNN, Levina-Bickel MLE, eps-occupancy
mg_run.py           driver: load -> per group -> results/<embodiment>.json
mg_plots.py         charts + markdown/csv tables from results/
run_all.sh          full sweep
results/            <embodiment>.json
figs/               <granularity>_*.png
tables/             <granularity>_summary.{md,csv}
```

## Run

```bash
conda activate gr00t-actlat          # numpy / pandas / sklearn / matplotlib
cd analysis/manifold_geometry
bash run_all.sh                      # all 5 embodiments, both granularities
# or one at a time
python mg_run.py --embodiment gr1_tabletop --granularity single
python mg_plots.py
```

CPU only. Cost is dominated by the nearest-neighbour queries, which are capped
by `--nn-subsample` (10k points × 5 bootstraps) and `--n-ambient` (100k probe
points against a 20k-point index), not by the dataset size.
