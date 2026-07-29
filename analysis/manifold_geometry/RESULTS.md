# Manifold geometry of raw actions — results (2026-07-27)

All episodes, no train/val split, per-timestep (`single`) and 16-step
non-overlapping chunk (`chunk`) granularity. Full numbers in
`tables/{single,chunk}_summary.{md,csv}`, raw JSON in `results/`, charts in
`figs/`. PR values reproduce the original M5 run to 2 decimals where both used
the full pool (dexjoco single/dual), and match it within ±0.02 where M5 used the
400-episode subsample (gr1/robocasa/bridge) — the subsample was already
converged.

## Finding 0 (data audit, changes how M5 must be read)

**The GR-1 "12-DoF dexterous hand" action is a binary open/close command, not a
continuous hand.** Across the whole dataset the 12-dim hand vector visits only
**4–6 distinct states** (each column takes 2–3 values, `results/discreteness.json`).
DexJoCo hands are genuinely continuous (~30k distinct values per column).

Consequences:
- GR-1 hand redundancy 5.22× (PR 2.30/12) is **not** evidence of postural
  synergies on a low-dim continuous manifold — there is no manifold, just a few
  points. NN estimators correctly return NaN there (all NN distances are 0).
- The M5 claim "GR1 reproduces the DexJoCo hand-synergy finding" is void; the
  only valid continuous-hand evidence is DexJoCo (hand 16 → PR 3.44–3.90).
- Same caveat applies to robocasa `base_motion`/`control_mode` (constant here)
  and all 1-dim grippers.

## Measurement 1 — dimension ratio d/D and codimension

`single` granularity, `all_trained` group (full trained action):

| embodiment | D | PR | d/D | pc95 | TwoNN | MLE | corr-dim |
|---|---|---|---|---|---|---|---|
| bridge | 7 | 6.49 | 0.93 | 7 | 5.71 | 5.32 | 2.61 |
| robocasa_mg | 12 | 6.56 | 0.55* | 7 | 5.50 | 5.36 | 2.69 |
| gr1_tabletop | 29 | 8.28 | 0.29 | 15 | 5.06 | 4.35 | 2.80 |
| gr1_unified_1000 | 29 | 8.29 | 0.29 | 15 | 5.14 | 4.38 | 2.80 |
| dexjoco_single | 22 | 5.58 | 0.25 | 14 | 5.09 | 5.23 | 3.99 |
| dexjoco_dual | 44 | 9.13 | 0.21 | 22 | 3.19 | 3.72 | 4.26 |

(*robocasa d/D inflated downward by 5 constant dims; on its live 7 dims d/D≈0.94.)

- **d/D falls monotonically with DoF** by every estimator: linear d/D goes
  0.93 → 0.21 from bridge to dexjoco_dual; nonlinear (TwoNN/MLE/corr-dim)
  estimates stay flat at ~3–6 while D grows 7 → 44, i.e. the *codimension grows
  roughly like D itself*. Linear and nonlinear families agree in trend
  (nonlinear sits below PR, as expected for curved manifolds), which is the
  robustness argument; neither number alone is exact.
- pc90/95/99 sensitivity: ordering identical at all three thresholds.
- Chunk granularity amplifies this: gr1 464 → PR 9.3 (d/D=0.020),
  dexjoco_dual 704 → PR 9.3 (d/D=0.013). Adding 16 steps of time adds ~1
  effective dim (temporal correlation, cf. S3), while nominal D multiplies 16×.
- Where the hand is continuous (DexJoCo), the hand group is exactly the locus of
  low d/D (0.20–0.24 vs arm 0.63–0.79) — H1 survives on DexJoCo only.

## Measurement 2 — ε-occupancy / volume deficit (null-calibrated)

ε is set so a *marginal-matched null* (each column independently permuted: same
per-dim distribution, no cross-DoF structure) is hit by ambient Gaussian queries
with p=1%; we report the real cloud's occupancy at that same ε.
`volume_deficit_log10 = log10(occ_real / 0.01)`; negative = real actions occupy
less volume than the structureless null.

`single`, selected:

| group | deficit |
|---|---|
| bridge / robocasa eef groups | ≈ 0.00 dex (full-rank control: no deficit — correct control result) |
| gr1 arm | −0.02 … −0.05 dex |
| gr1 all_trained | −0.21 dex |
| dexjoco_single hand | −0.30 dex |
| dexjoco_dual arm | −0.32 dex |
| dexjoco_dual hand (16d each) | −0.72 / −0.74 dex |
| dexjoco_dual all (44) | −0.48 dex |
| gr1 hand (discrete!) | −2.05 dex (an artifact of discreteness, not thinness) |

- Deficit deepens with DoF/dexterity exactly where Measurement 1 says the
  codimension grows: dexjoco_dual 16-d hands occupy ~5× less volume than their
  own null; low-DoF EEF control occupies the same volume as its null.
- The ε-tube slope, where measurable, tracks D − PR (e.g. dexjoco_dual all_44:
  tube 10.9 vs D−PR = 34.9 → slope is a *lower bound*, as documented), and the
  disjoint-ball `tail_slope` lands near D_eff everywhere — the estimator's
  health check passes.

**Chunk-granularity caveat (important).** At chunk D (112–704) the deficit
*inverts* (+1.3 … +2.0 dex): a low-dim but wide-spread subspace is Euclidean-
closer to random ambient queries than an isotropic null cloud of the same size,
so at these D the statistic measures subspace proximity, not tube volume. Use
Measurement 2 at `single` granularity only; at chunk granularity the meaningful
statements are the PR/pc95 collapse and the corr-dim staying O(3–5) while D
grows to 700.

## Scale-dependence (honesty note)

Robot actions are not scale-free: the local-slope curves
(`figs/*_corrdim_scale.png`) show intrinsic dim rising from ~1–2 at small radius
(trajectory curve) toward the manifold value at mid radius. All NN-based numbers
here use temporal decimation (stride 25) and the mid-scale window; without that,
TwoNN measures the trajectory (~1.5) rather than the manifold — a mistake the
earlier literature numbers may contain.

## Files

- `tables/single_summary.md` / `chunk_summary.md` — full per-group tables
- `figs/single_dimratio_vs_dof.png` — Measurement 1 headline
- `figs/single_occupancy_vs_dof.png` — Measurement 2 headline (null-calibrated)
- `figs/single_occupancy_curves.png` — real vs null CDFs
- `figs/single_corrdim_scale.png` — scale-dependence diagnostic
- `figs/single_estimator_agreement.png` — PR vs TwoNN robustness
- `results/discreteness.json` — Finding 0 audit
