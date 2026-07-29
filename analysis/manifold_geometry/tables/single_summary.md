# Manifold geometry — single granularity (ambient = gauss)

`d/D` uses PR. Intrinsic dim by four estimators: PR and PCA-k% (linear, upper bounds), TwoNN and corr-dim (nonlinear, computed after temporal decimation so they do not just measure the trajectory curve; corr-dim is the mid-scale window).

**Volume deficit** is the headline Measurement-2 number. ε is chosen so a marginal-matched null cloud (each column permuted independently — same per-dim distribution, no cross-DoF structure) is hit with probability 0.01; `occ_real @ null-1%` is what fraction of the same ambient points land within that ε of the REAL data, and the deficit is log10 of the ratio. −1 dex = the real manifold occupies 10× less volume than a structureless cloud with identical marginals. `<1e-0X` / `<−X dex` are censored at the 1/M resolution limit, i.e. lower bounds, not measured values.

`tube slope` is the ε-occupancy log-log slope in the merged-tube regime (≈ codimension, a lower bound), `censored` where too few ambient points reach that regime. `tail slope` is the disjoint-ball diagnostic and should sit near D_eff — it is a health check on the estimator, not a result.

| embodiment | group | D | PR | d/D (PR) | pc90 | pc95 | pc99 | TwoNN | MLE | corr-dim | occ_real @ null-1% | volume deficit | tube slope | tail slope |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bridge | eef_position | 3 | 2.99 | 0.997 | 3 | 3 | 3 | 3.03±0.04 | 3.03 | 1.58 | 1.1e-02 | +0.03 dex | 0.9 (R²0.89) | 2.7 |
| bridge | eef_rotation | 3 | 3.00 | 0.999 | 3 | 3 | 3 | 3.06±0.04 | 3.05 | 1.80 | 1.0e-02 | +0.00 dex | 0.8 (R²0.90) | 2.8 |
| bridge | gripper | 1 | 1.00 | 1.000 | 1 | 1 | 1 | nan±nan | nan | nan | — | — | censored | — |
| bridge | eef_posrot | 6 | 5.51 | 0.918 | 5 | 6 | 6 | 5.75±0.05 | 5.46 | 2.13 | 9.7e-03 | -0.01 dex | 1.6 (R²0.89) | 4.9 |
| bridge | all_trained | 7 | 6.49 | 0.927 | 6 | 7 | 7 | 5.71±0.09 | 5.32 | 2.61 | 8.7e-03 | -0.06 dex | 2.4 (R²0.90) | 5.5 |
| robocasa_mg | base_motion | 4 | 0.00 | 0.000 | 0 | 0 | 0 | nan±nan | nan | nan | — | — | censored | — |
| robocasa_mg | control_mode | 1 | 0.00 | 0.000 | 0 | 0 | 0 | nan±nan | nan | nan | — | — | censored | — |
| robocasa_mg | eef_position | 3 | 2.89 | 0.962 | 3 | 3 | 3 | 2.88±0.03 | 2.87 | 1.82 | 1.0e-02 | +0.01 dex | 0.7 (R²0.87) | 3.0 |
| robocasa_mg | eef_rotation | 3 | 2.95 | 0.984 | 3 | 3 | 3 | 2.94±0.03 | 3.03 | 1.86 | 1.0e-02 | +0.02 dex | 0.8 (R²0.92) | 2.9 |
| robocasa_mg | gripper_close | 1 | 1.00 | 1.000 | 1 | 1 | 1 | nan±nan | nan | nan | — | — | censored | — |
| robocasa_mg | eef_posrot | 6 | 5.60 | 0.934 | 6 | 6 | 6 | 5.62±0.05 | 5.51 | 2.35 | 1.2e-02 | +0.06 dex | 1.6 (R²0.90) | 5.0 |
| robocasa_mg | all_trained | 12 | 6.56 | 0.547 | 6 | 7 | 7 | 5.50±0.06 | 5.36 | 2.69 | 1.1e-02 | +0.03 dex | 2.3 (R²0.91) | 5.4 |
| gr1_tabletop | arm | 14 | 7.99 | 0.571 | 9 | 11 | 13 | 5.19±0.05 | 4.69 | 3.51 | 9.5e-03 | -0.02 dex | 4.0 (R²0.87) | 9.6 |
| gr1_tabletop | hand | 12 | 2.30 | 0.192 | 2 | 3 | 3 | nan±nan | nan | nan | 9.0e-05 | -2.05 dex | censored | 8.5 |
| gr1_tabletop | waist | 3 | 2.96 | 0.987 | 3 | 3 | 3 | 3.00±0.04 | 2.95 | 1.10 | 8.9e-03 | -0.05 dex | 1.4 (R²0.98) | 2.4 |
| gr1_tabletop | all_trained | 29 | 8.28 | 0.286 | 12 | 15 | 18 | 5.06±0.04 | 4.35 | 2.80 | 6.1e-03 | -0.21 dex | 6.7 (R²0.87) | 15.3 |
| gr1_unified_1000 | arm | 14 | 7.97 | 0.569 | 9 | 11 | 13 | 5.34±0.04 | 4.73 | 3.53 | 8.9e-03 | -0.05 dex | 4.0 (R²0.87) | 9.6 |
| gr1_unified_1000 | hand | 12 | 2.30 | 0.192 | 2 | 3 | 3 | nan±nan | nan | nan | 9.0e-05 | -2.05 dex | censored | 8.5 |
| gr1_unified_1000 | waist | 3 | 2.93 | 0.978 | 3 | 3 | 3 | 3.01±0.03 | 2.94 | 1.11 | 8.0e-03 | -0.10 dex | 1.4 (R²0.98) | 2.5 |
| gr1_unified_1000 | all_trained | 29 | 8.29 | 0.286 | 12 | 15 | 18 | 5.14±0.04 | 4.38 | 2.80 | 6.2e-03 | -0.21 dex | 6.7 (R²0.87) | 15.1 |
| dexjoco_single | arm_pos | 3 | 2.69 | 0.896 | 3 | 3 | 3 | 2.81±0.00 | 2.79 | 2.04 | 9.8e-03 | -0.01 dex | 0.7 (R²0.84) | 2.9 |
| dexjoco_single | arm_rot | 3 | 2.74 | 0.914 | 3 | 3 | 3 | 2.94±0.00 | 2.85 | 1.53 | 9.4e-03 | -0.03 dex | 1.0 (R²0.98) | 2.7 |
| dexjoco_single | arm_posrot | 6 | 4.66 | 0.777 | 5 | 6 | 6 | 4.57±0.00 | 3.93 | 2.50 | 7.6e-03 | -0.12 dex | 2.0 (R²0.92) | 4.8 |
| dexjoco_single | hand | 16 | 3.44 | 0.215 | 7 | 9 | 13 | 3.72±0.00 | 4.03 | 2.78 | 5.0e-03 | -0.30 dex | 6.1 (R²0.95) | 10.4 |
| dexjoco_single | all_trained | 22 | 5.58 | 0.254 | 11 | 14 | 18 | 5.09±0.00 | 5.23 | 3.99 | 6.1e-03 | -0.21 dex | 6.1 (R²0.86) | 13.6 |
| dexjoco_dual | right_arm | 6 | 4.52 | 0.754 | 5 | 5 | 6 | 3.91±0.03 | 3.53 | 2.38 | 6.5e-03 | -0.19 dex | 2.2 (R²0.94) | 4.5 |
| dexjoco_dual | left_arm | 6 | 4.75 | 0.792 | 5 | 6 | 6 | 3.42±0.01 | 3.27 | 2.57 | 6.5e-03 | -0.19 dex | 2.4 (R²0.95) | 4.5 |
| dexjoco_dual | arm | 12 | 7.62 | 0.635 | 8 | 10 | 12 | 4.31±0.04 | 4.53 | 3.39 | 4.8e-03 | -0.32 dex | 3.9 (R²0.89) | 8.3 |
| dexjoco_dual | right_hand | 16 | 3.90 | 0.244 | 6 | 7 | 11 | 2.77±0.03 | 2.23 | 2.26 | 1.9e-03 | -0.72 dex | censored | 10.4 |
| dexjoco_dual | left_hand | 16 | 3.48 | 0.218 | 5 | 8 | 13 | 2.79±0.01 | 1.56 | 2.02 | 1.8e-03 | -0.74 dex | censored | 11.0 |
| dexjoco_dual | hand | 32 | 6.54 | 0.204 | 9 | 14 | 23 | 2.85±0.01 | 1.92 | 3.37 | 3.2e-03 | -0.49 dex | censored | 18.8 |
| dexjoco_dual | all_trained | 44 | 9.13 | 0.208 | 15 | 22 | 33 | 3.19±0.02 | 3.72 | 4.26 | 3.3e-03 | -0.48 dex | 10.9 (R²0.89) | 24.2 |
