# Manifold geometry — chunk granularity (ambient = gauss)

`d/D` uses PR. Intrinsic dim by four estimators: PR and PCA-k% (linear, upper bounds), TwoNN and corr-dim (nonlinear, computed after temporal decimation so they do not just measure the trajectory curve; corr-dim is the mid-scale window).

**Volume deficit** is the headline Measurement-2 number. ε is chosen so a marginal-matched null cloud (each column permuted independently — same per-dim distribution, no cross-DoF structure) is hit with probability 0.01; `occ_real @ null-1%` is what fraction of the same ambient points land within that ε of the REAL data, and the deficit is log10 of the ratio. −1 dex = the real manifold occupies 10× less volume than a structureless cloud with identical marginals. `<1e-0X` / `<−X dex` are censored at the 1/M resolution limit, i.e. lower bounds, not measured values.

`tube slope` is the ε-occupancy log-log slope in the merged-tube regime (≈ codimension, a lower bound), `censored` where too few ambient points reach that regime. `tail slope` is the disjoint-ball diagnostic and should sit near D_eff — it is a health check on the estimator, not a result.

| embodiment | group | D | PR | d/D (PR) | pc90 | pc95 | pc99 | TwoNN | MLE | corr-dim | occ_real @ null-1% | volume deficit | tube slope | tail slope |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bridge | eef_position | 48 | 14.86 | 0.310 | 19 | 26 | 40 | 15.36±0.07 | 12.94 | 4.22 | 3.9e-02 | +0.60 dex | 8.8 (R²0.85) | 21.9 |
| bridge | eef_rotation | 48 | 26.43 | 0.551 | 32 | 39 | 46 | 19.03±0.16 | 14.05 | 4.68 | 3.3e-02 | +0.52 dex | 8.9 (R²0.85) | 22.2 |
| bridge | gripper | 16 | 2.63 | 0.164 | 4 | 6 | 12 | nan±nan | 8.00 | 2.23 | 3.5e-04 | -1.46 dex | censored | 12.4 |
| bridge | eef_posrot | 96 | 35.31 | 0.368 | 48 | 60 | 81 | 19.52±0.08 | 15.05 | 5.57 | 1.8e-01 | +1.24 dex | 12.8 (R²0.84) | 33.4 |
| bridge | all_trained | 112 | 32.40 | 0.289 | 51 | 66 | 92 | 14.49±0.15 | 8.62 | 5.95 | 1.8e-01 | +1.26 dex | 14.6 (R²0.84) | 38.1 |
| robocasa_mg | base_motion | 64 | 0.00 | 0.000 | 0 | 0 | 0 | nan±nan | nan | nan | — | — | censored | — |
| robocasa_mg | control_mode | 16 | 0.00 | 0.000 | 0 | 0 | 0 | nan±nan | nan | nan | — | — | censored | — |
| robocasa_mg | eef_position | 48 | 4.06 | 0.085 | 5 | 8 | 24 | 8.64±0.12 | 8.22 | 2.33 | 6.4e-02 | +0.80 dex | 8.5 (R²0.85) | 20.4 |
| robocasa_mg | eef_rotation | 48 | 5.06 | 0.105 | 11 | 26 | 43 | 18.95±0.29 | 12.27 | 3.02 | 4.5e-02 | +0.65 dex | 8.5 (R²0.85) | 21.2 |
| robocasa_mg | gripper_close | 16 | 1.12 | 0.070 | 1 | 2 | 4 | nan±nan | nan | 2.45 | 7.0e-05 | -2.15 dex | censored | 12.3 |
| robocasa_mg | eef_posrot | 96 | 8.54 | 0.089 | 13 | 28 | 61 | 9.48±0.16 | 9.74 | 2.82 | 2.9e-01 | +1.47 dex | 12.5 (R²0.84) | 32.8 |
| robocasa_mg | all_trained | 192 | 9.50 | 0.049 | 13 | 27 | 65 | 8.46±0.17 | 8.86 | 3.28 | 2.9e-01 | +1.46 dex | 14.1 (R²0.84) | 36.6 |
| gr1_tabletop | arm | 224 | 8.26 | 0.037 | 10 | 12 | 19 | 5.47±0.07 | 4.31 | 3.61 | 9.5e-01 | +1.98 dex | 20.0 (R²0.83) | 54.6 |
| gr1_tabletop | hand | 192 | 2.81 | 0.015 | 3 | 5 | 12 | nan±nan | nan | 0.08 | 1.9e-01 | +1.28 dex | censored | 48.6 |
| gr1_tabletop | waist | 48 | 3.40 | 0.071 | 3 | 5 | 12 | 12.31±0.10 | 8.57 | 1.32 | 6.4e-03 | -0.20 dex | 14.6 (R²0.97) | 20.4 |
| gr1_tabletop | all_trained | 464 | 9.30 | 0.020 | 14 | 18 | 39 | 4.90±0.07 | 3.22 | 3.22 | 1.0e+00 | +2.00 dex | 28.0 (R²0.83) | 77.6 |
| gr1_unified_1000 | arm | 224 | 8.24 | 0.037 | 10 | 12 | 19 | 5.65±0.10 | 4.36 | 3.59 | 9.5e-01 | +1.98 dex | 20.1 (R²0.83) | 53.1 |
| gr1_unified_1000 | hand | 192 | 2.81 | 0.015 | 3 | 5 | 12 | nan±nan | nan | 2.50 | 2.0e-01 | +1.30 dex | censored | 48.6 |
| gr1_unified_1000 | waist | 48 | 3.39 | 0.071 | 3 | 5 | 12 | 12.58±0.16 | 8.67 | 1.31 | 5.8e-03 | -0.24 dex | 15.4 (R²0.98) | 20.3 |
| gr1_unified_1000 | all_trained | 464 | 9.32 | 0.020 | 15 | 19 | 39 | 5.09±0.09 | 3.27 | 3.20 | 1.0e+00 | +2.00 dex | 28.1 (R²0.83) | 76.8 |
| dexjoco_single | arm_pos | 48 | 2.91 | 0.061 | 3 | 3 | 6 | 5.35±0.00 | 4.08 | 2.19 | 7.0e-02 | +0.84 dex | 14.5 (R²0.97) | 20.9 |
| dexjoco_single | arm_rot | 48 | 3.41 | 0.071 | 4 | 6 | 21 | 5.16±0.00 | 3.92 | 1.69 | 3.9e-02 | +0.60 dex | censored | 20.9 |
| dexjoco_single | arm_posrot | 96 | 5.27 | 0.055 | 6 | 8 | 22 | 5.60±0.00 | 4.31 | 2.72 | 3.1e-01 | +1.50 dex | 12.8 (R²0.84) | 32.3 |
| dexjoco_single | hand | 256 | 3.74 | 0.015 | 8 | 13 | 26 | 4.92±0.00 | 4.96 | 2.99 | 9.2e-01 | +1.97 dex | 21.8 (R²0.84) | 59.8 |
| dexjoco_single | all_trained | 352 | 6.06 | 0.017 | 13 | 19 | 45 | 6.20±0.00 | 5.86 | 4.25 | 9.8e-01 | +1.99 dex | 26.2 (R²0.83) | 69.9 |
| dexjoco_dual | right_arm | 96 | 4.80 | 0.050 | 5 | 6 | 14 | 4.80±0.00 | 3.84 | 2.48 | 2.8e-01 | +1.45 dex | 25.5 (R²0.98) | 31.9 |
| dexjoco_dual | left_arm | 96 | 5.03 | 0.052 | 5 | 6 | 13 | 4.10±0.00 | 3.63 | 2.65 | 3.2e-01 | +1.50 dex | 34.7 (R²0.99) | 32.7 |
| dexjoco_dual | arm | 192 | 8.00 | 0.042 | 9 | 11 | 26 | 5.32±0.00 | 4.96 | 3.48 | 7.7e-01 | +1.88 dex | 18.7 (R²0.84) | 49.2 |
| dexjoco_dual | right_hand | 256 | 4.02 | 0.016 | 6 | 9 | 18 | 3.33±0.00 | 2.45 | 2.21 | 6.4e-01 | +1.81 dex | censored | 62.6 |
| dexjoco_dual | left_hand | 256 | 3.58 | 0.014 | 6 | 9 | 20 | 3.36±0.00 | 1.65 | 2.08 | 7.3e-01 | +1.86 dex | censored | 58.9 |
| dexjoco_dual | hand | 512 | 6.71 | 0.013 | 10 | 16 | 36 | 3.47±0.00 | 2.08 | 3.44 | 9.7e-01 | +1.99 dex | censored | 86.0 |
| dexjoco_dual | all_trained | 704 | 9.34 | 0.013 | 17 | 26 | 57 | 4.00±0.00 | 4.22 | 4.30 | 9.8e-01 | +1.99 dex | 39.1 (R²0.83) | 106.1 |
