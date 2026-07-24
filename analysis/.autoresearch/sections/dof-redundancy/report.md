# S2 · Nominal vs effective DoF, and where intent-variance lives

## TL;DR
- **Main result:** raw action chunks are massively redundant — gr1 raw uses effective dim **9.5** out of nominal **464 (48.7× redundant)**; per-timestep, **8.5** of nominal **29**. The nominal DoF vastly overstates the information carried.
- **Why intent is hard (ties to S1):** only **22.9%** of raw-action variance is task-predictive (between/total), and it is front-loaded — captured by the top ~2 PCs — so most DoF is task-irrelevant detail.
- **Confound-free (within-embodiment):** on the *same* gr1 episodes, arm-only (14d) intent ≈ full-body (0.285 vs 0.298) and the top-18 PCs recover 92% — intent is arm-localized and low-dimensional, so "more DoF ≠ more intent" is a property of DoF itself, not the dataset. (v4 latent also spends ~2× the effective dims of v3 — 19.2 vs 11.0 — cf. S4.)

---

Effective dim via participation ratio PR = (Σλ)²/Σλ² of per-dim z-scored covariance eigenvalues;
n_pc95 = #PCs for 95% variance. NEW scripts under `analysis/` (originals **not** modified). See
**S1 §Shared setup**. Landed for gr1 + dexjoco (from the caches); robocasa/EgoDex pending.

## Nominal DoF is mostly unused

**Raw action effective dimensionality:**

| Dataset | granularity | nominal DoF | PR (effective) | n_pc95 | redundancy = nom/PR |
|---|---|---|---|---|---|
| gr1 | per-timestep | 29 | 8.52 | 14 | 3.4× |
| gr1 | 16-step chunk | 464 | 9.53 | 18 | **48.7×** |
| dexjoco | per-timestep | 44 | 7.14 | 14 | 6.2× |
| dexjoco | 16-step chunk | 704 | 7.28 | 16 | **96.7×** |
| robocasa | per-timestep | 12 | {{PH:s2-pr-robo | raw-action PR, robocasa (pending) | expect=~7–9}} | {{PH:s2-pc95-robo | n_pc95 raw, robocasa (pending) | expect=~12}} | {{PH:s2-redsingle-robo | nom/PR, robocasa (pending) | expect=~1.5}} |
| EgoDex | per-timestep | {{PH:s2-nom-ego | EgoDex nominal DoF | expect=~48+}} | {{PH:s2-pr-ego | raw-action PR, EgoDex | expect=~10–15}} | {{PH:s2-pc95-ego | n_pc95 raw, EgoDex | expect=~20}} | {{PH:s2-redsingle-ego | nom/PR, EgoDex | expect=~4}} |

The per-timestep effective DoF (~7–9) sits far below nominal for both embodiments; per chunk the
redundancy is extreme (48–97×) because the 16 steps are highly correlated in time (cf. S3
autocorrelation).

**Latent effective dimensionality (chunk):**

| Dataset | v3 PR | v3 n_pc95 | v4 PR | v4 n_pc95 |
|---|---|---|---|---|
| gr1 | 11.01 | 15 | 19.21 | 82 |
| dexjoco | 6.93 | 10 | 15.21 | 41 |
| robocasa *(pending)* | {{PH:s2-pr-v3-robo | v3 PR, robocasa | expect=~10}} | {{PH:s2-pc95-v3-robo | v3 n_pc95, robocasa | expect=~14}} | {{PH:s2-pr-v4-robo | v4 PR, robocasa | expect=~16}} | {{PH:s2-pc95-v4-robo | v4 n_pc95, robocasa | expect=~50}} |

<details>
<summary><b>Reproducibility — effective-dim (exp-0003)</b></summary>

- **Script:** `analysis/dof_effdim.py` (PR + n_pc95, z-scored; per-timestep + chunk), reading gr1/dexjoco caches.
- **Seed:** 0 · branch `master`, commit `f6e29c1` (dirty) · Raw: `.autoresearch/results/exp-0003/results.json`.
- Setup: see **S1 §Shared setup**.
</details>

## Intent lives in a few low-index PCs

Fraction of raw-action variance that is task-predictive (between-task/total), and how few PCs
carry it:

| Dataset | between/total var | #PCs for ~90% of between-task var | intent-acc saturation k* |
|---|---|---|---|
| gr1 | 0.229 | ~2 | 32 |
| dexjoco (control) | 0.430 | ~1 | ~4 |
| EgoDex *(pending)* | {{PH:s2-btw-ego | between/total task variance, EgoDex | expect=low, <gr1}} | {{PH:s2-btwpc-ego | #PCs for 90% between-task var, EgoDex | expect=~few}} | {{PH:s2-satk-ego | intent-acc saturation k*, EgoDex | expect=small}} |

gr1's between/total is flat from rank 2 (0.234) through rank 256 (0.229) — the task-predictive
variance is captured almost entirely by the first ~2 PCs; adding PCs adds only task-irrelevant
variance. dexjoco's higher 0.430 reflects its disjoint action spaces (actions carry more task
info — the S1 control regime). Note the split: task-predictive *variance* saturates by rank ~2,
but intent *accuracy* (S1 PCA sweep) keeps rising to k≈32 before declining — low-variance
directions still add a little linear separability.

<details>
<summary><b>Reproducibility — intent-variance decomposition (exp-0003)</b></summary>

- **Script:** `analysis/dof_effdim.py` (between/total variance per PCA rank). Raw: `.autoresearch/results/exp-0003/results.json` (`intent_between_total`, `intent_var_vs_rank_*`).
</details>

## Within-embodiment DoF control (kills the dataset confound)

Same gr1 episodes + LOEPTO split as S1 — only the *action representation* changes. Intent is
**arm-localized and low-dimensional**: arm-only ≈ full body, non-arm DoF adds ~nothing, and the
top-18 PCs (= n_pc95) recover ~92% of full.

| gr1 (linear CNA) | arm-only (14d) | full (29d) | hands (12d) | waist (3d) | top-18 PCs |
|---|---|---|---|---|---|
| accuracy | 0.285 | 0.298 | 0.061 | 0.039 | 0.273 |

**Reading:** full − arm-only = **+0.013** — the two arms carry essentially all the decodable
intent; **hands (0.061) and waist (0.039) add almost nothing** on top. The top-18 principal
components (= n_pc95) recover **91.6%** of the full-DoF accuracy. So *more raw DoF ≠ more intent*:
the effect is a property of DoF itself, not of which dataset it came from — the dataset confound
in S1 is controlled out. (`full-29` reproduces S1's raw gr1 CNA exactly, 0.298 — a
split-reproducibility check.) EgoDex arm-vs-full to follow once its cache lands.

<details>
<summary><b>Reproducibility — within-embodiment control (exp-0005)</b></summary>

- **Script:** `analysis/dof_within_embodiment.py` (arm/hands/waist column masks + top-k PCA on the full action; action order `left_arm[0:7] right_arm[7:14] left_hand[14:20] right_hand[20:26] waist[26:29]`). Same gr1 LOEPTO 3-fold split & seed 0. Raw: `.autoresearch/results/exp-0005/results.json`.
</details>

## Takeaway
Nominal DoF vastly overstates the information an action carries (gr1: 464 → 9.5 effective, 48.7×;
dexjoco 704 → 7.3, 96.7×) and only ~23% of the raw variance is task-predictive, front-loaded in
~2 PCs. Compressing raw actions therefore discards motion detail, not intent — the quantitative
license for a compact learned latent to be a lossless-for-intent, better VLA target, and the
robust landed motivation even where S1's mid-DoF probe gap is modest. S3 shows the discarded
surplus is not merely redundant but actively harmful; S4 shows the retained intent is
geometrically organized.
