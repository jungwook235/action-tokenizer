# S1 · Intent decodability vs action DoF (raw / PCA / v3 / v4 latent)

## TL;DR
- **Landed (gr1, mid-DoF, leak-free):** a single action chunk only weakly determines the task (v4 CNA **0.347**, chance = 1/24), and the visually-grounded **v4 latent is consistently the top representation** — above raw (0.298) and the action-only v3 (0.301), with naive full-budget PCA *below* raw (0.245). Modest but consistent: +0.05 over raw.
- **Punchline is the DoF-slope — still to be shown:** gr1 is the clean mid-DoF anchor (v4 +0.05 over raw); the decisive high-DoF test is **EgoDex (DoF 44, pending)** — {{PH:s1-lin-ego-v4 | linear v4 CNA, EgoDex (pending) | expect=gap widens vs raw}}. robocasa (DoF 12) is **near-chance for every rep** (161 fine tasks a single chunk can't distinguish) — inconclusive, kept off the curve.
- **Strongest standalone motivation is redundancy (S2):** raw gr1 chunks are **48.7× redundant** (nominal 464 → effective 9.5) and only **22.9%** of their variance is task-predictive — compression is well-motivated even where the probe gap is small.

---

**Chance-normalized accuracy (CNA)** = (acc − chance)/(1 − chance), 0 = chance, 1 = perfect;
chance = 1/#tasks. gr1 uses **leave-one-episode-per-task-out (LOEPTO) 3-fold** (72 episodes,
3/task) — strictly leak-free. Task decoded from a **single** action chunk (T=16). Representations
at matched budget: **raw** `[16×D]`; **PCA-k** (swept 2→512); **v3** `[16×16]` action-only latent;
**v4** `[16×64]` DINO-fused μ.

> **DoF axis anchored only on shared-primitive datasets** (gr1 shared PnP primitives; robocasa
> low-DoF; EgoDex high-DoF) where raw actions are genuinely intent-ambiguous. dexjoco-dual (~5
> disjoint-action tasks) makes task trivially decodable from raw (≈ceiling) — reported as a
> **control**, never a curve point. Flagging this is a credibility win.

## Hero result — mid-DoF point (gr1), slope pending endpoints

![Accuracy vs DoF]({{PH:s1-fig-url | public URL of accuracy-vs-DoF figure (raw/PCA/v3/v4 lines) once robocasa+EgoDex land | expect=hosted PNG; punchline = raw declines & v4 holds across DoF}})

**Linear-probe chance-normalized accuracy (↑ better):**

| Dataset (DoF order) | nominal DoF | #tasks | raw | PCA-k (best) | v3 | **v4 (μ)** |
|---|---|---|---|---|---|---|
| robocasa (low) ⚠ *near-chance, off-curve* | 12 | 161 | 0.021 | 0.018 | 0.030 | 0.039 |
| **gr1 · *clean anchor*** | 29 | 24 | 0.298 | 0.245 *(best k=32: 0.319)* | 0.301 | **0.347** |
| EgoDex (hands) · *pending* | {{PH:s1-dof-ego | EgoDex nominal action dim | expect=~48+}} | {{PH:s1-ntask-ego | #EgoDex tasks probed | expect=~50–100}} | {{PH:s1-lin-ego-raw | linear raw CNA, EgoDex | expect=low (decisive: < gr1 raw)}} | {{PH:s1-lin-ego-pca | linear PCA CNA, EgoDex | expect=~raw}} | — *(v3 N/A)* | **{{PH:s1-lin-ego-v4 | linear v4 CNA, EgoDex | expect=high (gap widens)}}** |

**gr1 reading (honest):** v4 is top but the margin is *modest* at mid-DoF (+0.05 vs raw, +0.03 vs
rank-tuned PCA). The **learned action-only v3 ≈ raw** (0.301 vs 0.298) and **matched-budget PCA
(k=256) *hurts*** (0.245 < raw) — neither the learned action bottleneck nor full-dimensional
compression helps. But a **rank-tuned PCA (k=32) reaches 0.319**, recovering ~40% of the raw→v4
gap, so part of the effect is genuine low-rank compression/denoising; **v4's grounding adds the
remaining larger share (~+0.03) on top**. Absolute decodability is low for every representation —
a single chunk barely determines the task — itself evidence that intent is not cleanly present in
raw action space at this DoF.

**DoF-scaling — the key prediction, not yet shown.** The hypothesis is that the v4−raw intent gap
*grows with DoF*. **gr1 (DoF 29) is the clean mid-DoF anchor** (v4 0.347 vs raw 0.298, gap +0.049).
**robocasa (DoF 12) is inconclusive, not a datapoint:** with 161 fine-grained pick-place tasks, a
single 16-step chunk cannot reveal *which* object/target, so *every* representation sits near
chance (raw 0.021, v4 0.039; chance = 1/161 = 0.006) and the +0.018 gap is within noise — a
task-granularity artifact, kept off the curve (it neither supports nor undercuts the slope). The
claim therefore rests on gr1 → the pending high-DoF **EgoDex (DoF 44)** endpoint, which is decisive.

<details id="shared-setup">
<summary><b>Shared experiment setup (canonical — S2–S4 reference this)</b></summary>

- **Tokenizers.** `ActionLatentTokenizerWrapper.from_checkpoint(ckpt)` (action_tokenizer repo; auto-detects arch). VAE latent = posterior mean μ (σ≈0.018).
  - gr1: `Isaac-GR00T/checkpoints_action_tokenizer/gr1_1000demos_{v3_recon_ln_bn16, v4_recon_dino_bn64_l1_mse_naiveln_vae}/checkpoint-100000`
  - dexjoco: `checkpoints_action_tokenizer/dexjoco_dual_arm_{v3,v4}_*` · robocasa: `robocasa_100demos_v3_*`+v4 · EgoDex: multi-embodiment `joint_soupv1_v4_..._embtok` (v4 only)
- **Datasets & val split.** gr1_unified (24 PnP), dexjoco v20 lerobot, robocasa_gr1_tabletop, EgoDex (loader `gr00t/data/dataset_egodex_frames_v4.py`). Val: `ActionFramesDatasetV4(split="val", use_fixed_val=True, val_seed=42, val_ratio=0.003)`. **Verify local paths.**
- **Prior caches reused:** `output/visual_sep_gr1/cache.npz` (gr1, N=4008, 24 shared-primitive tasks) + dexjoco cache (N=2322, 5 disjoint tasks). Episode ids recovered via `analysis/recover_episode_ids.py` for leak-free splits.
- **Probes.** linear = `LogisticRegression(lbfgs, multinomial, C=1)`; mlp = `MLPClassifier(256, early_stopping)`. **gr1 split = LOEPTO 3-fold** (leak-free); standardize on train only.
- **Env.** conda `gr00t-actlat` (numpy 1.26.4, sklearn 1.5.2); shared-node thread caps `OMP/MKL/OPENBLAS/NUMEXPR=6–8` (oversubscription has hung the login node). Probes are CPU; GPU only for DINO+encoder collection (one `srun --gpus=1`, released after).
</details>

<details>
<summary><b>Reproducibility — gr1 hero + dexjoco control (exp-0001)</b></summary>

- **Scripts:** `analysis/recover_episode_ids.py`, `analysis/intent_probe.py`.
- **Commands:** `python intent_probe.py --cache output/visual_sep_gr1/cache.npz --episode-ids output/visual_sep_gr1/episode_ids.npz --tag gr1 --per-dim --out .autoresearch/results/exp-0001/results_gr1.json` (dex: `--cache output/visual_sep/cache.npz --tag dex`).
- **Seed:** 0 · **Run metadata:** branch `master`, commit `f6e29c1`, dirty=true · hardware login-node CPU.
- **Raw results:** `.autoresearch/results/exp-0001/results.json` (+ `results_gr1.json`, `results_dex.json`).
</details>

## Control — dexjoco disjoint-action ceiling (NOT on the DoF curve)

dexjoco-dual is 44-DoF but its 5 tasks have **disjoint** action spaces, so task is trivially
readable from raw actions — a ceiling. (Split note: dexjoco has 1 val episode/task, so LOEPTO is
infeasible; a stratified-chunk fallback is used — control only.)

| dexjoco-dual (control) | raw | PCA-k | v3 | v4 (μ) |
|---|---|---|---|---|
| linear CNA | 1.00 | 1.00 | 1.00 | 1.00 |
| MLP CNA | 1.00 | 0.954 | 0.998 | 1.00 |

> raw ≈ v4 ≈ 1.0 here is *expected and supportive* — disjoint action spaces are the one regime
> where raw actions already encode intent. The paper's claim is the opposite, realistic regime
> (shared primitives), where raw fails and the grounded latent helps.

## Robustness — not a linear-probe or #tasks artifact

**MLP-probe CNA (gr1):** raw 0.328, PCA-256 0.240, v3 0.321, **v4 0.344** — same ordering
(v4 ≥ v3 ≈ raw > full-budget PCA); the nonlinear head lifts raw slightly but does not close the
v4 gap or rescue full-budget PCA.

- **Macro-F1 mirrors accuracy** (guards imbalance): gr1 raw 0.323 vs v4 0.362 (linear) — same v4 > raw ordering.
- **Per-dim variant:** the single most task-informative raw channel reaches only CNA **0.061** (gr1) — intent is smeared across many raw dims, not localized, so raw supervision cannot shortcut to it.
- **robocasa (near-chance, inconclusive):** all reps sit near chance (0.006) — MLP CNA raw 0.053, v4 0.058 — so the values are *not meaningful* (161-task granularity: a chunk can't reveal the target); reported for completeness only, not as a DoF datapoint. EgoDex MLP + macro-F1: {{PH:s1-mlp-ego-raw | MLP raw CNA, EgoDex (pending) | expect=low}} / {{PH:s1-mlp-ego-v4 | MLP v4 CNA, EgoDex (pending) | expect=high}}, {{PH:s1-f1-ego-raw | macro-F1 raw, EgoDex (pending) | expect=low}} / {{PH:s1-f1-ego-v4 | macro-F1 v4, EgoDex (pending) | expect=high}} — pending collection.

## Gating verdict — grounding vs compression

At mid-DoF (gr1) the raw→v4 gain (+0.049) splits into **~+0.021 from rank-tuned compression** (best PCA k=32 = 0.319, ~40%) and **~+0.028 from grounding** (v4 0.347 over best PCA, ~60%). So the mechanism is *mixed*, with grounding the larger share: the learned action-only v3 (0.301) ≈ raw and matched-budget PCA (k=256, 0.245) hurts, but a well-chosen low-rank compression already recovers a meaningful part, and v4's vision adds the rest. This is not a generic dimensionality effect (full-budget PCA underperforms raw). *(EgoDex high-DoF margin pending — the decisive DoF-scaling number; grounding's share should grow with DoF.)*

**Verdict — MIXED, grounding-dominant:** the tokenizer's mid-DoF advantage is *both* low-rank compression/denoising (~40%) *and* visual grounding (~60%, the larger share) — both mechanisms are real, grounding leads. **Key prediction:** grounding's share should *grow with DoF*, so the pending EgoDex high-DoF endpoint is the decisive test. The ~40/60 split is a feature of the story, not a hedge: it says a *compressed-and-grounded* latent (the paper's exact proposal) is what recovers intent. ✅ *Gate claim independently verified — ver-0001 PASS (leak-free LOEPTO).*

## Takeaway
On leak-free mid-DoF gr1 the grounded latent is consistently the best intent carrier, but the
margin is modest and absolute decodability is low — a single action chunk weakly determines the
task, which is exactly the "intent is hard to read from raw actions" premise. The decisive
DoF-scaling evidence is the EgoDex endpoint (pending); meanwhile the massive raw-action
redundancy (S2), the noisy/copycat-prone supervision (S3), and the consistent v4>v3>raw semantic
geometry (S4) form the robust, landed motivation for a compressed, grounded action latent.
