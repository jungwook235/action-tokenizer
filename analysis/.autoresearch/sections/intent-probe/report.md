# S1 · Intent decodability vs action DoF (raw / PCA / v3 / v4 latent)

## TL;DR
- **Main result:** as action DoF rises, task/intent gets harder to read from *raw* action chunks — chance-normalized decode accuracy falls from {{PH:s1-lin-robo-raw | linear raw CNA, robocasa (low DoF) | expect=~0.70}} to {{PH:s1-lin-ego-raw | linear raw CNA, EgoDex (high DoF, many tasks) | expect=~0.30}} — while the visually-grounded **v4 latent stays high** ({{PH:s1-lin-robo-v4 | linear v4 CNA, robocasa | expect=~0.82}} → {{PH:s1-lin-ego-v4 | linear v4 CNA, EgoDex | expect=~0.68}}).
- **Why it matters:** this is the load-bearing motivation for a universal action tokenizer — raw high-DoF actions are a poor carrier of intent, and a learned latent puts intent back.
- **Status:** validate the raw < PCA < v4 ordering on the favorable gr1 regime first (reuses `cache.npz`, no new GPU), then widen to robocasa / dexjoco / EgoDex; the v4−PCA margin gates the grounding-vs-compression story.

---

**Chance-normalized accuracy (CNA)** = (acc − chance) / (1 − chance), so 0 = chance, 1 = perfect;
chance = 1/#tasks per dataset. All splits are **by episode** (never chunk-level — adjacent chunks
share frames and near-identical actions, which leaks). Task label = the episode's task id decoded
from a **single** action chunk (T=16). Representations, all at matched budget:

- **raw** — flattened action chunk `[16×D]`.
- **PCA-k** — top-k PCA of raw, k set to the learned latent budget (v4 = 16×64 = 1024; report also k=16×16 to match v3).
- **v3** — action-only learned latent (`[16×16]`), no vision.
- **v4** — DINO-fused VAE latent, posterior mean μ (`[16×64]`).

## Hero result — intent accessibility collapses for raw, holds for v4

![Accuracy vs DoF]({{PH:s1-fig-url | public URL of accuracy-vs-DoF figure (4 lines: raw/PCA/v3/v4), x=nominal DoF | expect=hosted PNG; raw declines, v4 flat-high, PCA between, v3≈PCA}})

**Linear probe, chance-normalized accuracy (↑ better):**

| Dataset (DoF order) | nominal DoF | #tasks | raw | PCA-k | v3 | **v4 (μ)** |
|---|---|---|---|---|---|---|
| robocasa (low) | {{PH:s1-dof-robo | robocasa nominal action dim | expect=~14–19}} | {{PH:s1-ntask-robo | #robocasa tasks | expect=~8–24}} | {{PH:s1-lin-robo-raw}} | {{PH:s1-lin-robo-pca | linear PCA CNA, robocasa | expect=~0.72}} | {{PH:s1-lin-robo-v3 | linear v3 CNA, robocasa | expect=~0.72}} | **{{PH:s1-lin-robo-v4}}** |
| gr1 | 29 | {{PH:s1-ntask-gr1 | #gr1 tasks | expect=24}} | {{PH:s1-lin-gr1-raw | linear raw CNA, gr1 | expect=~0.50}} | {{PH:s1-lin-gr1-pca | linear PCA CNA, gr1 | expect=~0.58}} | {{PH:s1-lin-gr1-v3 | linear v3 CNA, gr1 | expect=~0.60}} | **{{PH:s1-lin-gr1-v4 | linear v4 CNA, gr1 | expect=~0.76}}** |
| dexjoco-dual | 44 | {{PH:s1-ntask-dex | #dexjoco-dual tasks | expect=~5}} | {{PH:s1-lin-dex-raw | linear raw CNA, dexjoco (⚠ few disjoint tasks → high) | expect=~0.75}} | {{PH:s1-lin-dex-pca | linear PCA CNA, dexjoco | expect=~0.80}} | {{PH:s1-lin-dex-v3 | linear v3 CNA, dexjoco | expect=~0.82}} | **{{PH:s1-lin-dex-v4 | linear v4 CNA, dexjoco | expect=~0.87}}** |
| EgoDex (hands) | {{PH:s1-dof-ego | EgoDex nominal action dim | expect=~48+}} | {{PH:s1-ntask-ego | #EgoDex tasks probed | expect=~50–100}} | {{PH:s1-lin-ego-raw}} | {{PH:s1-lin-ego-pca | linear PCA CNA, EgoDex | expect=~0.45}} | — *(v3 N/A)* | **{{PH:s1-lin-ego-v4}}** |

*v3 is per-embodiment (robocasa/gr1/dexjoco only); EgoDex is covered solely by the multi-embodiment v4, so its v3 cell is N/A by design, not omitted.*

> **Reading the trend honestly.** dexjoco-dual is high-DoF but has only ~5 near-disjoint tasks, so raw actions are trivially separable there — its high raw CNA is a *#tasks* effect, not a counter-example. The confound-free DoF signal therefore rests on **EgoDex** (high DoF *and* many tasks) here, and on the **within-embodiment DoF control in S2** (same episodes, DoF varied by representation). Chance-normalization + macro-F1 are reported precisely to blunt the #tasks confound.

<details id="shared-setup">
<summary><b>Shared experiment setup (canonical — S2–S4 reference this)</b></summary>

<!-- The one DRY setup block. Other sections point here; only per-table deltas are re-stated there. -->

- **Tokenizers.** Load via `ActionLatentTokenizerWrapper.from_checkpoint(ckpt)` from the **action_tokenizer** repo (auto-detects arch). VAE latent = posterior mean μ (deterministic; σ≈0.018).
  - gr1: `Isaac-GR00T/checkpoints_action_tokenizer/gr1_1000demos_{v3_recon_ln_bn16, v4_recon_dino_bn64_l1_mse_naiveln_vae}/checkpoint-100000`
  - dexjoco: `checkpoints_action_tokenizer/dexjoco_dual_arm_{v3,v4}_*`
  - robocasa: `Isaac-GR00T/checkpoints_action_tokenizer/robocasa_100demos_v3_*` + v4 variant
  - EgoDex: multi-embodiment `checkpoints_action_tokenizer/joint_soupv1_v4_recon_dino_bn64_l1_mse_naiveln_vae_embtok` (class tokens 0–4), v4 only
- **Datasets & val split.** gr1_unified (24 PnP), dexjoco v20 lerobot, robocasa_gr1_tabletop, EgoDex (hdf5+`_resized.mp4`, loader `gr00t/data/dataset_egodex_frames_v4.py`). Reproduce val with `ActionFramesDatasetV4(split="val", use_fixed_val=True, val_seed=42, val_ratio=0.003)`. **Verify local dataset paths before use** (b200 script paths are a different server).
- **Prior cache reused (no re-encode):** `analysis/output/visual_sep_gr1/cache.npz` (gr1, N=4008: A, Z3, Z4μ, DINO Vcontext/Vdyn, task ids) + dexjoco cache.
- **Probes.** Linear = multinomial logistic regression; MLP = 2-layer (sklearn / small torch). **Episode-level** train/val split; standardize features on train only.
- **Env.** conda `gr00t-actlat`; on shared nodes set `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8` (sklearn oversubscription has hung the login node). GPU only for DINO+encoder forward passes (one `srun --gpus=1`), released after collection; probes are CPU.
</details>

<details>
<summary><b>Reproducibility — hero table</b></summary>

- **Collection script:** NEW `analysis/intent_probe_collect.py` (model on `analysis/vsep_collect.py`); caches per-dataset `A, Z3, Z4μ, task, episode_id`. EgoDex capped ≤50 eps/task first.
- **Probe script:** NEW `analysis/intent_probe.py` → `.autoresearch/results/<EXPID>/intent_probe.json` (per dataset×rep: acc, CNA, macro-F1, chance, #tasks, n_train/n_val episodes).
- **Command:** `python analysis/intent_probe.py --datasets robocasa gr1 dexjoco egodex --probe linear mlp --split episode --seed 0`
- **Seed:** 0 · **Run metadata:** branch `{{PH:s1-git-branch | git branch | expect=master}}`, commit `{{PH:s1-git-commit | short commit | expect=<hash>}}`.
- **Raw results:** `.autoresearch/results/{{PH:s1-expid | experiment id | expect=exp-00NN}}/`
</details>

## Robustness — not a linear-probe or #tasks artifact

**MLP probe, chance-normalized accuracy** (nonlinear head; if raw's decline survived here it's not a linear-separability artifact):

| Dataset | raw | PCA-k | v3 | **v4 (μ)** |
|---|---|---|---|---|
| robocasa | {{PH:s1-mlp-robo-raw | MLP raw CNA, robocasa | expect=~0.74}} | {{PH:s1-mlp-robo-pca | MLP PCA CNA, robocasa | expect=~0.76}} | {{PH:s1-mlp-robo-v3 | MLP v3 CNA, robocasa | expect=~0.76}} | **{{PH:s1-mlp-robo-v4 | MLP v4 CNA, robocasa | expect=~0.84}}** |
| gr1 | {{PH:s1-mlp-gr1-raw | MLP raw CNA, gr1 | expect=~0.58}} | {{PH:s1-mlp-gr1-pca | MLP PCA CNA, gr1 | expect=~0.64}} | {{PH:s1-mlp-gr1-v3 | MLP v3 CNA, gr1 | expect=~0.66}} | **{{PH:s1-mlp-gr1-v4 | MLP v4 CNA, gr1 | expect=~0.78}}** |
| dexjoco-dual | {{PH:s1-mlp-dex-raw | MLP raw CNA, dexjoco | expect=~0.80}} | {{PH:s1-mlp-dex-pca | MLP PCA CNA, dexjoco | expect=~0.84}} | {{PH:s1-mlp-dex-v3 | MLP v3 CNA, dexjoco | expect=~0.85}} | **{{PH:s1-mlp-dex-v4 | MLP v4 CNA, dexjoco | expect=~0.89}}** |
| EgoDex | {{PH:s1-mlp-ego-raw | MLP raw CNA, EgoDex | expect=~0.38}} | {{PH:s1-mlp-ego-pca | MLP PCA CNA, EgoDex | expect=~0.50}} | — | **{{PH:s1-mlp-ego-v4 | MLP v4 CNA, EgoDex | expect=~0.70}}** |

- **Macro-F1 mirrors accuracy ordering** (guards against class imbalance): gr1 raw {{PH:s1-f1-gr1-raw | macro-F1 raw, gr1 | expect=~0.45}} vs v4 {{PH:s1-f1-gr1-v4 | macro-F1 v4, gr1 | expect=~0.73}}; EgoDex raw {{PH:s1-f1-ego-raw | macro-F1 raw, EgoDex | expect=~0.26}} vs v4 {{PH:s1-f1-ego-v4 | macro-F1 v4, EgoDex | expect=~0.64}}.
- **Per-dim variant** (best *single* action subspace for raw): stays near chance at high DoF — gr1 {{PH:s1-perdim-gr1 | best single-dim raw CNA, gr1 | expect=~0.15}}, EgoDex {{PH:s1-perdim-ego | best single-dim raw CNA, EgoDex | expect=~0.08}} — intent is smeared across many raw dims, not localized.

<details>
<summary><b>Reproducibility — robustness</b></summary>

- Same collection + probe scripts as the hero table (`--probe mlp`, `--per-dim`, `--metric macro_f1`); same episode split & seed 0. Raw: `.autoresearch/results/{{PH:s1-expid-rob | experiment id (robustness) | expect=exp-00NN}}/`.
</details>

## Gating verdict — grounding vs compression

PCA-k recovers part of the lost intent (compression/denoising); the **residual v4-over-PCA margin** is the visual-grounding contribution and decides the paper's framing:

- v4 − PCA margin (CNA, high-DoF avg over gr1+EgoDex): {{PH:s1-gate-margin | mean v4−PCA CNA margin, high-DoF | expect=~+0.15}}.
- **If margin ≳ +0.1 → "grounding recovers intent"** story (v4's vision does work PCA can't).
- **If margin ≈ 0 → "compression/denoising"** story (PCA suffices; vision is optional). Either is publishable motivation; we report the number and let it decide: {{PH:s1-gate-verdict | one-line gate outcome | expect=grounding — v4 clears PCA on high-DoF, many-task data}}.

## Takeaway
Intent is progressively lost from raw actions as DoF grows, and a learned latent recovers it — the single fact the universal-tokenizer paper is built on. S2 shows *why* (DoF is mostly redundant / task-irrelevant), S3 shows raw supervision is also noisier and shortcut-prone, and S4 shows the recovered intent is geometrically organized by task. If the S1 gate lands on "grounding", the recovery specifically needs vision — the strongest version of the motivation.
