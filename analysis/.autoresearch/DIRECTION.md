# Research direction — Phase 0 (analysis-only probe suite)
_Written by the user-facing coordinator session. This is the authoritative brief for this run.
Master: read fully, then seed sections + queue accordingly._

## Paper hypothesis (the single central idea this run must test)
**As action DoF increases, the intent/meaning of an action becomes progressively less
accessible from raw action labels — a semantically compressed, visually grounded action
latent recovers it.** This motivates the larger paper: a universal action tokenizer
(pretrain on 20M mixed human+robot samples → cheap per-embodiment finetune) whose latent is a
better VLA/diffusion target than raw actions.

Phase 0 = offline evidence for the motivation, gating the storyline choice:
- If only the learned (DINO-grounded) latent recovers intent → "grounding recovers intent" story.
- If PCA recovers most of it → "compression/denoising" story. Either outcome is publishable
  motivation; measure honestly and let results decide.

## ⛔ HARD CONSTRAINTS (user-imposed, absolute, restate in every sub-brief)
1. **Analysis + small probe training ONLY. NEVER train a VLA. NEVER use more than 1 GPU.**
   GPU access ONLY via: `srun --gpus=1 --nodes=1 --comment "train gr00t on robocasa" --pty /bin/bash`
   (batch equivalent: same flags with a script). Prefer CPU where possible.
2. **All new code and results live under `/sjw_alinlab1/home/jungwook/action_tokenizer/analysis/`**
   (results under `.autoresearch/results/` inside it are fine — that is within analysis/).
3. **NEVER modify or delete any pre-existing file** (user global rule; also never `scancel`
   any job). Creating new files is allowed.
4. Cluster etiquette: conda env `gr00t-actlat`; on shared/login nodes set
   `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8`
   (sklearn oversubscription hangs the login node — happened before).

## Assets already on this server (verified 2026-07-21)
- **Tokenizer checkpoints** `action_tokenizer/checkpoints_action_tokenizer/`:
  `joint_soupv1_v4_recon_dino_bn64_l1_mse_naiveln_vae_embtok` (pretrained, 5 embodiments,
  class tokens 0-4) + `_ft_dexdual*`, `_ft_dexsingle*`, `_ft_robocasa100` finetuned variants;
  per-embodiment `dexjoco_{dual,single}_arm_v3/v4_*`; `gr1_1000demos_v4_*`.
  Also `Isaac-GR00T/checkpoints_action_tokenizer/`: `gr1_1000demos_v3_recon_ln_bn16`,
  `gr1_1000demos_v4_recon_dino_bn64_l1_mse_naiveln_vae`, robocasa_100demos_v3_*, etc.
  Load with `ActionLatentTokenizerWrapper.from_checkpoint(ckpt_dir)` from the
  **action_tokenizer repo** (auto-detects arch from state_dict; Isaac-GR00T copy lacks v4 files).
  VAE: use posterior mean μ for analysis (deterministic; σ≈0.018 so z≈μ). Latent = [T=16, K].
- **Datasets**: gr1_unified (24 PnP tasks) at
  `/storage1/sjw_dataset/dataset/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.*`;
  dexjoco v20 lerobot; EgoDex (~194 task folders, hdf5+_resized.mp4, non-lerobot loader
  `gr00t/data/dataset_egodex_frames_v4.py`, see sbatch_scripts/multiemb configs); robocasa.
  Verify exact local paths before use (b200 script paths are a DIFFERENT server).
  Val split reproduction: `ActionFramesDatasetV4(split="val", use_fixed_val=True, val_seed=42,
  val_ratio=0.003)` reads/creates `<dataset>/meta/fixed_val_split.json`.
- **Prior analysis (reuse, do not redo)**: `analysis/output/visual_sep_gr1/cache.npz`
  (N=4008 gr1 chunks: A, Z3, Z4μ, DINO Vcontext/Vdyn, task ids) + dexjoco cache;
  `analysis/VISUAL_SEP_REPORT.md`. Key prior findings usable as motivation evidence:
  gr1 near-dup action groups (6 groups, 78-254 members) EACH SPAN 13-17 DIFFERENT TASKS =
  direct evidence raw actions lack intent; R²(act→z): v3 0.984 vs v4 0.899; partial-corr
  corr(Δz,Δvis|Δact): v4 0.11-0.21, v3 ≈0. `analysis/effdim_*.py` = effective-dim machinery.

## Sections to seed (Phase 0)
- **S1 `intent-probe` (PRIORITY, gates storyline):** task/intent decodability probes.
  Predict episode task label from a single action chunk: raw flattened [16×D] vs PCA-k
  (k = latent budget, e.g. 64·16 → match dims!) vs learned latent (v3 action-only AND v4
  DINO-fused, μ). Linear probe + small MLP probe, proper train/val split BY EPISODE (never
  chunk-level split — leakage). Datasets in DoF order: robocasa (low DoF) → gr1 29 →
  dexjoco dual 44-dim config → EgoDex hands (many tasks). Deliverable: accuracy-vs-DoF figure,
  raw declining / latent flat is the hypothesis-consistent outcome. Also per-dim
  (accuracy of best single subspace) variant. Careful: #tasks differs per dataset →
  report chance-normalized accuracy (and macro-F1).
- **S2 `dof-redundancy`:** nominal vs effective DoF (participation ratio, n_pc95) across
  datasets (reuse effdim code patterns as NEW scripts under analysis/, do not modify old);
  intent-variance decomposition: fraction of action variance that is task-predictive
  (between-task / total, per PCA rank); **within-embodiment DoF control:** on gr1 (29) and
  EgoDex, probe intent from arm-only subset vs full body+hands vs top-k effective PCs —
  same episodes, only representation varies (kills the dataset confound).
- **S3 `label-noise-shortcut`:** (a) spectral/jerk analysis: high-frequency energy of raw
  action dims vs tokenizer reconstruction (denoising evidence); (b) blind predictability:
  predict a_t chunk from action history alone (copycat shortcut) — raw vs latent targets,
  history-only R²; high raw predictability = shortcut-prone supervision (cite copycat/causal
  confusion literature); (c) temporal autocorrelation of raw vs latent token sequences.
- **S4 `semantic-geometry` (cheap, reuse caches):** NN retrieval precision@k by task label
  and clustering NMI/ARI: raw vs PCA vs v3 vs v4 on gr1 + dexjoco caches; add EgoDex if S1
  collection lands. Note prior finding: unsupervised action clusters ≠ task id (ARI≈0.05 on
  gr1) — that LOW number for raw actions is itself hypothesis evidence; frame it that way.

## Expected headline figure
X-axis: nominal DoF (or dataset ordered by DoF); Y-axis: chance-normalized intent-probe
accuracy; lines: raw / PCA / v3 latent / v4 latent. Hypothesis-consistent: raw declines with
DoF, v4 stays high; PCA in between tells us the mechanism split.

## Notes for Experiment agent
- New collection scripts: model NEW code on `analysis/vsep_collect.py` (encode val chunks →
  npz cache). For EgoDex use the egodex dataset class; cap episodes for tractability
  (e.g. ≤50 eps/task × ~50-100 tasks first, scale later); GPU only needed for DINO+encoder
  forward passes — batch them, one srun session, release when done.
- Probes are small (sklearn logistic / 2-layer MLP) — CPU or the same GPU session.
- Every result dir needs run_meta.json (git commit, command, seed).

## Notes for Writing agent
- Storyline C-spine ("universal pretrain→finetune action tokenizer") is the PAPER; Phase-0
  report sections here are the MOTIVATION evidence. Keep each section single-idea.
- The user's framing to preserve: "my research focuses on dexterous action; as DoF increases,
  intent is harder to get from raw labels."
