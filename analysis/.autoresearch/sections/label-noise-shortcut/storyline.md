# S3 Label noise & copycat shortcut — storyline

## Central idea (ONE sentence)
Raw action labels are a **copycat-prone supervision signal** — ~98% predictable from their own
recent history, so a policy can shortcut by extrapolating motion instead of reading the scene —
and the learned action-only latent (v3) does not fix this; only the visually-grounded v4 latent
resists the shortcut (least copyable, least autocorrelated), making it a better VLA target.

## Why it matters (the hook)
S1/S2 argue raw high-DoF actions hide intent; S3 shows that even so, raw actions are *dangerous
supervision*: the copycat / causal-confusion literature (Wen et al. 2020; de Haan et al. 2019)
shows highly self-predictable targets teach shortcuts. If switching the target from raw to the
grounded latent reduces blind predictability, that is a principled, DoF-independent reason to
tokenize — the supervised-learning pillar of the paper.

## Sub-claims that support the idea
1. **Copycat (headline):** blind history-only R² raw 0.980 ≈ v3 0.982 ≫ v4 0.915; naive-copy 3×
   harder for v4 (NMSE 0.104 vs 0.035). — evidence: (b) table.
2. **Autocorrelation:** v4 lag-1 0.941 vs raw/v3 ~0.99 — less temporal redundancy. — evidence: (c) table.
3. **De-correlation, not denoising (honest):** reconstruction is near-lossless (L1 0.0014, HF
   *higher* ×1.56, jerk unchanged) — the tokenizer does NOT smooth; instead the v4 latent spectrum
   is whiter (low-freq frac 0.387 vs 0.600), removing predictable drift. — evidence: (a) tables.

## Update — results landed (exp-0004, gr1, partial)
Pre-registered *denoising* sub-hypothesis **refuted** (recon is faithful, not smoother) — demoted
to an honest "de-correlation not denoising" note. Net a *stronger* story: the copycat headline is
solid and the recurring **v3 ≈ raw on supervision axes** theme (parallels S1) is clean — only v4
departs. Reframed the central idea from "denoising + copycat" to copycat-resistance via grounding.

## Favorable-first regime
gr1 `cache.npz` — all signals are CPU (ridge, autocorr, FFT); reconstruction decodes cached μ.

## Known risks / where it could break
- Blind-R² is high for everything (raw 0.98, v4 0.915) — the *reduction* is modest; lead with the
  more dramatic naive-copy 3× and frame v4 as "least copyable," not "breaks the shortcut."
- History-only predictor must use strictly past steps (it does: 2-step history) — no leakage.
- Denoising refutation must be stated plainly (no smoothing) to stay honest; the positive signal
  is de-correlation + copycat-resistance, not noise reduction.
