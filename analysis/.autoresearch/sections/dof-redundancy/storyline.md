# S2 DoF redundancy & intent-variance — storyline

## Central idea (ONE sentence)
High nominal action DoF is overwhelmingly redundant *and* the redundancy is task-irrelevant —
the effective (participation-ratio) dimensionality is a small, roughly constant fraction of
nominal, and only a handful of low-index principal components carry task/intent variance — so
piling on raw DoF adds motion detail but not readable intent, which is exactly why raw
high-DoF actions bury intent (S1) and a compact latent recovers it.

## Why it matters (the hook)
S1 shows raw intent-decodability falls with DoF; S2 explains the mechanism and removes the
cross-dataset confound. If intent is confined to a few effective dimensions regardless of
nominal DoF, then (a) the raw representation wastes most of its capacity on task-irrelevant
detail and (b) a low-dimensional learned latent loses nothing that matters — the compression
is *free* for intent. This is the quantitative backbone of the "compression helps" half of the
storyline gate.

## Sub-claims that support the idea
1. **Effective DoF ≪ nominal, and the ratio grows with nominal DoF.** Participation ratio and
   n_pc95 of raw actions stay ~8–12 while nominal grows 29→704. — evidence: effective-dim table.
2. **Intent lives in a few low-index PCs.** Between-task variance fraction is concentrated in
   the first few PCs; intent-probe accuracy saturates at top-k with k ≪ nominal. — evidence:
   intent-variance-per-rank curve + saturation-k.
3. **Within-embodiment control (confound-free):** on gr1 and EgoDex, intent accuracy from
   arm-only ≈ full body+hands ≈ top-k PCs — adding raw DoF does not add intent. Same episodes,
   only the representation changes, so this isolates DoF from every dataset difference. —
   evidence: within-embodiment table.

## Expected results (pre-registration)
- Raw effective dim ~8–12 across all datasets despite nominal 29→704 (prior effdim: gr1 raw
  PR≈8.5 single / 9.5 chunk; robocasa≈8.2/9.2).
- Between-task variance fraction small (order 10–30%) and front-loaded in low PCs.
- Within-embodiment: full − arm-only intent gap small; top-k (k≈effective dim) matches full.

## Update — results landed (gr1+dexjoco, exp-0003)
Confirmed strongly: raw gr1 chunk **48.7× redundant** (nominal 464 → PR 9.5), per-step 29→8.5;
only **22.9%** of raw variance is task-predictive, front-loaded in ~2 PCs. v4 PR 19.2 (n_pc95 82)
vs v3 11.0 (15). dexjoco even more redundant per chunk (96.7×), between/total 0.430 (disjoint →
more task info). **This is the strongest LANDED pillar** — robust regardless of S1's modest
mid-DoF probe gap. **Within-embodiment control landed (exp-0005):** intent is arm-localized —
arm-only (14d) CNA 0.285 ≈ full-29 0.298; hands 0.061, waist 0.039 add ~nothing; top-18 PCs
(=n_pc95) recover 91.6%. `full-29` == exp-0001 raw exactly (split reproducibility check). This
kills the dataset confound: more raw DoF ≠ more intent.

## Favorable-first regime
gr1 from `cache.npz` (raw A + latents already there) for the effective-dim + intent-variance
tables and the within-embodiment control (arm-only vs full is a column mask on A). No GPU.

## Known risks / where it could break
- Participation ratio is scale-sensitive → always z-score per dim (as prior effdim scripts do).
- "arm-only" DoF partition for gr1/EgoDex must match the true joint layout — verify indices
  against the data config before masking.
- If hands *do* carry large task variance in EgoDex, arm-only < full there — that is still
  informative (locates intent in the hand DoF) and should be reported, not suppressed.
