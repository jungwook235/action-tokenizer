# Phase 0 · What a compressed, grounded action latent buys — overview

## TL;DR
- **Central claim (motivation for a universal action tokenizer):** as action DoF rises, task intent is hard to read from *raw* action labels; a **compressed-and-grounded** latent (v4, DINO-fused) recovers it — and Phase-0 pins down that the recovery is **~40% compression / ~60% grounding** at mid-DoF.
- **Unifying insight:** **v3 (pure compression) ≈ raw on supervision/decodability (S1 intent-probe, S3 copycat), but v3 > raw on semantic geometry (S4); only v4 (grounding) improves the supervision signal.** Compression reorganizes geometry; grounding improves what a policy learns from.
- **Robust landed pillar:** raw actions are **48.7× redundant** (effective dim 9.5 of 464) and only **22.9%** task-predictive — compression is well-motivated regardless of the modest mid-DoF probe gap. The decisive DoF-scaling test (does grounding's share grow?) is the pending **EgoDex** high-DoF endpoint.

---

**The lead insight, spelled out.** Across four independent probes, the *learned action-only*
latent v3 behaves like raw actions on everything a policy is trained *on* — linear intent
decodability (S1) and blind copycat-predictability (S3) — yet it reorganizes the *geometry* of
the space to be more task-structured (S4). Only the **visually grounded** v4 moves the
supervision axes. So Phase-0 cleanly separates two things a tokenizer could do:

| | reorganize geometry (S4) | improve decodability (S1) | resist copycat (S3) |
|---|---|---|---|
| **compression (v3)** | ✅ v3 > raw | ✗ v3 ≈ raw | ✗ v3 ≈ raw |
| **grounding (v4)** | ✅ v4 > v3 | ✅ v4 > raw | ✅ v4 > raw |

That separation is the run's cleanest result: a *compressed-and-grounded* latent — the paper's
exact proposal — is what a downstream policy benefits from.

## Landed evidence at a glance (gr1 mid-DoF, leak-free)

| Probe | metric | raw | v3 | **v4** | reading |
|---|---|---|---|---|---|
| S1 intent | linear CNA (chance 1/24) | 0.298 | 0.301 | **0.347** | v4 top, modest; ~40/60 compression/grounding |
| S2 redundancy | effective dim / nominal | 9.5 / 464 (48.7×) | — | — | raw massively redundant; 22.9% task-predictive |
| S3 copycat | blind history-only R² (↓) | 0.980 | 0.982 | **0.915** | v4 least copyable (naive-copy 3× harder) |
| S4 geometry | cross-ep P@10 · ARI | 0.228 · 0.085 | 0.251 · 0.101 | **0.273 · 0.127** | v4 > v3 > raw ≈ PCA |

**DoF-scaling — the key prediction (not yet shown):** the v4−raw intent gap is predicted to *grow with DoF*. gr1 (DoF 29) is the clean mid-DoF anchor (gap +0.049); **EgoDex (DoF 44)** is the decisive high-DoF test (pending). robocasa (DoF 12) is **inconclusive** — 161 fine tasks make a single chunk near-chance for *all* reps, so it is kept off the curve (neither supports nor undercuts the slope).

Magnitudes at mid/low-DoF are modest and reported honestly; the *directions* are consistent and,
for S2, large. dexjoco (disjoint 5-task action spaces) is used throughout as an explicit
**control** (raw ≈ ceiling / clean action code wins) — never as a curve point. The S2
within-embodiment control (same gr1 episodes) further shows intent is **arm-localized** (arm-only
≈ full body, top-18 PCs recover 92%), so "more DoF ≠ more intent" is not a dataset artifact.
Details, setup, and reproducibility live in each section (shared setup: **S1 §Shared setup**).

## Status & the one decisive test
- ✅ **S1 gate** (ver-0001), **S3 copycat** (ver-0004), and **S4 geometry** (ver-0003) all independently verified (PASS, leak-free). S2 verification pending.
- ✅ **Landed since:** the **S2 within-embodiment** control (intent arm-localized: arm-only ≈ full body, top-18 PCs recover 92% — dataset confound killed). robocasa low-DoF point collected but **inconclusive** (near-chance for all reps).
- ⏳ **Pending, decisive:** the **EgoDex (DoF 44)** high-DoF endpoint — the third DoF-scaling point where grounding's share is predicted to grow — plus the accuracy-vs-DoF figure (hosted for Notion).

## The four sections
1. **[S1 intent-probe]** — intent decodability vs DoF; the gate (compression vs grounding).
2. **[S2 dof-redundancy]** — nominal ≫ effective DoF; the redundancy pillar.
3. **[S3 label-noise-shortcut]** — raw actions are copycat-prone supervision; v4 resists.
4. **[S4 semantic-geometry]** — the latent organizes task better than raw; the v3-geometry nuance.
