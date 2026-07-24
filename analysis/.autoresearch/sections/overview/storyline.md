# Overview / Abstract — storyline

## Central idea (ONE sentence)
Phase-0 offline evidence for a universal action tokenizer: as action DoF rises, task intent
becomes hard to read from raw action labels, and a **compressed-and-grounded** action latent (v4,
DINO-fused) recovers it — where the split between the two mechanisms (compression vs. grounding)
is itself the headline finding.

## Why it matters (the hook)
This is the motivation layer for the paper's proposal (pretrain a universal tokenizer on mixed
human+robot data → cheap per-embodiment finetune; use its latent as the VLA/diffusion target).
The overview must state, honestly and in one place, exactly how strong that motivation is after
Phase-0: which claims landed, how large, and what the one decisive pending test is.

## The unifying insight (lead with this)
**v3 (pure learned compression) ≈ raw on the supervision/decodability axes (S1 intent-probe, S3
copycat) but v3 > raw on semantic geometry (S4); only v4 (grounding) improves the supervision
signal.** This cleanly separates *compression* from *grounding* and threads S1/S3/S4 into one
story: compression reorganizes geometry but does not make actions more decodable or less
copyable; grounding is what improves the signal a policy actually learns from.

## Sub-claims (one per section)
1. S1 — intent decodability: v4 top at mid-DoF, mixed (~40% compression / ~60% grounding); slope pending EgoDex.
2. S2 — redundancy: raw 48.7× redundant, 22.9% task-predictive (the robust standalone pillar).
3. S3 — copycat: raw ~98% self-predictable, v3 no better, v4 least copyable; not denoising but de-correlation.
4. S4 — geometry: v4 > v3 > raw ≈ PCA; the v3 > raw geometry nuance above.

## Known risks
- The mid-DoF signals are modest; the DoF-scaling claim rests on the pending EgoDex endpoint.
  Keep the abstract honest about magnitude while leading with the (real) consistent direction.
