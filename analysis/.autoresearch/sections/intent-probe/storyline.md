# S1 Intent decodability vs DoF — storyline

## Central idea (ONE sentence)
As embodiment action DoF grows, the *task/intent* behind a motion becomes progressively
unreadable from the raw action chunk, but a learned, visually-grounded action latent (v4,
DINO-fused μ) keeps intent linearly decodable — and the size of the v4-over-PCA margin tells
us how much of that recovery needs visual grounding versus mere compression.

## Why it matters (the hook)
The whole paper argues for a *universal action tokenizer* whose latent is a better VLA/
diffusion target than raw actions. That only pays off if raw actions are actually a poor
carrier of intent where it matters most — high-DoF, dexterous embodiments. Phase 0 must show,
offline and cheaply, that (a) intent leaks out of raw actions as DoF rises and (b) the learned
latent puts it back. This section is the load-bearing motivation; the other three explain
*why* it happens (redundancy, label noise, geometry).

## Sub-claims that support the idea
1. **Raw intent-decodability falls with DoF.** Chance-normalized task-label accuracy from a
   raw action chunk drops from low-DoF (robocasa) to high-DoF, many-task (EgoDex hands).
   — evidence: hero accuracy-vs-DoF figure, raw line.
2. **The v4 latent stays high and roughly flat across DoF.** — evidence: same figure, v4 line;
   main table v4 column ≥ every other representation at every DoF.
3. **The v4−PCA margin isolates grounding from compression.** PCA at matched dim recovers part
   of the intent (compression/denoising); the *residual* v4 advantage is the visual-grounding
   contribution. This margin is the storyline gate. — evidence: figure PCA vs v4 lines + gating verdict.
4. **The trend is not a linear-probe or #tasks artifact.** MLP probe preserves the ordering;
   macro-F1 mirrors accuracy; per-dim (best single subspace) stays low for raw. — evidence:
   robustness table + per-dim mini-table.

## Expected results (pre-registration)
- Hero figure: raw declines with DoF; v4 stays high; PCA between; v3 near PCA (action-only
  learned compression, no vision).
- v4 is the top representation at every DoF; the v4−raw gap widens with DoF.
- **DoF curve anchored on shared-primitive datasets only:** gr1 (clean anchor, runnable now),
  robocasa (low-DoF endpoint), EgoDex (high-DoF endpoint). These have tasks that *share* action
  primitives, so raw actions are genuinely intent-ambiguous — the regime the hypothesis is about.
- **dexjoco-dual is a deliberate CONTROL, off the curve:** 44-DoF but only ~5 tasks with
  *disjoint* action spaces → task is trivially decodable from raw (≈ceiling), so raw≈v4 there.
  That is *expected and supportive* (disjoint spaces are the one regime raw already carries
  intent), reported separately to demonstrate the confound the curve avoids — a credibility win,
  not a counter-example. Chance-normalized acc + macro-F1 further blunt the #tasks confound.

## Update — results landed (gr1 mid-DoF, exp-0001, leak-free LOEPTO)
Honest outcome: **v4 is consistently top but modest at mid-DoF** — CNA raw 0.298 / v3 0.301 /
v4 0.347 (linear); v3≈raw (compression alone adds nothing), full-budget PCA (0.245) *below* raw,
PCA peaks at k=32 (0.319). Absolute decodability is low (chance 1/24) — the premise holds.
Gate claim independently verified (ver-0001 PASS). So the
narrative shifts: (1) the mid-DoF gap is *small-but-consistent*, not dramatic; (2) the headline
becomes the **slope across DoF** — the v4−raw gap must *widen* at the EgoDex high-DoF endpoint
(pending) for the DoF-scaling claim; (3) the robust *landed* pillar is redundancy (S2: raw gr1
48.7× redundant, 22.9% task-predictive) + consistent v4>v3>raw geometry (S4). dexjoco control at
ceiling (raw=v3=v4≈1.0) as designed.
**robocasa (exp-0006) landed but INCONCLUSIVE:** 161 fine pick-place tasks make a single chunk
near-chance for *all* reps (raw 0.021, v4 0.039, chance 0.006); the +0.018 gap is within noise — a
task-granularity artifact, kept OFF the DoF curve. The slope therefore rests on gr1 (clean mid) →
EgoDex (clean high, pending), NOT on robocasa.

## Favorable-first regime
gr1 (29-DoF, 24 tasks) with the existing `output/visual_sep_gr1/cache.npz` (A, Z3, Z4μ already
encoded, N=4008) — linear probe, episode-level split. It needs no new GPU collection and has
enough tasks for a stable chance level. Validate the raw<PCA<v4 ordering here first, then widen
to robocasa / dexjoco / EgoDex.

## Known risks / where it could break
- **Episode-level split is mandatory** — chunk-level split leaks (adjacent chunks share frames
  and near-identical actions) and would inflate every representation, killing the contrast.
- **#tasks differs 5→~194 across datasets** — always report chance-normalized acc + macro-F1;
  lean on within-embodiment control (S2) for the confound-free DoF statement.
- **v3 unavailable for EgoDex** (only the multi-embodiment v4 covers EgoDex) — v3 line is
  robocasa/gr1/dexjoco only; state N/A rather than fabricate.
- If PCA ≈ v4 everywhere, the story shifts from "grounding recovers intent" to
  "compression/denoising recovers intent" — still publishable motivation; the gate decides.
