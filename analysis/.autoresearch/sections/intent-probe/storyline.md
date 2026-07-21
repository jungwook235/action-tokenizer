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
- **Pre-registered caveat (honest):** dexjoco-dual (44-DoF, only ~5 near-disjoint tasks) will
  show *high* raw decodability — few disjoint tasks make raw actions trivially separable
  despite high DoF. This breaks a naive monotone raw-vs-nominal-DoF line, so the clean DoF
  signal rests on (i) EgoDex (high DoF *and* many tasks) and (ii) the within-embodiment DoF
  control in S2, where #tasks is held fixed. We report chance-normalized accuracy + macro-F1
  precisely to blunt the #tasks confound, and flag dexjoco explicitly rather than hide it.

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
