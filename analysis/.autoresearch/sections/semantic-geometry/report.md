# S4 · Semantic geometry — the latent organizes task better than raw actions

## TL;DR
- **Main result (gr1, leak-free cross-episode):** the grounded latent has the strongest task geometry — P@10 **v4 0.273 > v3 0.251 > raw 0.228 ≈ PCA 0.228** and **ARI v4 0.127 > v3 0.101 > raw 0.085 ≈ PCA 0.079** — a consistent monotone **v4 > v3 > raw ≈ PCA**.
- **Interesting nuance:** v3 **>** raw here (learned compression reorganizes geometry) *even though* v3 ≈ raw for linear decodability (S1) and copycat-predictability (S3); pure linear PCA ≈ raw (adds no geometry). Grounding (v4) sharpens further.
- **Honest magnitude:** absolute ARI is low even for v4 (~0.13) — the win is the *consistent ordering*, not strong clustering; the grounded latent recovers *relatively* the most task structure.

---

Retrieval = task-label precision@k of a chunk's k nearest neighbours; we report **cross-episode**
P@k (same-episode neighbours excluded — temporal adjacency otherwise inflates naive P@1 to ~0.95).
Clustering = KMeans (k=#tasks), scored by NMI and chance-adjusted ARI vs task labels. Reuses
`output/visual_sep_gr1/cache.npz` + dexjoco cache; raw / PCA-256 / v3 / v4-μ, CPU. See **S1 §Shared setup**.

## gr1 (shared primitives — the story)

Chance P@k (by task) ≈ 0.042.

| representation | P@1 (cross-ep) | P@10 (cross-ep) | NMI | ARI |
|---|---|---|---|---|
| raw action | 0.271 | 0.228 | 0.328 | 0.085 |
| PCA-256 | 0.271 | 0.228 | 0.309 | 0.079 |
| v3 latent | 0.293 | 0.251 | 0.335 | 0.101 |
| **v4 latent** | **0.305** | **0.273** | **0.384** | **0.127** |

**Reading:** the ordering is **v4 > v3 > raw ≈ PCA** on all four metrics. **PCA-256 ≈ raw**
(P@10 0.228 = raw; ARI 0.079 ≈ raw 0.085) — a pure linear truncation of the raw actions neither
adds nor removes task geometry. But **v3 is distinctly above raw** (P@10 0.251 > 0.228; ARI 0.101
> 0.085): the *learned* action-only compression reorganizes the geometry to be more
task-structured. That is a real, interesting contrast with S1/S3, where v3 ≈ raw for linear
intent-decodability and for copycat-predictability — learned compression helps *geometry* even
where it does not help *decodability or supervision*. v4's grounding then sharpens further
(ARI 0.127, P@10 0.273). Raw itself has **weak-but-nonzero** task geometry (P@10 0.228 = 5.5× the
0.042 chance; ARI 0.085 > 0), just markedly less than the latents — the honest claim is "raw has
weak structure; learned latents (esp. v4) sharpen it", and the result is the *consistent monotone
ordering*, not strong absolute clustering.

<details>
<summary><b>Reproducibility — gr1 retrieval & clustering (exp-0002)</b></summary>

- **Script:** `analysis/semantic_geometry.py` (kNN retrieval with same-episode mask → cross-ep P@k; KMeans k=#tasks; NMI/ARI). Reads `output/visual_sep_gr1/cache.npz`.
- **Seed:** 0 · branch `master`, commit `f6e29c1` (dirty) · Raw: `.autoresearch/results/exp-0002/results.json`.
- Note: naive (same-episode-included) P@1 reaches ~0.95 for all reps — reported cross-ep to remove the temporal-adjacency confound.
- ✅ Core S4 claim independently verified — **ver-0003 PASS** (results reproduced bit-for-bit).
</details>

## dexjoco (disjoint-action control)

Cross-episode P@k is **degenerate/N/A** here (1 val episode per task ⇒ no cross-episode same-task
pairs); naive P@k is at ceiling (1.0) for all reps. Clustering still discriminates:

| representation | naive P@10 | NMI | ARI |
|---|---|---|---|
| raw action | 1.00 | 0.532 | 0.389 |
| PCA-256 | 1.00 | 0.532 | 0.389 |
| **v3 latent** | 1.00 | **0.708** | **0.595** |
| v4 latent | 1.00 | 0.546 | 0.413 |

> On disjoint-action dexjoco the **action-only v3 clusters best** (ARI 0.595 > v4 0.413 > raw 0.389):
> when action ≈ task, a clean *action* code separates tasks best, and v4's added visual axes
> slightly blur the pure-action partition — matching the prior vsep finding. This is the expected
> control behaviour and the mirror image of the shared-primitive regime, where v4's grounding is
> what helps. It is *not* a counter-example to the paper's claim.

<details>
<summary><b>Reproducibility — dexjoco control (exp-0002)</b></summary>

- Same script, dexjoco cache. Raw: `.autoresearch/results/exp-0002/results.json` (`dexjoco`).
</details>

## EgoDex (pending S1 collection)

Same tables to be added if the EgoDex cache lands (S1); the hypothesis-consistent outcome is the
gr1 ordering, ideally sharper at higher DoF: raw ARI {{PH:s4-ego-raw-ari | ARI, EgoDex raw | expect=~low, near raw gr1}} → v4 ARI {{PH:s4-ego-v4-ari | ARI, EgoDex v4 | expect=> v3 > raw, widening}}.

## Takeaway
Unsupervised geometry echoes the supervised probe: on realistic shared-primitive gr1 the grounded
latent recovers the most task structure (v4 > v3 > raw ≈ PCA on retrieval and clustering), while
raw actions have only **weak-but-nonzero** task geometry (ARI 0.085 > 0). Magnitudes are modest —
the result is a *consistent monotone ordering*, not strong clustering — and the dexjoco control
confirms the metric is honest (a clean action code wins precisely when action ≈ task). Uniquely
among the sections, **v3 > raw here** (learned compression reorganizes geometry) even though
v3 ≈ raw for decodability (S1) and copycat (S3); grounding (v4) adds the rest — the geometric
strand of the motivation.
