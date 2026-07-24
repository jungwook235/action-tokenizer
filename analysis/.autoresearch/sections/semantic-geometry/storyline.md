# S4 Semantic geometry — storyline

## Central idea (ONE sentence)
The intent that raw actions bury is *geometrically organized by task* in the grounded latent —
nearest-neighbour retrieval precision and clustering agreement with task labels rise sharply
from raw → PCA → v3 → v4, whereas raw-action geometry barely tracks task (ARI ≈ 0.05), which is
itself direct evidence that intent is not present in raw-action space.

## Why it matters (the hook)
S1 measures intent with a *supervised* probe; S4 shows the same conclusion holds
*unsupervised* — the latent's geometry alone clusters by task, so the recovered intent is a
real structural property, not something a probe manufactured. The prior negative (unsupervised
raw-action clusters ≠ task id, ARI ≈ 0.05 on gr1) is the motivating puzzle: raw geometry is
task-blind; the grounded latent makes it task-aware. Negative-as-motivation → positive payload.

## Sub-claims that support the idea
1. **Retrieval:** task-label precision@k of nearest neighbours climbs raw → v4 on shared-primitive
   gr1. — evidence: P@1 / P@10 table.
2. **Clustering:** NMI and ARI against task labels climb raw → v4; raw ARI ≈ 0.05 is near-random.
   — evidence: NMI/ARI table.
3. **Disjoint control:** on dexjoco (disjoint action spaces) raw already retrieves/clusters near
   ceiling — the same confound as S1, shown for consistency, not as counter-evidence.

## Expected results (pre-registration)
- gr1: P@k and NMI/ARI increase monotonically raw < PCA ≤ v3 < v4; raw ARI ≈ 0.05 (prior),
  v4 ARI ≫ raw.
- dexjoco: all representations high (ceiling) — disjoint spaces make raw geometry task-separable.
- EgoDex (if S1 collection lands): matches gr1 pattern, ideally sharper (higher DoF).

## Update — results landed (gr1+dexjoco, exp-0002)
Consistent monotone **v4 > v3 > raw ≈ PCA** on gr1 cross-ep retrieval (P@10 0.273/0.251/0.228/0.228)
and clustering (ARI 0.127/0.101/0.085/0.079) — *modest* magnitude; raw is **weak-but-nonzero**
(P@10 5.5× chance, ARI 0.085>0), not near-random. Key nuance: **v3 > raw here** (learned
compression reorganizes geometry) though v3 ≈ raw for decodability (S1) and copycat (S3); pure
linear PCA ≈ raw. dexjoco control: **v3 clusters BEST** (ARI 0.595 > v4 0.413 > raw 0.389) — a
clean action code wins when action≈task; expected mirror-image, not counter-evidence. The win is
the consistent **ordering** + the v3-geometry nuance, not strong absolute clustering.

## Favorable-first regime
gr1 `cache.npz` (raw A, Z3, Z4μ, task ids already present) — retrieval and clustering are pure
CPU on cached vectors. No new collection. dexjoco cache for the control.

## Known risks / where it could break
- Retrieval must exclude same-episode neighbours (temporal near-duplicates) or every
  representation looks artificially good — mask by episode id (the cache stores it).
- Clustering metrics depend on #clusters; fix k = #tasks and report both NMI (less
  chance-sensitive) and ARI (chance-adjusted).
- v3 ≈ v4 would weaken the "grounding" nuance but not the central idea (latent ≫ raw); report
  the v3–v4 gap honestly as the finer mechanism question shared with S1's gate.
