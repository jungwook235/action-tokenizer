"""Aggregate the per-tokenizer clustering studies into one comparison + a written
recommendation answering "which clustering best classifies actions, and which
tokenizer latent best preserves action structure".

Parses analysis/output/cluster/<tag>_clustering.txt files. No GPU needed.
"""

import re
from pathlib import Path

CDIR = Path(__file__).resolve().parent / "output" / "cluster"

TAGS = [
    ("dexjoco_dual_arm", "dexjoco (V4+VAE, K64)"),
    ("gr1_1000demos", "gr1_1k (V4+VAE, K64)"),
    ("gr1_100demos_v4_vae", "gr1_100 (V4+VAE, K64)"),
    ("gr1_100demos_v4_novae", "gr1_100 (V4 noVAE, K64)"),
    ("gr1_1000demos_v3", "gr1_1k (V3, K16)"),
]


def parse(tag):
    p = CDIR / f"{tag}_clustering.txt"
    if not p.exists():
        return None
    t = p.read_text()

    def g(pat, cast=float, d=None):
        m = re.search(pat, t)
        return cast(m.group(1)) if m else d

    d = {}
    d["N"] = g(r"N samples\s*:\s*(\d+)", int)
    d["best_method"] = g(r"best silhouette: method=(\w+)", str)
    d["best_k"] = g(r"best silhouette: method=\w+ k=(\d+)", int)
    d["best_sil"] = g(r"best silhouette:.*sil=([\d.]+)\)")
    d["final_k"] = g(r"chosen action-cluster k \(KMeans\):\s*(\d+)", int)
    d["sil_action"] = g(r"INPUT action\s+([\d.\-]+)")
    d["sil_latent"] = g(r"LATENT z\s+([\d.\-]+)")
    d["sil_decoded"] = g(r"DECODED action\s+([\d.\-]+)")
    d["ari_lat_act"] = g(r"ARI\( KMeans-on-latent , euclid-clusters \) :\s*([\d.\-]+)")
    m = re.search(r"ARI\( euclid-clusters , task \)\s*:\s*([\d.\-]+)", t)
    d["ari_act_task"] = float(m.group(1)) if m else None
    # DTW block
    d["dtw_sil"] = g(r"DTW silhouette in DTW space \(subsample\) :\s*([\d.\-]+)")
    d["dtw_vs_eu"] = g(r"ARI\( DTW-clusters , euclid-clusters \)\s*:\s*([\d.\-]+)")
    # silhouette table: 'LATENT z   <eu>   <dtw>'
    m = re.search(r"LATENT z\s+([\d.\-]+)\s+([\d.\-]+)", t)
    d["sil_latent_dtw"] = float(m.group(2)) if m else None
    return d


def f(v, p=4):
    return "—" if v is None else (f"{v:d}" if isinstance(v, int) else f"{v:.{p}f}")


def main():
    cols = [(tag, label, parse(tag)) for tag, label in TAGS]
    cols = [(t, l, d) for t, l, d in cols if d]

    label_w, cw = 26, 22
    rows = [
        ("N samples", "N"),
        ("best cluster method", "best_method"),
        ("best k (by silhouette)", "best_k"),
        ("best silhouette (action)", "best_sil"),
        ("chosen k (KMeans)", "final_k"),
        ("__sep__", None),
        ("silhouette: INPUT action", "sil_action"),
        ("silhouette: LATENT z", "sil_latent"),
        ("silhouette: DECODED action", "sil_decoded"),
        ("ARI(latent vs action-clust)", "ari_lat_act"),
        ("ARI(action-clust vs task)", "ari_act_task"),
        ("__sep__", None),
        ("DTW silhouette (DTW spc)", "dtw_sil"),
        ("ARI(DTW vs euclid clust)", "dtw_vs_eu"),
        ("silhouette LATENT|DTW lbls", "sil_latent_dtw"),
    ]
    W = label_w + 1 + (cw + 1) * len(cols)
    L = ["=" * W, "CLUSTERING / EMBEDDING COMPARISON  (5 tokenizers)", "=" * W]
    L.append("Action chunks were clustered (KMeans on standardized flattened action) to define")
    L.append("classes; the same classes color the t-SNE of input action, tokenizer latent z, and")
    L.append("decoded action. Silhouette is measured in EACH space under those shared labels.")
    L.append("")
    head = f"{'metric':<{label_w}}│" + "│".join(f"{l.split(' (')[0]:^{cw}}" for _, l, _ in cols)
    L.append(head)
    L.append("─" * label_w + "┼" + "┼".join("─" * cw for _ in cols))
    for rlabel, key in rows:
        if rlabel == "__sep__":
            L.append("─" * label_w + "┼" + "┼".join("─" * cw for _ in cols))
            continue
        cells = []
        for _, _, d in cols:
            v = d.get(key)
            cells.append(f"{(v if isinstance(v,str) else f(v)):^{cw}}")
        L.append(f"{rlabel:<{label_w}}│" + "│".join(cells))
    L.append("=" * W)
    L.append("")
    L.append("column legend:")
    for tag, label, _ in cols:
        L.append(f"  {label}")
    L.append("")
    L.append("HOW TO READ")
    L.append("  • silhouette(INPUT action)  — how cluster-able the raw action chunks are.")
    L.append("  • silhouette(LATENT z)      — does the tokenizer latent keep those action")
    L.append("       classes separated? (≈ action value = preserved; ≈0 = entangled/collapsed)")
    L.append("  • silhouette(DECODED action)— the decoder should land back in action space, so")
    L.append("       this ≈ silhouette(INPUT action) for a faithful tokenizer.")
    L.append("  • ARI(latent vs action-clusters) — if you cluster the LATENT directly, do you")
    L.append("       recover the action groups? high = latent is organized by action.")
    L.append("  • ARI(action-clusters vs task)   — do unsupervised action clusters match the")
    L.append("       source task id? low = low-level motions are shared across tasks.")
    L.append("")
    L.append("=" * W)
    L.append("RECOMMENDATION — how to classify actions, and what the latents show")
    L.append("=" * W)
    L.append("1) CLUSTERING METHOD for actions:")
    L.append("   - Cluster the standardized, flattened action chunk ([T*D], z-scored per feature).")
    L.append("   - KMeans is the practical choice; GMM(diag)/Ward(agglomerative) score almost")
    L.append("     identically in the sweeps, so the extra cost is not worth it.")
    L.append("   - DTW (TimeSeriesKMeans, time-warping) was TRIED but did NOT help here: its")
    L.append("     silhouette is low (~0.04-0.08) and it diverges from the Euclidean grouping")
    L.append("     (ARI(DTW,euclid)≈0.01-0.24). Because the chunks are fixed-length T=16 windows")
    L.append("     sampled densely along trajectories, there is little phase/speed misalignment for")
    L.append("     DTW to fix, so plain Euclidean KMeans is both cheaper and cleaner. (DTW would")
    L.append("     matter more for variable-length / phase-shifted whole-episode sequences.)")
    L.append("   - Action chunks are temporally autocorrelated (overlapping windows) → t-SNE/UMAP")
    L.append("     show thread/trajectory shapes, and silhouette peaks at a MODEST k (gr1: k≈4;")
    L.append("     robocasa/dexjoco: k≈12-16). These clusters capture MOTION PRIMITIVES, not tasks.")
    L.append("   - Both t-SNE and UMAP were rendered (<tag>_tsne_/_umap_embedding.png); they give")
    L.append("     the same qualitative conclusions (UMAP separates the trajectory threads a bit")
    L.append("     more globally).")
    L.append("")
    L.append("2) ACTION CLUSTERS != TASK ID:")
    L.append("   - On gr1_unified (24 tasks) ARI(action-cluster, task) ≈ 0.05 — i.e. unsupervised")
    L.append("     action clusters do NOT recover the task, because different PnP tasks share the")
    L.append("     same low-level arm/waist motions. If you want TASK labels, use the dataset/task")
    L.append("     id directly (it is available); action geometry alone won't give it to you.")
    L.append("   - On dexjoco (5 distinct bimanual skills) the alignment is higher (ARI≈0.43).")
    L.append("")
    L.append("3) WHICH LATENT PRESERVES ACTION STRUCTURE BEST:")
    L.append("   - DECODED action recovers the input-action silhouette for ALL tokenizers → every")
    L.append("     decoder faithfully inverts the latent back to action space.")
    L.append("   - In LATENT space, V3 (action-only, K=16) preserves the action-cluster structure")
    L.append("     markedly better than V4 (DINO-fused VAE): on gr1_1k, latent silhouette 0.17 vs")
    L.append("     0.05 and ARI(latent,action) 0.57 vs 0.11. Reason: V4's latent is conditioned on")
    L.append("     the DINO visual diff (x1-x0), so it mixes in visual context and is NOT purely")
    L.append("     action-organized; V3 encodes actions only.")
    L.append("   - Among V4 variants, no-VAE keeps slightly more action structure than +VAE")
    L.append("     (gr1_100: latent silhouette 0.31 vs 0.23) — the VAE's stochastic target blurs it.")
    L.append("")
    L.append("TAKEAWAY: to *classify actions*, KMeans on the action chunk (small k) is the right")
    L.append("tool, but expect motion-primitive clusters rather than task clusters. If you need the")
    L.append("tokenizer latent itself to be action-discriminative, the action-only V3 latent is the")
    L.append("most action-organized; the DINO-fused V4 latent trades some action separability for")
    L.append("visual/inverse-dynamics information.")

    out = CDIR.parent / "ALL_clustering_comparison.txt"
    out.write_text("\n".join(L))
    print("\n".join(L))
    print(f"\n[combine] wrote -> {out}")


if __name__ == "__main__":
    main()
