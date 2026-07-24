"""S2 within-embodiment intent control (exp-0005): on the SAME gr1 episodes and the
SAME leak-free LOEPTO split as exp-0001, vary ONLY the action representation fed to
the intent probe:
  (a) ARM-ONLY   : left_arm[0:7]+right_arm[7:14]  (14 dims)
  (b) FULL 29    : all dims (== exp-0001 raw baseline)
  (c) TOP-K PCs  : PCA of full (fit per-fold on train, leak-free) at k in --pca-ks
  (bonus) HANDS  : [14:26];  WAIST : [26:29]
Isolates whether intent lives in a low-eff-dim subspace and whether extra body DoF
adds or dilutes intent signal. Reuses intent_probe's split + probe.

gr1 fourier_gr1_arms_waist action order (29): left_arm[0:7] right_arm[7:14]
left_hand[14:20] right_hand[20:26] waist[26:29].

Usage:
  python intent_probe_subset.py --cache output/visual_sep_gr1/cache.npz \
     --episode-ids output/visual_sep_gr1/episode_ids.npz --out <dir>/results.json
"""
import argparse, json
from pathlib import Path
import numpy as np
from intent_probe import loepto_folds, strat_folds, eval_probe, cna
from intent_probe_pcafix import pca_probe_fold

SUBSETS = {"arm": (0, 14), "hands": (14, 26), "waist": (26, 29)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--episode-ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pca-ks", default="4,6,10,18")
    ap.add_argument("--mlp-hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    pca_ks = [int(x) for x in args.pca_ks.split(",") if x.strip()]

    d = np.load(args.cache, allow_pickle=True)
    A, y = d["A"], d["task"].astype(int)
    N, T, D = A.shape
    assert D == 29, f"expected gr1 29-dim action, got {D}"
    chance = 1.0 / len(np.unique(y))
    ep = np.load(args.episode_ids, allow_pickle=True)
    fr = ep["fold_loepto"].astype(int)
    nf = int(json.loads(str(ep["meta"]))["n_folds_loepto"])
    folds_fn = lambda: loepto_folds(fr, y, nf)

    res = {"exp_id": "exp-0005", "section": "dof-redundancy", "cache": args.cache,
           "N": int(N), "n_tasks": int(len(np.unique(y))), "chance": chance,
           "split_kind": f"loepto-{nf}fold", "action_order":
           "left_arm[0:7] right_arm[7:14] left_hand[14:20] right_hand[20:26] waist[26:29]",
           "subsets": {}, "pca_topk": {}}

    # full 29 (sanity vs exp-0001 raw) + body-part subsets
    reps = {"full29": A.reshape(N, -1).astype(np.float64)}
    for name, (a, b) in SUBSETS.items():
        reps[name] = A[:, :, a:b].reshape(N, -1).astype(np.float64)
    for name, X in reps.items():
        entry = {"dim": int(X.shape[1])}
        for probe in ("linear", "mlp"):
            r = eval_probe(X, y, folds_fn(), probe, args.seed, hidden=args.mlp_hidden)
            r["cna"] = cna(r["acc"], chance)
            entry[probe] = {"acc": round(r["acc"], 4), "cna": round(r["cna"], 4),
                            "macro_f1": round(r["macro_f1"], 4)}
        res["subsets"][name] = entry
        print(f"[s2ctrl] {name} (dim {X.shape[1]}): lin CNA={entry['linear']['cna']:.3f} "
              f"mlp CNA={entry['mlp']['cna']:.3f}", flush=True)

    # top-k PCs of full (leak-free per-fold PCA)
    raw = A.reshape(N, -1).astype(np.float64)
    maxk = max(pca_ks)
    for k in pca_ks:
        entry = {"k": k}
        for probe in ("linear", "mlp"):
            r = pca_probe_fold(raw, y, folds_fn(), min(k, raw.shape[1]), maxk, probe, args.seed)
            entry[probe] = {"acc": round(r["acc"], 4), "cna": round(cna(r["acc"], chance), 4),
                            "macro_f1": round(r["macro_f1"], 4)}
        res["pca_topk"][str(k)] = entry
        print(f"[s2ctrl] pca-{k}: lin CNA={entry['linear']['cna']:.3f} "
              f"mlp CNA={entry['mlp']['cna']:.3f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print("WROTE", args.out)
    print("#### S2 CONTROL DONE ####")


if __name__ == "__main__":
    main()
