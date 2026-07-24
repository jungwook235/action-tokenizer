"""Leak-free PCA recomputation for the S1 intent-probe (exp-0001) PCA rows.

The original intent_probe.py fit PCA on the FULL dataset before CV (a mild
unsupervised leak of held-out X into the PCA basis). raw/v3/v4 rows are unaffected
(they use per-fold StandardScaler only). This script recomputes ONLY the PCA
canonical point and the PCA-rank sweep with PCA fit INSIDE each fold on TRAIN only:

  per fold:  StandardScaler.fit(raw[train]) -> PCA.fit(scaled train) -> take top-k
             -> StandardScaler(top-k train) -> LogisticRegression / MLP

(The final per-dim StandardScaler on the PCA scores matches how raw/v3/v4 are
standardized before the probe, so all reps get identical per-dim treatment; this
is the correct choice for a probe, unlike the distance-based S4 case where it whitens.)

Emits a json with the corrected PCA numbers for merging into results_{tag}.json.
"""
import argparse, json
from pathlib import Path
import numpy as np
from intent_probe import loepto_folds, strat_folds, cna


def pca_probe_fold(raw, y, folds, k, maxk, probe, seed, hidden=256):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import f1_score
    yt_all, yp_all, fa = [], [], []
    for tr, te in folds:
        s1 = StandardScaler().fit(raw[tr])
        Rtr, Rte = s1.transform(raw[tr]), s1.transform(raw[te])
        pca = PCA(n_components=min(maxk, Rtr.shape[1]), random_state=seed).fit(Rtr)
        Ztr, Zte = pca.transform(Rtr)[:, :k], pca.transform(Rte)[:, :k]
        s2 = StandardScaler().fit(Ztr)
        Ztr, Zte = s2.transform(Ztr), s2.transform(Zte)
        if probe == "linear":
            clf = LogisticRegression(max_iter=3000, C=1.0, n_jobs=6)
        else:
            clf = MLPClassifier(hidden_layer_sizes=(hidden,), max_iter=300,
                                early_stopping=True, random_state=seed)
        clf.fit(Ztr, y[tr])
        pred = clf.predict(Zte)
        fa.append(float((pred == y[te]).mean()))
        yt_all.append(y[te]); yp_all.append(pred)
    yt, yp = np.concatenate(yt_all), np.concatenate(yp_all)
    acc = float((yt == yp).mean())
    return {"acc": acc, "macro_f1": float(f1_score(yt, yp, average="macro")),
            "acc_folds_std": float(np.std(fa))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--episode-ids", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pca-dims", default="2,4,8,16,32,64,128,256,512")
    ap.add_argument("--pca-canonical", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    pca_dims = [int(x) for x in args.pca_dims.split(",") if x.strip()]

    d = np.load(args.cache, allow_pickle=True)
    A, y = d["A"], d["task"].astype(int)
    N = A.shape[0]
    raw = A.reshape(N, -1).astype(np.float64)
    maxk = min(max(pca_dims + [args.pca_canonical]), raw.shape[1])
    chance = 1.0 / len(np.unique(y))

    ep = np.load(args.episode_ids, allow_pickle=True)
    fr = ep["fold_loepto"].astype(int)
    nf = int(json.loads(str(ep["meta"]))["n_folds_loepto"])
    if nf >= 2:
        folds_fn = lambda: loepto_folds(fr, y, nf); split = f"loepto-{nf}fold"
    else:
        nf = 3
        folds_fn = lambda: strat_folds(y, nf, args.seed); split = "stratified-chunk-FALLBACK"

    out = {"tag": args.tag, "cache": args.cache, "split_kind": split, "chance": chance,
           "pca_fit": "per-fold-train-only (leak-free)", "raw_dim": int(raw.shape[1]),
           "canonical": {}, "sweep": {}}
    for probe in ("linear", "mlp"):
        kc = min(args.pca_canonical, raw.shape[1])
        r = pca_probe_fold(raw, y, folds_fn(), kc, maxk, probe, args.seed)
        r["cna"] = cna(r["acc"], chance)
        out["canonical"][probe] = r
        print(f"[{args.tag}] PCA-{kc} {probe} (leakfree): acc={r['acc']:.3f} CNA={r['cna']:.3f}", flush=True)
    for k in pca_dims:
        kk = min(k, raw.shape[1])
        entry = {"dim": kk}
        for probe in ("linear", "mlp"):
            r = pca_probe_fold(raw, y, folds_fn(), kk, maxk, probe, args.seed)
            entry[probe] = {"acc": r["acc"], "cna": cna(r["acc"], chance), "macro_f1": r["macro_f1"]}
        out["sweep"][str(k)] = entry
        print(f"[{args.tag}] PCA-{k} (leakfree): lin CNA={entry['linear']['cna']:.3f} "
              f"mlp CNA={entry['mlp']['cna']:.3f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("WROTE", args.out)
    print("#### PCA FIX DONE ####")


if __name__ == "__main__":
    main()
