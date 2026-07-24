"""S1 intent-probe: task (intent) decodability of {raw, PCA-k, v3, v4} action
representations, with a LEAK-FREE by-episode train/val split.

For each representation:
  raw = A.reshape(N, T*D)          v3 = Z3.reshape(N, T*K3)   v4 = Z4.reshape(N, T*K4)
  pca = PCA(standardized raw) at a sweep of ranks k  (the "decodability vs DoF" curve)
train a LINEAR (LogisticRegression) and a 2-LAYER MLP probe to predict the task label,
scored under leave-one-episode-per-task-out (LOEPTO) cross-validation so no episode's
chunks appear in both train and val. Metrics: accuracy, chance-normalized accuracy
CNA=(acc-chance)/(1-chance), macro-F1 (all pooled over held-out folds).

Requires the episode sidecar from recover_episode_ids.py.
If min episodes/task < 2 (LOEPTO infeasible, e.g. dexjoco 1 val ep/task) the split
falls back to stratified chunk-level with `split_leak_warning=True` recorded — used
only for the disjoint-action CONTROL where intent is trivially separable.

Usage:
  python intent_probe.py --cache output/visual_sep_gr1/cache.npz \
      --episode-ids output/visual_sep_gr1/episode_ids.npz --tag gr1 --out <dir>/results.json
"""
import argparse, json, time
from pathlib import Path
import numpy as np


def zscore_fit(X):
    mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd


def build_reps(A, Z3, Z4, pca_dims, seed):
    from sklearn.decomposition import PCA
    N = A.shape[0]
    raw = A.reshape(N, -1).astype(np.float64)
    v3 = Z3.reshape(N, -1).astype(np.float64)
    v4 = Z4.reshape(N, -1).astype(np.float64)
    mu, sd = zscore_fit(raw)
    raw_z = (raw - mu) / sd
    reps = {"raw": raw, "v3": v3, "v4": v4}
    maxk = min(max(pca_dims), raw.shape[1])
    pca_full = PCA(n_components=maxk, random_state=seed).fit(raw_z)
    scores = pca_full.transform(raw_z)  # [N, maxk]
    pca_reps = {}
    for k in pca_dims:
        kk = min(k, raw.shape[1])
        pca_reps[k] = scores[:, :kk]
    return reps, pca_reps, raw.shape[1]


def loepto_folds(fold_rank, y, n_folds):
    """Yield (train_idx, test_idx): fold f holds out each task's f-th episode."""
    idx = np.arange(len(y))
    for f in range(n_folds):
        te = idx[fold_rank == f]
        tr = idx[fold_rank != f]
        # keep only classes present in train (safety); with LOEPTO all present
        yield tr, te


def strat_folds(y, n_folds, seed):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(np.zeros(len(y)), y):
        yield tr, te


def eval_probe(X, y, folds, probe, seed, hidden=256):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score
    y = np.asarray(y)
    y_true_all, y_pred_all, fold_acc = [], [], []
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        if probe == "linear":
            clf = LogisticRegression(max_iter=3000, C=1.0, n_jobs=8)
        else:
            clf = MLPClassifier(hidden_layer_sizes=(hidden,), max_iter=300,
                                early_stopping=True, random_state=seed)
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        fold_acc.append(float((pred == y[te]).mean()))
        y_true_all.append(y[te]); y_pred_all.append(pred)
    yt = np.concatenate(y_true_all); yp = np.concatenate(y_pred_all)
    acc = float((yt == yp).mean())
    macro_f1 = float(f1_score(yt, yp, average="macro"))
    return {"acc": acc, "acc_folds_mean": float(np.mean(fold_acc)),
            "acc_folds_std": float(np.std(fold_acc)), "macro_f1": macro_f1}


def cna(acc, chance):
    return (acc - chance) / (1 - chance) if (1 - chance) > 0 else 0.0


def per_dim_best(A, y, folds_fn, chance, seed):
    """Best single ACTION CHANNEL (uses its T-trajectory) CNA under the same split."""
    N, T, D = A.shape
    best = {"channel": -1, "cna": -1e9, "acc": 0.0}
    for d in range(D):
        X = A[:, :, d].astype(np.float64)  # [N, T]
        r = eval_probe(X, y, folds_fn(), "linear", seed)
        c = cna(r["acc"], chance)
        if c > best["cna"]:
            best = {"channel": int(d), "cna": float(c), "acc": r["acc"], "macro_f1": r["macro_f1"]}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--episode-ids", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pca-dims", default="2,4,8,16,32,64,128,256,512")
    ap.add_argument("--pca-canonical", type=int, default=256)
    ap.add_argument("--mlp-hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-dim", action="store_true", help="compute best-single-channel probe (raw)")
    args = ap.parse_args()

    pca_dims = [int(x) for x in args.pca_dims.split(",") if x.strip()]
    d = np.load(args.cache, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    A, Z3, Z4, y = d["A"], d["Z3"], d["Z4"], d["task"].astype(int)
    ep = np.load(args.episode_ids, allow_pickle=True)
    fold_rank = ep["fold_loepto"].astype(int)
    epinfo = json.loads(str(ep["meta"]))
    n_folds = int(epinfo["n_folds_loepto"])

    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    chance = 1.0 / n_classes
    majority = float(counts.max() / counts.sum())

    leak = False
    if n_folds >= 2:
        folds_fn = lambda: loepto_folds(fold_rank, y, n_folds)
        split_kind = f"loepto-{n_folds}fold"
    else:
        leak = True
        n_folds = 3
        folds_fn = lambda: strat_folds(y, n_folds, args.seed)
        split_kind = "stratified-chunk-FALLBACK"

    reps, pca_reps, raw_dim = build_reps(A, Z3, Z4, pca_dims, args.seed)

    res = {"tag": args.tag, "cache": args.cache, "N": int(A.shape[0]),
           "n_tasks": n_classes, "chance": chance, "majority": majority,
           "split_kind": split_kind, "split_leak_warning": leak, "n_folds": n_folds,
           "raw_dim": raw_dim, "pca_canonical": args.pca_canonical, "seed": args.seed,
           "v3_ckpt": meta.get("v3_ckpt"), "v4_ckpt": meta.get("v4_ckpt"),
           "probes": {}, "pca_sweep": {}}

    # main reps: raw / v3 / v4 (+ canonical PCA)
    canon = min(args.pca_canonical, raw_dim)
    main_reps = {"raw": reps["raw"], "v3": reps["v3"], "v4": reps["v4"],
                 "pca": pca_reps[args.pca_canonical] if args.pca_canonical in pca_reps
                        else pca_reps[max(k for k in pca_reps)]}
    for name, X in main_reps.items():
        res["probes"][name] = {"dim": int(X.shape[1])}
        for probe in ("linear", "mlp"):
            t0 = time.time()
            r = eval_probe(X, y, folds_fn(), probe, args.seed, hidden=args.mlp_hidden)
            r["cna"] = cna(r["acc"], chance)
            r["seconds"] = round(time.time() - t0, 1)
            res["probes"][name][probe] = r
            print(f"[{args.tag}] {name}/{probe}: acc={r['acc']:.3f} CNA={r['cna']:.3f} "
                  f"F1={r['macro_f1']:.3f} ({r['seconds']}s)", flush=True)

    # PCA-rank sweep (decodability vs DoF), linear + mlp
    for k in pca_dims:
        X = pca_reps[k]
        entry = {"dim": int(X.shape[1])}
        for probe in ("linear", "mlp"):
            r = eval_probe(X, y, folds_fn(), probe, args.seed, hidden=args.mlp_hidden)
            r["cna"] = cna(r["acc"], chance)
            entry[probe] = {"acc": r["acc"], "cna": r["cna"], "macro_f1": r["macro_f1"]}
        res["pca_sweep"][str(k)] = entry
        print(f"[{args.tag}] pca-{k}: lin CNA={entry['linear']['cna']:.3f} "
              f"mlp CNA={entry['mlp']['cna']:.3f}", flush=True)

    if args.per_dim:
        t0 = time.time()
        res["per_dim_best_raw"] = per_dim_best(A.astype(np.float64), y, folds_fn, chance, args.seed)
        print(f"[{args.tag}] per-dim best raw channel: {res['per_dim_best_raw']} "
              f"({round(time.time()-t0,1)}s)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print("WROTE", args.out)
    print("#### INTENT PROBE DONE ####")


if __name__ == "__main__":
    main()
