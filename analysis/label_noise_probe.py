"""S3 label-noise / copycat-shortcut probes on action chunks (and latent token
sequences) from the shared vsep cache. CPU-only.

Three axes (per the section title):

  (1) AUTOCORRELATION — temporal smoothness of the action sequence within a chunk.
      lag-k autocorrelation r_k averaged over action dims & samples, k=1..L.
      High r_1 => the action barely changes step-to-step (copyable).

  (2) BLIND-PREDICTABILITY (copycat) — how well the NEXT action is predicted from
      only the PREVIOUS action(s), with no observation. Two baselines:
        - naive copy  a_hat[t]=a[t-1]: normalized MSE vs the marginal action variance
          (nmse<<1 => copying is a strong shortcut).
        - linear AR(1)/AR(2): pooled OLS R^2 of a[t] from a[t-1] (and a[t-2]).
      A tokenizer / BC policy can "cheat" by copying the previous action; a high
      copycat R^2 quantifies how much of the target is trivially predictable.

  (3) SPECTRAL — temporal power spectrum of the action sequence. Fraction of power
      in the lowest frequency bins (low-freq dominance => smooth => shortcut-prone)
      and spectral flatness (0=peaky/low-rank temporal, 1=white). Also the
      covariance eigenspectrum effective rank is reported for cross-ref with S2.

Reported for RAW actions, and (optionally) for the v3/v4 latent token sequences
[N,T,K] so we can see whether the tokenizer inherits/reduces the temporal shortcut.
Runs on gr1 and dexjoco.

Usage:
  python label_noise_probe.py --cache output/visual_sep_gr1/cache.npz --tag gr1 --out <dir>/results_gr1.json
"""
import argparse, json, time
from pathlib import Path
import numpy as np


def autocorr_lags(S, max_lag):
    """S: [N, T, D]. Return mean lag-k autocorr over dims&samples for k=1..max_lag."""
    N, T, D = S.shape
    res = {}
    for k in range(1, min(max_lag, T - 1) + 1):
        rs = []
        a0 = S[:, :-k, :].reshape(-1, D)
        a1 = S[:, k:, :].reshape(-1, D)
        for d in range(D):
            x, y = a0[:, d], a1[:, d]
            if x.std() > 1e-8 and y.std() > 1e-8:
                rs.append(float(np.corrcoef(x, y)[0, 1]))
        res[f"lag{k}"] = float(np.mean(rs)) if rs else 0.0
    return res


def copycat(S):
    """S: [N,T,D]. Naive-copy nMSE and linear AR(1)/AR(2) pooled R^2."""
    N, T, D = S.shape
    prev = S[:, :-1, :].reshape(-1, D)
    nxt = S[:, 1:, :].reshape(-1, D)
    # naive copy a_hat=a[t-1]
    copy_nmse = float(((nxt - prev) ** 2).mean() / (S.reshape(-1, D).var(0).mean() + 1e-12))
    # AR(1) pooled per-dim R^2
    r1 = []
    for d in range(D):
        x, y = prev[:, d], nxt[:, d]
        if x.std() < 1e-8:
            continue
        b = np.cov(x, y)[0, 1] / np.var(x)
        pred = y.mean() + b * (x - x.mean())
        ss = ((y - pred) ** 2).sum(); tot = ((y - y.mean()) ** 2).sum()
        r1.append(1 - ss / tot if tot > 0 else 0.0)
    # AR(2) pooled per-dim R^2 (least squares on [a[t-1],a[t-2]])
    r2 = []
    if T >= 3:
        p2 = S[:, 1:-1, :].reshape(-1, D)   # a[t-1]
        p3 = S[:, :-2, :].reshape(-1, D)    # a[t-2]
        tgt = S[:, 2:, :].reshape(-1, D)
        for d in range(D):
            X = np.stack([p2[:, d], p3[:, d], np.ones_like(p2[:, d])], 1)
            y = tgt[:, d]
            if X[:, 0].std() < 1e-8:
                continue
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            pred = X @ beta
            ss = ((y - pred) ** 2).sum(); tot = ((y - y.mean()) ** 2).sum()
            r2.append(1 - ss / tot if tot > 0 else 0.0)
    return {"naive_copy_nmse": copy_nmse,
            "ar1_r2_mean": float(np.mean(r1)) if r1 else 0.0,
            "ar2_r2_mean": float(np.mean(r2)) if r2 else 0.0}


def blind_predict_ridge(S, alpha=1.0, hist=2, seed=0):
    """History-only (copycat) predictability: predict the full vector S[t] from the
    previous `hist` vectors [S[t-1],...,S[t-hist]] with multi-output Ridge, scored by
    R^2 on held-out transitions (episode boundaries respected since each chunk is one
    episode window). S: [N,T,D]. Returns R^2 (uniform-avg over dims) + per-lag copy R^2."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score
    N, T, D = S.shape
    if T <= hist:
        return {"ridge_r2": None}
    Xh, Y = [], []
    for t in range(hist, T):
        feats = [S[:, t - k, :] for k in range(1, hist + 1)]  # a[t-1..t-hist]
        Xh.append(np.concatenate(feats, axis=1))   # [N, hist*D]
        Y.append(S[:, t, :])                        # [N, D]
    Xh = np.concatenate(Xh, 0); Y = np.concatenate(Y, 0)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    r2s = []
    for tr, te in kf.split(Xh):
        m = Ridge(alpha=alpha).fit(Xh[tr], Y[tr])
        r2s.append(float(r2_score(Y[te], m.predict(Xh[te]), multioutput="uniform_average")))
    return {"ridge_r2": float(np.mean(r2s)), "ridge_r2_std": float(np.std(r2s)),
            "hist": hist, "alpha": alpha, "n_transitions": int(Xh.shape[0])}


def spectral(S):
    """S: [N,T,D]. Temporal power spectrum (rfft over T), averaged over samples&dims."""
    N, T, D = S.shape
    Sd = S - S.mean(axis=1, keepdims=True)  # remove per-chunk mean (DC)
    F = np.fft.rfft(Sd, axis=1)             # [N, T//2+1, D]
    power = (np.abs(F) ** 2).mean(axis=(0, 2))  # [T//2+1] mean power per freq bin
    power = power / (power.sum() + 1e-12)
    nlow = max(1, len(power) // 4)
    low_frac = float(power[:nlow].sum())    # fraction of AC power in lowest 25% freqs
    # spectral flatness (geo mean / arith mean) over non-DC bins
    p = power[1:] if len(power) > 1 else power
    p = p[p > 0]
    flat = float(np.exp(np.log(p).mean()) / (p.mean() + 1e-12)) if len(p) else 0.0
    return {"low_freq_power_frac": low_frac, "spectral_flatness": flat,
            "n_freq_bins": int(len(power))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-lag", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--latents", action="store_true", help="also probe v3/v4 token sequences")
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    A = d["A"].astype(np.float64)
    signals = {"raw": A}
    if args.latents:
        signals["v3"] = d["Z3"].astype(np.float64)
        signals["v4"] = d["Z4"].astype(np.float64)

    res = {"tag": args.tag, "cache": args.cache, "N": int(A.shape[0]),
           "T": int(A.shape[1]), "Dact": int(A.shape[2]),
           "v3_ckpt": meta.get("v3_ckpt"), "v4_ckpt": meta.get("v4_ckpt"), "signals": {}}
    for name, S in signals.items():
        t0 = time.time()
        entry = {"shape": list(S.shape),
                 "autocorr": autocorr_lags(S, args.max_lag),
                 "copycat": copycat(S),
                 "blind_predict": blind_predict_ridge(S, seed=args.seed),
                 "spectral": spectral(S)}
        entry["seconds"] = round(time.time() - t0, 1)
        res["signals"][name] = entry
        ac1 = entry["autocorr"]["lag1"]; cc = entry["copycat"]; bp = entry["blind_predict"]
        print(f"[{args.tag}] {name}: lag1_ac={ac1:.3f} copy_nmse={cc['naive_copy_nmse']:.3f} "
              f"ar1R2={cc['ar1_r2_mean']:.3f} ridgeR2={bp['ridge_r2']:.3f} "
              f"lowfreq={entry['spectral']['low_freq_power_frac']:.3f} "
              f"flat={entry['spectral']['spectral_flatness']:.3f} ({entry['seconds']}s)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print("WROTE", args.out)
    print("#### LABEL NOISE PROBE DONE ####")


if __name__ == "__main__":
    main()
