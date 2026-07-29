"""Intrinsic-dimension and volume-occupancy estimators.

Three families, deliberately complementary:

  (A) LINEAR / spectral  -- upper bound on intrinsic dim, threshold-free + thresholded
        participation_ratio  PR = (Sum_i lam_i)^2 / Sum_i lam_i^2
        n_pc90 / n_pc95 / n_pc99

  (B) NONLINEAR / nearest-neighbour -- no linearity assumption
        twonn       Facco et al. 2017, ratio of 1st/2nd NN distances
        mle_levina_bickel  Levina & Bickel 2004 (MacKay-Ghahramani averaging)

  (C) VOLUME occupancy -- measures manifold *thinness* directly rather than its dim
        eps_occupancy   P(random ambient point lands within eps of the data)
                        For a d-dim manifold in D-dim ambient space the eps-tube
                        volume scales as eps^(D-d), so
                            slope of log P vs log eps  ~=  D - d   (codimension)
                        which we fit from the lower tail of the ambient->data
                        nearest-neighbour distance CDF.

All estimators operate on z-scored data so they are scale-invariant and directly
comparable to PR_corr.

Caveats that the report must carry (they are real, not boilerplate):
  * NN estimators over-estimate d when the data is noisy ("shadow dimension") and
    under-estimate it when d > log(n_samples). Agreement between (A) and (B) is
    the robustness argument, not either number alone.
  * The eps-occupancy slope equals the codimension only for eps below the
    manifold's reach. In high D the smallest resolvable ambient->data distance is
    typically well above the reach, so the fitted slope is a LOWER bound on the
    codimension (equivalently an upper bound on d).
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


# --------------------------------------------------------------- preprocessing
def zscore(X, eps=1e-12):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (X - mu) / sd


def minmax(X, eps=1e-12):
    lo = X.min(axis=0, keepdims=True)
    hi = X.max(axis=0, keepdims=True)
    rng = np.where((hi - lo) < eps, 1.0, hi - lo)
    return (X - lo) / rng * 2.0 - 1.0


def nondegenerate_mask(X, rel_tol=1e-8):
    """Columns with non-negligible variance. Constant dims (e.g. robocasa
    base_motion in the tabletop subset) are excluded from the NN/occupancy
    estimators, which would otherwise be dominated by exact duplicates."""
    sd = X.std(axis=0)
    scale = sd.max() if sd.max() > 0 else 1.0
    return sd > rel_tol * scale


# ---------------------------------------------------------------- (A) spectral
def eig_descending(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    w = np.linalg.eigvalsh(cov)
    return np.clip(w[::-1], 0.0, None)


def participation_ratio(eigs):
    s1, s2 = float(eigs.sum()), float((eigs ** 2).sum())
    return (s1 * s1) / s2 if s2 > 0 else 0.0


def n_pc_for(eigs, frac):
    total = eigs.sum()
    if total <= 0:
        return 0
    return int(np.searchsorted(np.cumsum(eigs) / total, frac) + 1)


def spectral_metrics(X, keep_spectrum=True):
    """PR + n_pc{90,95,99} in both corr (z-scored) and minmax space."""
    D = X.shape[1]
    out = {"nominal_dim": int(D), "n_samples": int(X.shape[0])}
    for tag, Xt in (("corr", zscore(X)), ("minmax", minmax(X))):
        eigs = eig_descending(Xt)
        pr = participation_ratio(eigs)
        out[f"PR_{tag}"] = float(pr)
        out[f"redundancy_{tag}"] = float(D / pr) if pr > 0 else float("inf")
        for f in (90, 95, 99):
            out[f"n_pc{f}_{tag}"] = n_pc_for(eigs, f / 100.0)
        if keep_spectrum and tag == "corr":
            tot = eigs.sum()
            out["var_explained_corr"] = ((np.cumsum(eigs) / tot).tolist()
                                         if tot > 0 else [])
    return out


# ------------------------------------------------------------- (B) NN-based ID
def _knn_distances(X, k, n_jobs=-1):
    """Distances to the k nearest *other* points. Returns [n, k]."""
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=n_jobs)
    nn.fit(X)
    d, _ = nn.kneighbors(X, return_distance=True)
    return d[:, 1:]  # drop self


def twonn(X, discard_frac=0.10, n_jobs=-1):
    """Facco et al. 2017 two-NN intrinsic dimension.

    mu_i = r2_i / r1_i follows a Pareto(d) under a locally-uniform density, so
        -log(1 - F(mu))  =  d * log(mu)
    and d is the slope of a through-origin fit. The largest `discard_frac` of mu
    is dropped (standard) because the tail is where local uniformity fails.
    """
    n = X.shape[0]
    if X.shape[1] < 2 or n < 50:
        return {"id": float("nan"), "n_used": 0, "note": "too few dims/samples"}
    d2 = _knn_distances(X, k=2, n_jobs=n_jobs)
    r1, r2 = d2[:, 0], d2[:, 1]
    ok = (r1 > 0) & np.isfinite(r1) & np.isfinite(r2)
    n_dup = int((~ok).sum())
    r1, r2 = r1[ok], r2[ok]
    mu = r2 / r1
    mu = mu[mu > 1.0 + 1e-12]
    if mu.size < 50:
        return {"id": float("nan"), "n_used": int(mu.size),
                "n_zero_r1": n_dup, "note": "degenerate (duplicates)"}
    mu.sort()
    m = mu.size
    F = np.arange(1, m + 1) / (m + 1.0)
    keep = int(np.floor(m * (1.0 - discard_frac)))
    x = np.log(mu[:keep])
    y = -np.log(1.0 - F[:keep])
    d_hat = float((x @ y) / (x @ x))
    ss_res = float(((y - d_hat * x) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"id": d_hat, "n_used": int(keep), "n_zero_r1": n_dup,
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")}


def mle_levina_bickel(X, k1=10, k2=20, n_jobs=-1):
    """Levina-Bickel (2004) MLE with MacKay-Ghahramani inverse averaging.

    Reported as a lower-variance cross-check on TwoNN; not a primary number.
    """
    n = X.shape[0]
    if X.shape[1] < 2 or n < k2 + 10:
        return {"id": float("nan"), "note": "too few dims/samples"}
    d = _knn_distances(X, k=k2, n_jobs=n_jobs)
    d = np.where(d <= 0, np.nan, d)
    logd = np.log(d)
    ids = []
    for k in range(k1, k2 + 1):
        # m_k(i)^-1 = mean_{j<k} log( T_k / T_j )
        inv = (logd[:, k - 1][:, None] - logd[:, : k - 1]).mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            mk = 1.0 / inv
        good = np.isfinite(mk) & (mk > 0)
        if good.sum() < 10:
            continue
        ids.append(1.0 / np.mean(1.0 / mk[good]))  # MacKay averaging
    if not ids:
        return {"id": float("nan"), "note": "degenerate"}
    return {"id": float(np.mean(ids)), "id_per_k": [float(v) for v in ids],
            "k1": k1, "k2": k2}


def nn_id_bootstrap(X, n_sub=10000, n_boot=5, seed=0, with_mle=True, n_jobs=-1):
    """TwoNN (+MLE) on `n_boot` independent subsamples of size `n_sub`."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    n_sub = min(n_sub, n)
    tw, ml = [], []
    detail = []
    for b in range(n_boot):
        idx = rng.choice(n, size=n_sub, replace=False) if n_sub < n else np.arange(n)
        Xs = np.ascontiguousarray(X[idx])
        t = twonn(Xs, n_jobs=n_jobs)
        tw.append(t["id"])
        rec = {"twonn": t}
        if with_mle:
            m = mle_levina_bickel(Xs, n_jobs=n_jobs)
            ml.append(m["id"])
            rec["mle"] = {kk: vv for kk, vv in m.items() if kk != "id_per_k"}
        detail.append(rec)
    tw = np.asarray(tw, dtype=float)
    out = {"n_sub": int(n_sub), "n_boot": int(n_boot),
           "twonn_mean": float(np.nanmean(tw)), "twonn_std": float(np.nanstd(tw)),
           "twonn_runs": [float(v) for v in tw], "detail": detail}
    if with_mle:
        ml = np.asarray(ml, dtype=float)
        out["mle_mean"] = float(np.nanmean(ml))
        out["mle_std"] = float(np.nanstd(ml))
    return out


# ------------------------------------------------------- (C) volume occupancy
def correlation_dimension(X, n_sub=5000, c_lo=1e-3, c_hi=1e-1, seed=0,
                          c_mid=(0.05, 0.5)):
    """Grassberger-Procaccia correlation dimension.

    C(r) = P(||x_i - x_j|| < r) ~ r^d in the scaling region, so d is the log-log
    slope of the pair-distance CDF. Unlike the ambient-occupancy slope this stays
    measurable at every D (it only uses data-data distances), which makes it the
    workhorse for turning "how thin is the manifold" into a codimension D - d.

    Robot actions are NOT scale-free: at short range the point cloud looks like
    the 1-D trajectory curve it was sampled from, and only at longer range does
    the across-episode manifold appear. We therefore report the slope in two
    windows and the full local-slope curve, instead of pretending one plateau
    exists:
        id      C in [c_lo, c_hi]  -- short range (local / trajectory scale)
        id_mid  C in c_mid         -- mid range (cross-episode manifold scale)
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    m = min(n_sub, n)
    idx = rng.choice(n, size=m, replace=False) if m < n else np.arange(n)
    Y = np.ascontiguousarray(X[idx])
    # pairwise upper-triangle distances
    sq = (Y * Y).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (Y @ Y.T)
    iu = np.triu_indices(m, k=1)
    d = np.sqrt(np.clip(d2[iu], 0.0, None))
    d = d[np.isfinite(d)]
    P_all = d.size
    d = np.sort(d[d > 0])
    P = d.size
    dup_frac = (P_all - P) / P_all if P_all else 1.0
    if P < 1000:
        return {"id": float("nan"), "note": "too few non-degenerate pairs",
                "dup_pair_frac": float(dup_frac)}
    C = np.arange(1, P + 1) / P

    def _fit(lo, hi):
        a, b_ = int(np.ceil(lo * P)), int(np.floor(hi * P))
        a, b_ = max(a, 10), min(max(b_, a + 50), P)
        xx, yy = np.log(d[a - 1:b_]), np.log(C[a - 1:b_])
        m_ = np.isfinite(xx) & np.isfinite(yy)
        xx, yy = xx[m_], yy[m_]
        if xx.size < 20 or np.ptp(xx) < 1e-9:
            return float("nan"), float("nan"), [float("nan")] * 2
        A_ = np.vstack([xx, np.ones_like(xx)]).T
        s_, i_ = np.linalg.lstsq(A_, yy, rcond=None)[0]
        sst = float(((yy - yy.mean()) ** 2).sum())
        r2_ = (1.0 - float(((yy - (s_ * xx + i_)) ** 2).sum()) / sst
               if sst > 0 else float("nan"))
        return float(s_), float(r2_), [float(d[a - 1]), float(d[b_ - 1])]

    slope, r2, r_rng = _fit(c_lo, c_hi)
    slope_mid, r2_mid, r_rng_mid = _fit(*c_mid)
    if not np.isfinite(slope):
        return {"id": float("nan"), "note": "degenerate scaling region",
                "dup_pair_frac": float(dup_frac)}
    k_lo, k_hi = max(int(np.ceil(c_lo * P)), 10), min(int(np.floor(c_hi * P)), P)
    # local slope curve on a decimated log grid (diagnostic for the plateau)
    ks = np.unique(np.geomspace(10, P, 60).astype(int))
    lr, lc = np.log(d[ks - 1]), np.log(C[ks - 1])
    keep = np.concatenate([[True], np.diff(lr) > 1e-9])
    lr, lc, ks = lr[keep], lc[keep], ks[keep]
    loc = np.gradient(lc, lr) if lr.size >= 3 else np.full(lr.size, np.nan)
    out = {"id": float(slope), "r2": float(r2),
           "id_mid": float(slope_mid), "r2_mid": float(r2_mid),
           "n_sub": int(m), "n_pairs": int(P), "dup_pair_frac": float(dup_frac),
           "fit_C_range": [c_lo, c_hi], "fit_r_range": r_rng,
           "fit_C_range_mid": list(c_mid), "fit_r_range_mid": r_rng_mid,
           "local_slope_r": [float(v) for v in d[ks - 1]],
           "local_slope": [float(v) for v in loc]}
    if dup_frac > 0.05:
        out["warning"] = (f"{dup_frac:.1%} of pairs are exact duplicates "
                          "(discrete/binary dims) -- corr-dim unreliable")
    return out


def eps_occupancy(X, n_ambient=100000, n_ref=20000, ambient="gauss",
                  seed=0, tail_lo=1e-4, tail_hi=5e-2, n_jobs=-1):
    """Fraction of random ambient points falling within eps of the data manifold.

    X is z-scored internally, so the ambient reference has matched per-dim scale
    and only the *correlation structure* (i.e. the manifold) distinguishes the
    data cloud from the ambient cloud.

      ambient='gauss'   : N(0, I_D)  -- same marginal variance, no correlations
      ambient='uniform' : U[min, max] per dim of the z-scored data

    Two distinct scaling regimes exist and conflating them is the easy mistake:

      eps << r_med  the eps-balls around the n_ref sampled points are disjoint,
                    so P ~ n_ref * eps^D  and the slope recovers D, NOT the
                    codimension. Reported as `tail_slope`, purely a diagnostic
                    (it should come out near D_effective).
      eps >~ r_med  the balls merge into a tube around the manifold, volume
                    ~ eps^(D-d), so the slope is the codimension. Reported as
                    `tube_slope`, fitted over eps in [1, 8] * r_med.

    In high D almost no uniform ambient sample lands within 8*r_med of the data,
    so `tube_slope` is often unmeasurable -- that censoring IS the volume-collapse
    result, and it is reported explicitly rather than silently fitted.

    NULL CALIBRATION (the number to actually quote). Absolute occupancy vanishes
    for *any* finite point set once D is large -- that is the curse of
    dimensionality, not manifold thinness -- and r_med itself shrinks with the
    sample size, so neither is comparable across D. We therefore also build a
    matched null cloud by permuting each column of the data independently: same
    marginals, same sample count, cross-dimension structure destroyed, i.e. the
    "no manifold" version of this exact dataset. Choosing eps so that the NULL is
    hit with probability p, the real cloud's occupancy at that same eps is a
    direct, D-comparable measure of how much less volume the manifold occupies:

        occ_real_at_null_p  <<  p   =>  real actions occupy far less space
        volume_deficit_log10 = log10(occ_real_at_null_p / p)
    """
    Xz = np.ascontiguousarray(zscore(X))
    n, D = Xz.shape
    rng = np.random.default_rng(seed)

    ref_idx = rng.choice(n, size=min(n_ref, n), replace=False)
    ref = np.ascontiguousarray(Xz[ref_idx])

    # matched "no manifold" null: independent per-dim permutation of the SAME rows
    null = np.empty_like(ref)
    for j in range(D):
        null[:, j] = ref[rng.permutation(ref.shape[0]), j]

    # data's own scale: median NN distance among reference points
    dd = _knn_distances(ref, k=1, n_jobs=n_jobs)[:, 0]
    dd = dd[np.isfinite(dd) & (dd > 0)]
    r_med = float(np.median(dd)) if dd.size else float("nan")

    if ambient == "gauss":
        Q = rng.standard_normal(size=(n_ambient, D))
    elif ambient == "uniform":
        lo, hi = Xz.min(axis=0), Xz.max(axis=0)
        Q = rng.uniform(lo, hi, size=(n_ambient, D))
    else:
        raise ValueError(ambient)

    nn = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=n_jobs).fit(ref)
    dq, _ = nn.kneighbors(Q, return_distance=True)
    dq = np.sort(dq[:, 0])

    nn0 = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=n_jobs).fit(null)
    dq0, _ = nn0.kneighbors(Q, return_distance=True)
    dq0 = np.sort(dq0[:, 0])

    M = dq.size
    qs = [1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    quantiles = {f"q{q:g}": float(dq[min(M - 1, max(0, int(np.ceil(q * M)) - 1))])
                 for q in qs}

    occ = {}
    for mult in (0.5, 1.0, 2.0, 4.0, 8.0):
        if np.isfinite(r_med):
            c = int((dq < mult * r_med).sum())
            occ[f"occ_at_{mult:g}x_rmed"] = c / M
            occ[f"count_at_{mult:g}x_rmed"] = c
    occ["occ_resolution"] = 1.0 / M  # smallest non-zero occupancy resolvable

    # ---- null-calibrated occupancy: eps set so the null is hit w.p. p
    nullcal = {}
    for p in (0.1, 0.01, 0.001):
        kp = max(1, int(np.ceil(p * M)))
        if kp > M:
            continue
        eps_p = float(dq0[kp - 1])
        c = int((dq < eps_p).sum())
        occ_real = c / M
        nullcal[f"p{p:g}"] = {
            "eps": eps_p, "eps_over_sqrtD": eps_p / np.sqrt(D),
            "occ_real": occ_real, "count_real": c, "occ_null": kp / M,
            "ratio_real_over_null": (occ_real / (kp / M)) if kp else float("nan"),
            "volume_deficit_log10": (float(np.log10(occ_real / (kp / M)))
                                     if c > 0 else float("nan")),
            "censored": c == 0,
            "volume_deficit_log10_bound": float(np.log10((1.0 / M) / (kp / M))),
        }

    def _loglog_slope(lo_eps, hi_eps, min_pts=50):
        sel = (dq >= lo_eps) & (dq <= hi_eps) & (dq > 0)
        k = np.nonzero(sel)[0] + 1  # rank -> CDF
        if k.size < min_pts:
            return {"slope": float("nan"), "r2": float("nan"), "n_pts": int(k.size),
                    "eps_range": [float(lo_eps), float(hi_eps)],
                    "note": "censored: too few ambient samples in this eps window"}
        x, y = np.log(dq[k - 1]), np.log(k / M)
        good = np.isfinite(x) & np.isfinite(y)
        x, y = x[good], y[good]
        if x.size < min_pts or np.ptp(x) < 1e-9:
            return {"slope": float("nan"), "r2": float("nan"), "n_pts": int(x.size),
                    "eps_range": [float(lo_eps), float(hi_eps)], "note": "degenerate"}
        A = np.vstack([x, np.ones_like(x)]).T
        s, b = np.linalg.lstsq(A, y, rcond=None)[0]
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = (1.0 - float(((y - (s * x + b)) ** 2).sum()) / ss_tot
              if ss_tot > 0 else float("nan"))
        return {"slope": float(s), "r2": float(r2), "n_pts": int(x.size),
                "eps_range": [float(lo_eps), float(hi_eps)]}

    # regime 1: eps >~ r_med -> merged tube, slope = codimension (the target)
    tube = (_loglog_slope(1.0 * r_med, 8.0 * r_med) if np.isfinite(r_med)
            else {"slope": float("nan"), "note": "no r_med"})
    # regime 2: far lower tail -> disjoint balls, slope ~ D (sanity diagnostic)
    k_lo = max(5, int(np.ceil(tail_lo * M)))
    k_hi = min(M, max(k_lo + 10, int(np.floor(tail_hi * M))))
    tail = _loglog_slope(dq[k_lo - 1], dq[k_hi - 1], min_pts=10)

    return {
        "ambient": ambient, "D": int(D), "n_ambient": int(M),
        "n_ref": int(ref.shape[0]), "r_med_data_nn": r_med,
        "ambient_nn_quantiles": quantiles,
        "null_calibrated": nullcal,
        "median_ambient_nn_real": float(np.median(dq)),
        "median_ambient_nn_null": float(np.median(dq0)),
        **occ,
        # codimension (target): fitted only where the tube regime is populated
        "tube_slope": tube.get("slope", float("nan")),
        "tube_slope_r2": tube.get("r2", float("nan")),
        "tube_fit": tube,
        "d_hat_from_tube": (float(D - tube["slope"])
                            if np.isfinite(tube.get("slope", np.nan))
                            else float("nan")),
        # diagnostic: should land near D_effective if the estimator is healthy
        "tail_slope": tail.get("slope", float("nan")),
        "tail_slope_r2": tail.get("r2", float("nan")),
        "tail_fit": tail,
        # decimated curves for plotting: (eps, CDF) for real and null clouds
        "curve_eps": [float(v) for v in dq[np.unique(
            np.geomspace(1, M, 200).astype(int) - 1)]],
        "curve_eps_null": [float(v) for v in dq0[np.unique(
            np.geomspace(1, M, 200).astype(int) - 1)]],
        "curve_cdf": [float(v) for v in (np.unique(
            np.geomspace(1, M, 200).astype(int)) / M)],
    }
