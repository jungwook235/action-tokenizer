"""S3 part (a): SPECTRAL / JERK denoising test.

Decode the cached tokenizer latents back to actions (a_hat) on CPU and compare the
high-frequency energy and jerk of RAW actions vs the tokenizer RECONSTRUCTION.
Hypothesis: the reconstruction is temporally smoother (less HF power, lower jerk)
= evidence the tokenizer denoises the (label-noisy) action supervision.

Metrics per action dim, averaged:
  HF_frac   = fraction of temporal AC power in the upper half of the rfft spectrum
              (per chunk, DC removed), mean over dims & samples
  jerk_ms   = mean squared 2nd temporal difference a[t+1]-2a[t]+a[t-1]

Decodes v4 (Z4 mu -> a_hat) [primary] and v3 (Z3 -> a_hat) [if the decoder accepts
empty global/hand tokens, Ng=Nh=0]. CPU-only, no DINO needed for decode.

Usage:
  python label_noise_recon.py --cache output/visual_sep_gr1/cache.npz --tag gr1 --out <dir>/recon_gr1.json
"""
import sys, argparse, json, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))


def hf_frac(S):
    """S: [N,T,D]. Fraction of AC temporal power in the upper-half freq bins."""
    Sd = S - S.mean(axis=1, keepdims=True)
    F = np.fft.rfft(Sd, axis=1)
    power = (np.abs(F) ** 2)                     # [N, nf, D]
    nf = power.shape[1]
    ac = power[:, 1:, :]                          # drop DC
    if ac.shape[1] == 0:
        return 0.0
    hi = ac[:, ac.shape[1] // 2:, :].sum(axis=1)  # upper half of AC bins
    tot = ac.sum(axis=1) + 1e-12
    return float((hi / tot).mean())


def jerk_ms(S):
    """S: [N,T,D]. Mean squared 2nd temporal difference."""
    if S.shape[1] < 3:
        return 0.0
    j = S[:, 2:, :] - 2 * S[:, 1:-1, :] + S[:, :-2, :]
    return float((j ** 2).mean())


def decode_all(tok, Z, batch=256):
    import torch
    dtype = tok.encoder.action_proj.weight.dtype
    outs = []
    with torch.no_grad():
        for i in range(0, Z.shape[0], batch):
            z = torch.from_numpy(Z[i:i + batch]).to(dtype)
            zero_g = z[:, :0]
            a = tok.decode(zero_g, z, zero_g)
            outs.append(a.float().numpy())
    return np.concatenate(outs, 0)


def main():
    import torch
    torch.set_num_threads(6)
    import gr00t.experiment.data_config_v3  # noqa
    from gr00t.model.action_latent_tokenizer_wrapper import ActionLatentTokenizerWrapper

    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    A = d["A"].astype(np.float64)
    Z3, Z4 = d["Z3"], d["Z4"]

    res = {"tag": args.tag, "cache": args.cache, "N": int(A.shape[0]),
           "v3_ckpt": meta.get("v3_ckpt"), "v4_ckpt": meta.get("v4_ckpt"),
           "raw": {"HF_frac": hf_frac(A), "jerk_ms": jerk_ms(A)}, "recon": {}}
    print(f"[{args.tag}] raw: HF_frac={res['raw']['HF_frac']:.4f} jerk_ms={res['raw']['jerk_ms']:.5f}", flush=True)

    for name, ckpt, Z in (("v4", meta["v4_ckpt"], Z4), ("v3", meta["v3_ckpt"], Z3)):
        try:
            t0 = time.time()
            wrap = ActionLatentTokenizerWrapper.from_checkpoint(ckpt, device="cpu")
            wrap.eval()
            a_hat = decode_all(wrap.tokenizer, Z).astype(np.float64)
            l1 = float(np.abs(a_hat - A).mean())
            entry = {"HF_frac": hf_frac(a_hat), "jerk_ms": jerk_ms(a_hat),
                     "decode_l1_vs_gt": l1, "seconds": round(time.time() - t0, 1),
                     "HF_frac_ratio_recon_over_raw": hf_frac(a_hat) / (res["raw"]["HF_frac"] + 1e-12),
                     "jerk_ratio_recon_over_raw": jerk_ms(a_hat) / (res["raw"]["jerk_ms"] + 1e-12)}
            res["recon"][name] = entry
            print(f"[{args.tag}] {name} recon: HF_frac={entry['HF_frac']:.4f} "
                  f"(x{entry['HF_frac_ratio_recon_over_raw']:.2f} raw) jerk_ms={entry['jerk_ms']:.5f} "
                  f"(x{entry['jerk_ratio_recon_over_raw']:.2f} raw) L1={l1:.4f} ({entry['seconds']}s)", flush=True)
            del wrap
        except Exception as e:
            res["recon"][name] = {"error": repr(e)}
            print(f"[{args.tag}] {name} recon FAILED: {e!r}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print("WROTE", args.out)
    print("#### LABEL NOISE RECON DONE ####")


if __name__ == "__main__":
    main()
