"""CPU-only smoke test: one HRDexDB episode -> EgoDex-57 action -> tokenizer latent.

Decisions baked in (per user):
  * rightHand_rot gets the 9-deg MANO->ARKit frame correction (R_fix)
  * camera dims use the real HRDexDB head-camera pose (no zero-fill)
  * left-hand 24 dims are zeroed AFTER min-max normalization
  * only episodes with cam_param/ego_calib.json are usable (413/441)

Run (env gr00t-actlat, no GPU needed):
    python analysis/hrdexdb_egodex/extract_latent_cpu_smoketest.py [episode_dir]
"""

import importlib.util
import json
import os
import sys
import time

import decord
import numpy as np
import torch
import torchvision.transforms.v2 as TV

CKPT = ("/data/rlwrld-unified-checkpoints/jungwook/action-tokenizer/"
        "checkpoints_action_tokenizer/joint_soupv1_v4_recon_dino_bn64_l1_mse_naiveln_vae_embtok/"
        "checkpoint-400000")
EMBODIMENT = "egodex_naivekey"
EGODEX_ROOT = "/data/shared_dataset/egodex-lerobot/part2/basic_pick_place"
EGO_CAM = "25452062"
H = 16  # action horizon
IMAGE_SIZE = 224

_here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "chk", os.path.join(_here, "check_mano_egodex_correspondence.py"))
chk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chk)


def normalize_57(a):
    """min_max -> [-1,1] with EgoDex action stats, then zero the left-hand block."""
    mn, mx = chk.egodex_minmax_57()
    rng = np.where(mx - mn == 0, 1.0, mx - mn)
    z = 2 * (a - mn) / rng - 1
    z[:, 33:57] = 0.0  # left hand: no HRDexDB source
    return z


def read_frames(ep, i0, i1):
    vr = decord.VideoReader(f"{ep}/vid/{EGO_CAM}.mp4")
    f = torch.from_numpy(vr.get_batch([i0, i1]).asnumpy()).float() / 255.0  # [2,H,W,3]
    del vr
    f = f.permute(0, 3, 1, 2)
    resize = TV.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=TV.InterpolationMode.BILINEAR,
                       antialias=True)
    return resize(f)  # [2,3,224,224] in [0,1]


def main():
    ep = sys.argv[1] if len(sys.argv) > 1 else "/data/shared_dataset/HRDexDB/human/apple/0"
    torch.set_num_threads(8)
    torch.manual_seed(0)

    t0 = time.time()
    R_fix, ang, rms, _ = chk.local_frame_geometry()
    print(f"[{time.time()-t0:5.1f}s] R_fix ready (MANO->ARKit {ang:.1f} deg, RMS {1000*rms:.1f} mm)")

    a57 = chk.hrdex_to_egodex57(ep, R_fix=R_fix)          # (T,57), left block = NaN
    assert a57 is not None, f"{ep} has no ego_calib.json"
    T = len(a57)
    a57 = np.nan_to_num(a57, nan=0.0)
    z = normalize_57(a57)
    print(f"[{time.time()-t0:5.1f}s] episode {ep}  T={T}  chunks(stride16)={T//H}")
    print(f"          normalized |z| p50 {np.percentile(np.abs(z[:, :33]),50):.2f} "
          f"p95 {np.percentile(np.abs(z[:, :33]),95):.2f} (cam+right dims)")

    from gr00t.model.action_latent_tokenizer_wrapper import ActionLatentTokenizerWrapper
    tok = ActionLatentTokenizerWrapper.from_checkpoint(
        CKPT, device="cpu", embodiment_id=EMBODIMENT, vae_sample_override=False)
    tok.eval()
    print(f"[{time.time()-t0:5.1f}s] tokenizer loaded on CPU "
          f"(action_dim={tok.action_dim}, horizon={tok.action_horizon}, emb_dim={tok.emb_dim}, "
          f"main_tok={tok.num_main_tokens}, global={tok.num_global_tokens}, hand={tok.num_hand_tokens})")

    start = 0  # first chunk only
    actions = torch.from_numpy(z[start:start + H]).float()[None]      # [1,16,57]
    frames = read_frames(ep, start, start + H - 1)
    x0, x1 = frames[0:1], frames[1:2]

    t1 = time.time()
    with torch.no_grad():
        g, main, hand = tok.encode(actions, x0=x0, x1=x1)
    t_enc = time.time() - t1
    print(f"[{time.time()-t0:5.1f}s] encode done in {t_enc:.1f}s  "
          f"latent(time)={tuple(main.shape)} global={tuple(g.shape)} hand={tuple(hand.shape)}")
    print(f"          latent mu: mean {main.mean():+.4f} std {main.std():.4f} "
          f"min {main.min():+.3f} max {main.max():+.3f}")

    with torch.no_grad():
        recon = tok.decode_latent(main, target_tokens="time")
    err = (recon[0, :, :33] - actions[0, :, :33]).abs()
    print(f"          recon L1 (normalized, cam+right dims): mean {err.mean():.4f} max {err.max():.4f}")

    # steady-state cost: DINO + tokenizer are now warm
    n = 3
    t2 = time.time()
    for k in range(1, n + 1):
        s0 = k * H
        with torch.no_grad():
            fr = read_frames(ep, s0, s0 + H - 1)
            tok.encode(torch.from_numpy(z[s0:s0 + H]).float()[None], x0=fr[0:1], x1=fr[1:2])
    print(f"          warm cost: {(time.time()-t2)/n:.2f} s / chunk (16 frames, 8 CPU threads)")
    print(f"[{time.time()-t0:5.1f}s] total")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    main()
