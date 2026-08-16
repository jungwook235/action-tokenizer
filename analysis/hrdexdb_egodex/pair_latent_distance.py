"""HRDexDB human<->robot paired latent-distance analysis through the EgoDex tokenizer.

Question: is the tokenizer latent closer for a SAME-ACTION human/robot pair than for
DIFFERENT-ACTION pairs?  For each robot anchor chunk we measure the latent distance to
  * H-same : the paired human episode of the SAME object   (same action, cross-embodiment)
  * R-same : another robot episode of the SAME object       (same action, same embodiment)
  * R-diff : a robot episode of a DIFFERENT object          (diff action, same embodiment)
  * H-diff : a human episode of a DIFFERENT object          (diff action, cross-embodiment)

Both embodiments are pushed through the SAME encoder (`egodex_naivekey`, 57-dim) with an
identical, symmetric mapping:
  camera 9 dims  <- one shared static camera (same camera id on both sides)
  rightHand 9    <- human: MANO wrist (+9deg R_fix) | robot: arm EE flange pose (C2R)
  fingertips 15  <- ZERO on both sides (Inspire hand FK unavailable -> kept symmetric)
  left hand 24   <- ZERO on both sides
  DINO x0/x1     <- frames from that same static camera
World frame: per-episode, gravity aligned (both sessions have up = -z), origin at the
object's rest position, yaw from the camera->object direction, placed at an EgoDex-like
tabletop point.  Object-anchored (not camera-anchored) because the camera rig was
re-placed between the human and robot capture sessions.

Two variants are encoded from the same DINO features:
  A "wrist"  : camera dims zeroed too -> removes the per-session calibration confound
  B "wristcam": camera dims kept as mapped

Run (CPU only, env gr00t-actlat):
    python analysis/hrdexdb_egodex/pair_latent_distance.py [n_objects] [pairs_per_object]
"""

import glob
import importlib.util
import json
import os
import sys
import time

import decord
import numpy as np
import torch
import torchvision.transforms.v2 as TV

HUMAN = "/data/shared_dataset/HRDexDB/human"
ROBOT = "/data/shared_dataset/HRDexDB/inspire_dftp"
CAM = "22641005"          # static camera present in both sessions, close side view
CKPT = ("/data/rlwrld-unified-checkpoints/jungwook/action-tokenizer/"
        "checkpoints_action_tokenizer/joint_soupv1_v4_recon_dino_bn64_l1_mse_naiveln_vae_embtok/"
        "checkpoint-400000")
EMB = "egodex_naivekey"
H = 16
IMAGE_SIZE = 224
UP = np.array([0.0, 0.0, -1.0])
OBJ_ANCHOR = np.array([0.0, 0.80, -0.45])   # where the object's rest pose is placed
MOVE_THRESH = 0.02                          # m, lift-onset detection
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

_spec = importlib.util.spec_from_file_location(
    "chk", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "check_mano_egodex_correspondence.py"))
chk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chk)

_resize = TV.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=TV.InterpolationMode.BILINEAR,
                    antialias=True)


# --------------------------------------------------------------------- loading
def object_traj(ep):
    z = np.load(f"{ep}/object_6d_pose_v2.npz")
    return np.array([z[f"frame_{i}"][:3, 3] for i in range(len(z.files))], dtype=np.float64)


def lift_onset(T):
    d = np.linalg.norm(T - T[0], axis=1)
    hit = np.where(d > MOVE_THRESH)[0]
    return int(hit[0]) if len(hit) else -1


def cam_pose(ep):
    """(R_wc, C): chosen static camera's world->cam rotation and its world position."""
    E = np.array(json.load(open(f"{ep}/cam_param/extrinsics.json"))[CAM], dtype=np.float64)
    return E[:, :3], -E[:, :3].T @ E[:, 3]


def world_transform(obj0, C):
    """Rigid map: session world -> EgoDex-like frame (y up, camera looks along -z)."""
    f = C - obj0
    f = f - (f @ UP) * UP
    f /= np.linalg.norm(f)          # horizontal camera->... direction (object to camera)
    # camera sits at +z of the object in the new frame => object-to-camera maps to +z
    right = np.cross(UP, f)
    M = np.stack([right, UP, f])
    if np.linalg.det(M) < 0:
        M = np.stack([-right, UP, f])
    return M, obj0, OBJ_ANCHOR


def human_wrist(ep):
    J, G, _ = chk.load_hrdex_episode(ep, need_ego=False)
    return J[:, 0], G


def robot_wrist(ep):
    """EE flange pose resampled to video-frame times, in the camera/world frame."""
    C2R = np.load(f"{ep}/C2R.npy")                      # applies robot -> cam-world
    raw = np.load(f"{ep}/raw/arm/action.npy", allow_pickle=True)
    ee = np.array([np.array(x.tolist(), dtype=np.float64) for x in raw])
    at = np.load(f"{ep}/raw/arm/time.npy", allow_pickle=True).astype(np.float64)
    ts = np.load(f"{ep}/raw/timestamps/timestamp.npy")
    idx = np.searchsorted(at, ts).clip(0, len(at) - 1)
    ee = ee[idx]
    pos = (C2R[:3, :3] @ ee[:, :3, 3].T).T + C2R[:3, 3]
    rot = C2R[:3, :3] @ ee[:, :3, :3]
    return pos, rot


def build_geom(ep, kind, start, R_fix):
    """Aligned-frame geometry for one chunk: (wrist[16,3], handR[16,3,3], camC, camR) or None.

    No video read -- so this can run first, over everything, to fit the calibration.
    """
    T = object_traj(ep)
    wp, wR = human_wrist(ep) if kind == "human" else robot_wrist(ep)
    n = min(len(T), len(wp))
    if start < 0 or start + H > n:
        return None
    R_wc, C = cam_pose(ep)
    M, o, t_new = world_transform(T[0], C)
    xf = lambda p: (M @ (p - o).T).T + t_new
    xfR = lambda R: M @ R
    sl = slice(start, start + H)
    hand_R = np.stack([xfR(r) for r in wR[sl]])
    if kind == "human" and R_fix is not None:
        hand_R = hand_R @ R_fix.T                       # MANO -> ARKit hand frame (9 deg)
    return xf(wp[sl]), hand_R, xf(C[None])[0], xfR(R_wc.T)


def mean_rotation(Rs):
    U, _, Vt = np.linalg.svd(np.asarray(Rs).mean(0))
    return U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def fit_mount(geoms):
    """Global flange->hand calibration for the ROBOT side. Uses NO object/pair labels.

    R_mount: rotation taking the robot's mean EE frame onto the human's mean hand frame
             (the fixed hardware mount rotation; human side already has R_fix applied).
    t_mount: fixed offset in the corrected hand frame that matches the two global mean
             wrist positions (flange is ~proximal of where a wrist would be).
    Both are single constants for the whole dataset, so they cannot create
    same-action vs different-action structure.
    """
    hR = mean_rotation([g[1][t] for g in geoms if g[3] == "human" for t in range(H)])
    rR = mean_rotation([g[1][t] for g in geoms if g[3] == "robot" for t in range(H)])
    R_mount = rR.T @ hR                                  # R_corrected = R_ee @ R_mount
    hp = np.mean([g[0] for g in geoms if g[3] == "human"], axis=(0, 1))
    rp = np.mean([g[0] for g in geoms if g[3] == "robot"], axis=(0, 1))
    t_mount = (rR @ R_mount).T @ (hp - rp)               # expressed in the corrected frame
    ang = np.degrees(np.arccos(np.clip((np.trace(R_mount) - 1) / 2, -1, 1)))
    print(f"    [calib] flange->hand mount: rotation {ang:.1f} deg, "
          f"offset {1000*np.linalg.norm(t_mount):.0f} mm {np.round(t_mount, 3)}")
    return R_mount, t_mount


def to_57(geom, kind, mount):
    wrist, hand_R, camC, camR = geom[:4]
    if kind == "robot" and mount is not None:
        R_mount, t_mount = mount
        hand_R = hand_R @ R_mount
        wrist = wrist + np.einsum("tij,j->ti", hand_R, t_mount)
    a = np.zeros((H, 57))
    a[:, 0:3] = camC
    a[:, 3:9] = chk.R_to_rot6d(np.tile(camR, (H, 1, 1)))
    a[:, 9:12] = wrist
    a[:, 12:18] = chk.R_to_rot6d(hand_R)
    mn, mx = chk.egodex_minmax_57()
    rng = np.where(mx - mn == 0, 1.0, mx - mn)
    z = 2 * (a - mn) / rng - 1
    z[:, 18:57] = 0.0                                   # fingertips + left hand: symmetric zero
    return z


def read_frames(ep, start):
    vr = decord.VideoReader(f"{ep}/vid/{CAM}.mp4")
    i1 = min(start + H - 1, len(vr) - 1)
    fr = torch.from_numpy(vr.get_batch([min(start, len(vr) - 1), i1]).asnumpy()).float() / 255.0
    del vr
    fr = _resize(fr.permute(0, 3, 1, 2))
    return fr[0:1], fr[1:2]


# ----------------------------------------------------------------------- main
def main():
    n_obj = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    per_obj = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(8)
    t0 = time.time()

    # ---- pair list
    pairs = {}
    for g in sorted(glob.glob(f"{ROBOT}/*/[0-9]*/grasp_result.json")):
        r = os.path.dirname(g)
        obj = r.split("/")[-2]
        d = json.load(open(g))
        if not d.get("grasp_success"):
            continue
        h = f"{HUMAN}/{obj}/{d['human_paired_episode']}"
        need = [f"{r}/object_6d_pose_v2.npz", f"{h}/object_6d_pose_v2.npz",
                f"{r}/vid/{CAM}.mp4", f"{h}/vid/{CAM}.mp4", f"{r}/raw/arm/action.npy"]
        if all(os.path.exists(p) for p in need):
            pairs.setdefault(obj, []).append((r, h))
    objs = [o for o in sorted(pairs) if len(pairs[o]) >= per_obj][:n_obj]
    print(f"[{time.time()-t0:5.1f}s] {len(objs)} objects x {per_obj} pairs")

    R_fix, ang, rms, _ = chk.local_frame_geometry()
    print(f"[{time.time()-t0:5.1f}s] R_fix ready ({ang:.1f} deg)")

    from gr00t.model.action_latent_tokenizer_wrapper import ActionLatentTokenizerWrapper
    tok = ActionLatentTokenizerWrapper.from_checkpoint(
        CKPT, device="cpu", embodiment_id=EMB, vae_sample_override=False)
    tok.eval()

    # ---- pass 1: geometry only (no video), 2 phases per episode, anchored on lift onset
    recs = []
    for obj in objs:
        for pi, (r, h) in enumerate(pairs[obj][:per_obj]):
            for kind, ep in (("robot", r), ("human", h)):
                on = lift_onset(object_traj(ep))
                if on < H:
                    continue
                for phase, start in (("approach", on - H), ("lift", on)):
                    g = build_geom(ep, kind, start, R_fix)
                    if g is None:
                        continue
                    recs.append(dict(obj=obj, pair=pi, kind=kind, phase=phase, ep=ep,
                                     start=start, geom=g))
    print(f"[{time.time()-t0:5.1f}s] {len(recs)} chunks collected "
          f"(robot {sum(r['kind']=='robot' for r in recs)}, human {sum(r['kind']=='human' for r in recs)})")

    # ---- global flange->hand calibration (label-free), then the 57-dim vectors
    mount = fit_mount([(r["geom"][0], r["geom"][1], r["geom"][2], r["kind"]) for r in recs])
    for r in recs:
        r["z"] = to_57(r["geom"], r["kind"], mount)

    # ---- pass 2: DINO features once, two action variants
    lat = {"wrist": [], "wristcam": []}
    for i, rec in enumerate(recs):
        x0, x1 = read_frames(rec["ep"], rec["start"])
        with torch.no_grad():
            f0 = tok._frames_to_feats(x0, "cpu")
            f1 = tok._frames_to_feats(x1, "cpu")
            for name in ("wrist", "wristcam"):
                z = rec["z"].copy()
                if name == "wrist":
                    z[:, 0:9] = 0.0                     # drop camera dims
                a = torch.from_numpy(z).float()[None]
                _, main, _ = tok.encode(a, x0_feat=f0, x1_feat=f1)
                lat[name].append(main[0].numpy().ravel())
        if (i + 1) % 20 == 0:
            print(f"[{time.time()-t0:5.1f}s]   encoded {i+1}/{len(recs)}")
    np.savez_compressed(
        f"{OUT}/pair_latents.npz",
        lat_wrist=np.array(lat["wrist"]), lat_wristcam=np.array(lat["wristcam"]),
        raw=np.array([r["z"].ravel() for r in recs]),
        obj=np.array([r["obj"] for r in recs]), pair=np.array([r["pair"] for r in recs]),
        kind=np.array([r["kind"] for r in recs]), phase=np.array([r["phase"] for r in recs]),
        ep=np.array([r["ep"] for r in recs]))
    print(f"[{time.time()-t0:5.1f}s] saved {OUT}/pair_latents.npz")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    main()
