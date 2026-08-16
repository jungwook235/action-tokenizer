"""Check that HRDexDB MANO annotations correspond to the EgoDex-lerobot `action` vector
that the joint tokenizer (data_config `human_egodex_camera_hand_unit`) consumes.

EgoDex tokenizer action = 57 dims, in this concat order:
    camera_pos(3) camera_rot6(6)
    rightHand_pos(3) rightHand_rot6(6) right{Thumb,Index,Middle,Ring,Little}Tip_pos(15)
    leftHand_pos(3)  leftHand_rot6(6)  left{...}Tip_pos(15)
all min-max normalized to [-1,1] with meta/stats.json min/max.

HRDexDB source per frame (single RIGHT hand):
    hand/mano_params/%05d.json : joints(21,3) global_orient(1,1,3,3) transl(1,3) betas(1,10)
    cam_param/ego_calib.json   : per-frame world->cam [3x4] for 2 head cameras
Both datasets are metric (m) and 30 fps.

Run:  python analysis/hrdexdb_egodex/check_mano_egodex_correspondence.py
"""

import glob
import json
import os
import random

import numpy as np
import pyarrow.parquet as pq

EGODEX_ROOT = "/data/shared_dataset/egodex-lerobot/part2/basic_pick_place"
HRDEX_HUMAN = "/data/shared_dataset/HRDexDB/human"
EGO_CAM = "25452062"  # left head camera; "25452066" is the stereo partner

# MANO joints are in manopth/OpenPose order: wrist, thumb, index, middle, ring, pinky
MANO_TIPS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, little -- same order as EgoDex
MANO_MCP = [5, 9, 13, 17]  # index/middle/ring/pinky MCP (rigid wrt the wrist frame)

ACTION_57 = [
    ("camera_pos", 3), ("camera_rot", 6),
    ("rightHand_pos", 3), ("rightHand_rot", 6),
    ("rightThumbTip_pos", 3), ("rightIndexFingerTip_pos", 3), ("rightMiddleFingerTip_pos", 3),
    ("rightRingFingerTip_pos", 3), ("rightLittleFingerTip_pos", 3),
    ("leftHand_pos", 3), ("leftHand_rot", 6),
    ("leftThumbTip_pos", 3), ("leftIndexFingerTip_pos", 3), ("leftMiddleFingerTip_pos", 3),
    ("leftRingFingerTip_pos", 3), ("leftLittleFingerTip_pos", 3),
]
EGO_KNUCKLES = [
    "rightIndexFingerKnuckle_pos", "rightMiddleFingerKnuckle_pos",
    "rightRingFingerKnuckle_pos", "rightLittleFingerKnuckle_pos",
]

HRDEX_UP = np.array([0.0, 0.0, -1.0])  # verified: HRDexDB world +z points down


# ---------------------------------------------------------------- helpers
def rot6d_to_R(a):
    """EgoDex convention (verified): the 6 numbers are the first two COLUMNS of R."""
    a = np.asarray(a, dtype=np.float64)
    x = a[..., :3] / np.linalg.norm(a[..., :3], axis=-1, keepdims=True)
    y = a[..., 3:] - (a[..., 3:] * x).sum(-1, keepdims=True) * x
    y = y / np.linalg.norm(y, axis=-1, keepdims=True)
    return np.stack([x, y, np.cross(x, y)], axis=-1)


def R_to_rot6d(R):
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def egodex_slices():
    mod = json.load(open(f"{EGODEX_ROOT}/meta/modality.json"))["action"]
    return mod


def load_egodex_actions(n_eps=40, seed=0):
    mod = egodex_slices()
    files = sorted(glob.glob(f"{EGODEX_ROOT}/data/chunk-000/*.parquet"))
    files = random.Random(seed).sample(files, n_eps)
    out = []
    for f in files:
        A = np.stack(pq.read_table(f)["action"].to_numpy()).astype(np.float64)
        out.append(A)
    return np.concatenate(out), mod


def egodex_57(A, mod):
    return np.concatenate([A[:, mod[k]["start"]:mod[k]["end"]] for k, _ in ACTION_57], axis=1)


def egodex_minmax_57():
    st = json.load(open(f"{EGODEX_ROOT}/meta/stats.json"))["action"]
    mod = egodex_slices()
    mn = np.array(st["min"], dtype=np.float64)
    mx = np.array(st["max"], dtype=np.float64)
    sl = np.concatenate([np.arange(mod[k]["start"], mod[k]["end"]) for k, _ in ACTION_57])
    return mn[sl], mx[sl]


def load_hrdex_episode(ep, need_ego=True):
    fs = sorted(glob.glob(f"{ep}/hand/mano_params/*.json"))
    J, G = [], []
    for f in fs:
        d = json.load(open(f))
        J.append(d["joints"])
        G.append(np.array(d["global_orient"])[0, 0])
    J = np.array(J, dtype=np.float64)          # (T,21,3) world
    G = np.array(G, dtype=np.float64)          # (T,3,3)  world rotation of the wrist
    E = None
    p = f"{ep}/cam_param/ego_calib.json"
    if os.path.exists(p):                      # 413/441 episodes have head cameras
        ego = json.load(open(p))["extrinsics"][EGO_CAM]
        E = np.array([ego["%05d" % t] for t in range(len(fs))], dtype=np.float64)  # world->cam
    elif need_ego:
        return None, None, None
    return J, G, E


# ------------------------------------------------- geometry correspondence
def local_frame_geometry():
    """Compare MANO's wrist frame against EgoDex's ARKit right-hand frame."""
    A, mod = load_egodex_actions()
    g = lambda k: A[:, mod[k]["start"]:mod[k]["end"]]
    R = rot6d_to_R(g("rightHand_rot"))
    Rt = np.transpose(R, (0, 2, 1))
    w = g("rightHand_pos")
    ego_knu = np.stack([np.einsum("tij,tj->ti", Rt, g(k) - w) for k in EGO_KNUCKLES], 1).mean(0)
    ego_tip = np.stack(
        [np.einsum("tij,tj->ti", Rt, g(f"right{f}Tip_pos") - w)
         for f in ["Thumb", "IndexFinger", "MiddleFinger", "RingFinger", "LittleFinger"]], 1).mean(0)

    eps = random.Random(0).sample(sorted(glob.glob(f"{HRDEX_HUMAN}/*/[0-9]")), 40)
    mcp, tip = [], []
    for ep in eps:
        J, G, _ = load_hrdex_episode(ep, need_ego=False)
        L = np.einsum("tij,tkj->tki", np.transpose(G, (0, 2, 1)), J - J[:, 0:1])
        mcp.append(L[:, MANO_MCP])
        tip.append(L[:, MANO_TIPS])
    mano_mcp = np.concatenate(mcp).mean(0)
    mano_tip = np.concatenate(tip).mean(0)

    # Kabsch: MANO wrist frame -> EgoDex hand frame, from the 4 rigid finger MCPs
    H = mano_mcp.T @ ego_knu
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R_fix = Vt.T @ np.diag([1, 1, d]) @ U.T
    res = (R_fix @ mano_mcp.T).T - ego_knu
    ang = np.degrees(np.arccos(np.clip((np.trace(R_fix) - 1) / 2, -1, 1)))
    tip_err = np.linalg.norm((R_fix @ mano_tip.T).T - ego_tip, axis=1)
    return R_fix, ang, np.sqrt((res ** 2).sum(1).mean()), tip_err


# --------------------------------------------------------- world alignment
def world_transform(E0, head_height=1.166, head_xz=(-0.042, -0.212)):
    """Rigid HRDexDB-world -> EgoDex-like world (y up, head forward = -z).

    E0: (3,4) world->cam of the head camera at frame 0 of the episode.
    """
    R_wc, t = E0[:, :3], E0[:, 3]
    C = -R_wc.T @ t                       # head position, HRDexDB world
    f = R_wc[2, :]                        # optical axis in world
    f = f - (f @ HRDEX_UP) * HRDEX_UP     # horizontalize
    f /= np.linalg.norm(f)
    right = np.cross(f, HRDEX_UP)         # = up x (-f), keeps [x,y,z] right-handed (det M = +1)
    M = np.stack([right, HRDEX_UP, -f])   # rows -> new x (right), y (up), z (back)
    assert np.linalg.det(M) > 0
    t_new = np.array([head_xz[0], head_height, head_xz[1]])
    return M, C, t_new


def hrdex_to_egodex57(ep, R_fix=None, head_height=1.166, left_fill=None):
    J, G, E = load_hrdex_episode(ep, need_ego=True)
    if J is None:
        return None
    M, C, t_new = world_transform(E[0], head_height=head_height)
    xf = lambda p: (M @ (p - C).T).T + t_new          # positions
    xfR = lambda R: M @ R                             # rotations

    T = len(J)
    cam_R = np.stack([xfR(E[t, :, :3].T) for t in range(T)])       # world-from-cam
    cam_C = xf(np.stack([-E[t, :, :3].T @ E[t, :, 3] for t in range(T)]))
    wrist = xf(J[:, 0])
    hand_R = np.stack([xfR(G[t]) for t in range(T)])
    if R_fix is not None:
        hand_R = hand_R @ R_fix.T                     # MANO wrist frame -> ARKit hand frame
    tips = np.stack([xf(J[:, i]) for i in MANO_TIPS], 1)          # (T,5,3)

    right = np.concatenate([wrist, R_to_rot6d(hand_R), tips.reshape(T, 15)], 1)
    cam = np.concatenate([cam_C, R_to_rot6d(cam_R)], 1)
    if left_fill is None:
        left = np.full((T, 24), np.nan)
    else:
        left = np.tile(left_fill, (T, 1))
    return np.concatenate([cam, right, left], 1)


# ------------------------------------------------------------------- main
def main():
    R_fix, ang, rms, tip_err = local_frame_geometry()
    print("=" * 78)
    print("1) MANO wrist frame  vs  EgoDex ARKit right-hand frame")
    print("   Kabsch on the 4 rigid finger MCP/knuckle offsets:")
    print("     residual rotation %.2f deg, RMS %.1f mm" % (ang, 1000 * rms))
    print("     fingertip mean |diff| after alignment (mm):",
          np.round(1000 * tip_err, 1), "  order thumb,index,middle,ring,little")

    mn, mx = egodex_minmax_57()
    A, mod = load_egodex_actions()
    ego57 = egodex_57(A, mod)
    print("\n2) EgoDex reference: normalized |value| quantiles over 40 episodes")
    z = 2 * (ego57 - mn) / (mx - mn) - 1
    print("   p50 %.2f  p95 %.2f  frac inside [-1,1] %.4f" %
          (np.percentile(np.abs(z), 50), np.percentile(np.abs(z), 95), (np.abs(z) <= 1).mean()))

    eps = random.Random(2).sample(sorted(glob.glob(f"{HRDEX_HUMAN}/*/[0-9]")), 20)
    for hh in (1.166, 1.46):
        out = [hrdex_to_egodex57(e, R_fix=R_fix, head_height=hh) for e in eps]
        vals = np.concatenate([o for o in out if o is not None])
        v = vals[:, :33]                                  # camera + right hand (left is absent)
        zz = 2 * (v - mn[:33]) / (mx[:33] - mn[:33]) - 1
        print("\n3) HRDexDB -> EgoDex-57 (head placed at y=%.3f), 20 episodes, %d frames" % (hh, len(v)))
        print("   normalized: p50 %.2f  p95 %.2f  max %.2f  frac inside [-1,1] %.4f" %
              (np.percentile(np.abs(zz), 50), np.percentile(np.abs(zz), 95),
               np.abs(zz).max(), (np.abs(zz) <= 1).mean()))
        names = [n for n, d in ACTION_57 for _ in range(d)][:33]
        bad = [(names[i], i, float(np.abs(zz[:, i]).max())) for i in range(33)
               if (np.abs(zz[:, i]) > 1).mean() > 0.02]
        for n, i, m in bad:
            print("     out-of-range dim %2d (%s): frac %.2f, max |z| %.2f"
                  % (i, n, (np.abs(zz[:, i]) > 1).mean(), m))
    print("\n   NOTE: dims 33..56 (left hand) have no HRDexDB source -> NaN above.")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    main()
