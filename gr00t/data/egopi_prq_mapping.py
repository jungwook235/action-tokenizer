"""Ego-Pi {p,r,q} mapping — land human + robot onto ONE robot-anchored canonical space.

VERBATIM copy of RLDX-1-egopi/rldx/data/egopi/prq_mapping.py (junhyeong, FRAME V1).
Do not edit here — the constants (P_OFFSET / R_CONV / head pose) form ONE matched set
with the FK cache this repo consumes; any change must come from a refit upstream.

Robot-anchored: the canonical {p,r,q} IS robot space. Robot is native (identity);
human is mapped ONTO robot via per-channel affine:
  q (hand joints) : q_robot = (q_human + Q_DELTA) * Q_SCALE   # IDENTITY (validated 13mm)
  p (wrist pos)   : p_robot = (p_human - P_OFFSET) * P_SCALE  # const offset
  r (wrist rot)   : R_robot = R_CONV @ R_human                # const conv rotation

Layout of the 15D vector: [ p(3) | rot6d(6) | q(6) ].

Robot side does NOT recompute FK here — it reuses the precomputed openarm FK cache
(wrist_pos_R + wrist_rot_R rotvec, ZED frame) via `robot_cache_to_prq`. Keeping the
SAME frame conventions the cache was built under is what makes P_OFFSET / R_CONV valid.

Pure numpy + scipy (no mujoco, no rldx imports) so it is unit-testable standalone.
Ported from Isaac-GR00T/scripts/prq_human_robot_mapping.py + q_sanity / prq_accuracy.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as _Rot

# ──────────────────────────────────────────────────────────────────────────────
# FRAME SET (head pose + human→robot affine) — ONE MATCHED SET. Do not edit piecemeal.
#
# The robot canonical {p,r,q} lives in the ZED-camera frame that PointsFK builds at
# (EGOPI_HEAD_PITCH, EGOPI_HEAD_YAW). P_OFFSET / R_CONV were fit to bridge RAW human↔robot
# *in that camera frame*. So the head pose and the affine are coupled: change the head and
# both P_OFFSET and R_CONV must be REFIT (robot wrist moves ~25°/Δp in the camera frame —
# verified kinematically). The cache builder (build_egopi_cache.py) and the deploy converter
# (prq_deploy.py) both read EGOPI_HEAD_PITCH/YAW from here so the whole pipeline stays on ONE
# frame; never hard-code a head value at a call site again.
#
# ⚠️ FRAME V1 (current ckpt): head=(1.2, -0.3). This is the PointsFK default ported from
# Isaac, NOT the real openarm_teleop_v3 neck (measured ≈0.879/0.001, fixed). The training
# *images* are at the real neck, so robot prq carries a fixed ~25.16° / ~0.2-0.3m camera-frame
# warp the model had to learn, and human prq (HaWoR real-cam frame) does not — an asymmetry
# that weakens transfer. To move to the image-aligned frame: set the head below to the real
# neck, REFIT P_OFFSET/R_CONV on the regenerated cache (build_egopi_cache --emit-affine-fit),
# regenerate egopi_prq_stats.json, and RETRAIN. Keep V1 until a V2 ckpt exists.
# ──────────────────────────────────────────────────────────────────────────────
EGOPI_HEAD_PITCH = 1.2   # FRAME V1 — refit P_OFFSET/R_CONV below if you change this
EGOPI_HEAD_YAW = -0.3

Q_DELTA = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
Q_SCALE = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
# q = IDENTITY: pose-level fingertip check (13mm, openness corr +0.96, 0% OOB) showed
# HaWoR inspire6 already reproduces the hand; the distribution-fit scales were circular.
P_OFFSET = np.array([0.116, 0.117, -0.017], dtype=np.float64)   # human_mean - robot_mean (m) @ V1 head
P_SCALE = np.array([1.0, 1.0, 1.0], dtype=np.float64)
R_CONV = _Rot.from_rotvec(np.deg2rad(31.7) * np.array([0.5, -0.74, -0.45])).as_matrix()  # @ V1 head

PRQ_DIM = 15  # p(3) + rot6d(6) + q(6)

# Human rlwrld lerobot state layout (30D): right slots.
_H_R_TRANS = slice(15, 18)
_H_R_ROT6D = slice(18, 24)
_H_R_INSP6 = slice(24, 30)
# Robot openarm state layout (28D): right hand joints.
_R_HAND_R = slice(22, 28)


# ──────────────────────────────────────────────────────────────────────────────
# rot6d <-> R (Zhou et al.: rot6d = first two COLUMNS of R)
# ──────────────────────────────────────────────────────────────────────────────
def rot6d_to_R(a6: np.ndarray) -> np.ndarray:
    a1, a2 = a6[:3], a6[3:]
    b1 = a1 / (np.linalg.norm(a1) + 1e-9)
    a2p = a2 - (b1 @ a2) * b1
    b2 = a2p / (np.linalg.norm(a2p) + 1e-9)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=-1)


def R_to_rot6d(R: np.ndarray) -> np.ndarray:
    return np.concatenate([R[:, 0], R[:, 1]])


def rotvec_to_rot6d(rv: np.ndarray) -> np.ndarray:
    """openarm FK cache stores wrist_rot as axis-angle (rotvec) → rot6d."""
    return R_to_rot6d(_Rot.from_rotvec(rv).as_matrix())


# ──────────────────────────────────────────────────────────────────────────────
# Human → robot-anchored {p,r,q} 15D
# ──────────────────────────────────────────────────────────────────────────────
def human_to_prq(h30: np.ndarray) -> np.ndarray:
    """rlwrld human state 30D → robot-frame {p,r,q} 15D (RIGHT hand)."""
    p = h30[_H_R_TRANS]
    r6 = h30[_H_R_ROT6D]
    q = h30[_H_R_INSP6]
    p_m = (p - P_OFFSET) * P_SCALE
    R_m = R_CONV @ rot6d_to_R(r6)
    q_m = (q + Q_DELTA) * Q_SCALE
    return np.concatenate([p_m, R_to_rot6d(R_m), q_m]).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Robot → canonical {p,r,q} 15D (FK cache + raw hand q; robot is native/identity)
# ──────────────────────────────────────────────────────────────────────────────
def robot_cache_to_prq(wrist_pos: np.ndarray, wrist_rotvec: np.ndarray, hand_q6: np.ndarray) -> np.ndarray:
    """FK-cache right wrist (pos 3 + rotvec 3, ZED frame) + raw inspire6 → 15D {p,r,q}."""
    return np.concatenate(
        [np.asarray(wrist_pos, np.float64),
         rotvec_to_rot6d(np.asarray(wrist_rotvec, np.float64)),
         np.asarray(hand_q6, np.float64)]
    ).astype(np.float32)
