#!/bin/bash
# Visual-separation analysis: does the DINO-fused v4 tokenizer separate action
# chunks that are identical in action space but different in visual dynamics,
# while the action-only v3 tokenizer cannot?
#
# Stage 1 (collect) needs a GPU + internet (dinov2-large from HF). Stages 2-3
# (stats/frames) are CPU-only and reuse the cache.
#
# Run from the action_tokenizer repo root, in the gr00t-actlat conda env.
set -e
cd "$(dirname "$0")/.."   # -> action_tokenizer repo root
REPO=$(pwd)
OUT="$REPO/analysis/output/visual_sep"
mkdir -p "$OUT"

unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

V3_CKPT="$REPO/checkpoints_action_tokenizer/dexjoco_dual_arm_v3_recon_ln_bn16/checkpoint-100000"
V4_CKPT="$REPO/checkpoints_action_tokenizer/dexjoco_dual_arm_v4_recon_dino_bn64_l1_mse_naiveln_vae/checkpoint-100000"
DATA=(
  /sjw_alinlab1/home/jungwook/dataset/dexjoco_lerobot/v20/bimanual/bimanual_assembly
  /sjw_alinlab1/home/jungwook/dataset/dexjoco_lerobot/v20/bimanual/bimanual_hanoi
  /sjw_alinlab1/home/jungwook/dataset/dexjoco_lerobot/v20/bimanual/bimanual_microwave_cook
  /sjw_alinlab1/home/jungwook/dataset/dexjoco_lerobot/v20/bimanual/bimanual_photograph
  /sjw_alinlab1/home/jungwook/dataset/dexjoco_lerobot/v20/bimanual/bimanual_unlock_ipad
)

echo "############## STAGE 1: COLLECT (GPU) ##############"
python analysis/vsep_collect.py \
  --v3-ckpt "$V3_CKPT" --v4-ckpt "$V4_CKPT" \
  --data-config dexjoco_dual_arm_front \
  --dataset-path "${DATA[@]}" \
  --target-total 3000 --min-per-dataset 60 \
  --batch-size 64 --num-workers 8

echo "############## STAGE 2: STATS ①③④ ##############"
python analysis/vsep_stats.py --k 8

echo "############## STAGE 3: NEAR-DUP FRAMES ② ##############"
# NOTE: on dexjoco dual-arm this finds 0 groups — the 5 tasks have near-disjoint
# action spaces, so there are essentially no same-action/different-visual pairs to
# observe. Kept for datasets that DO have action-collisions (e.g. gr1 robocasa).
python analysis/vsep_frames.py --radius-pct 1.0 --min-size 4 --max-groups 6 || true

echo "############## STAGE 4: FRAME-SWAP INTERVENTION ②′ ##############"
# The decisive test that does NOT need natural action-collisions: hold the action
# byte-fixed and swap the visual, measuring how far the latent moves (v3 == 0).
python analysis/vsep_swap.py \
  --v3-ckpt "$V3_CKPT" --v4-ckpt "$V4_CKPT" \
  --data-config dexjoco_dual_arm_front \
  --dataset-path "${DATA[@]}" \
  --target-total 750 --min-per-dataset 60 --n-donors 24 --batch-size 128

echo "DONE. Results in $OUT"
ls -la "$OUT"
