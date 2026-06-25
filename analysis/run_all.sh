#!/bin/bash
# Run the action-latent analysis for both V4-VAE tokenizers on their validation
# sets. Must run on a GPU node, from the action_tokenizer repo root, in the
# gr00t-actlat conda env. dinov2-large is fetched from HF on first use (needs
# internet), so HF offline flags are intentionally NOT set.
set -e
cd "$(dirname "$0")/.."   # -> action_tokenizer repo root
REPO=$(pwd)
OUT="$REPO/analysis/output"
mkdir -p "$OUT"

unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

DEXJOCO_CKPT="$REPO/checkpoints_action_tokenizer/dexjoco_dual_arm_v4_recon_dino_bn64_l1_mse_naiveln_vae/checkpoint-100000"
DEXJOCO_DATA=(
  /sjw_alinlab1/home/jungwook/dataset/dexjoco_lerobot/v20/bimanual/bimanual_assembly
  /sjw_alinlab1/home/jungwook/dataset/dexjoco_lerobot/v20/bimanual/bimanual_hanoi
  /sjw_alinlab1/home/jungwook/dataset/dexjoco_lerobot/v20/bimanual/bimanual_microwave_cook
  /sjw_alinlab1/home/jungwook/dataset/dexjoco_lerobot/v20/bimanual/bimanual_photograph
  /sjw_alinlab1/home/jungwook/dataset/dexjoco_lerobot/v20/bimanual/bimanual_unlock_ipad
)

GR1_CKPT="/sjw_alinlab1/home/jungwook/Isaac-GR00T/checkpoints_action_tokenizer/gr1_1000demos_v4_recon_dino_bn64_l1_mse_naiveln_vae/checkpoint-100000"
GR1_DATA=(/storage1/sjw_dataset/dataset/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.*)

echo "############## DEXJOCO DUAL ARM ##############"
python analysis/analyze_latents.py \
  --tag dexjoco_dual_arm \
  --checkpoint "$DEXJOCO_CKPT" \
  --data-config dexjoco_dual_arm_front \
  --dataset-path "${DEXJOCO_DATA[@]}" \
  --output "$OUT/dexjoco_dual_arm_latent_summary.txt" \
  --max-samples 4096

echo "############## GR1 1000 DEMOS (v4 VAE) ##############"
python analysis/analyze_latents.py \
  --tag gr1_1000demos \
  --checkpoint "$GR1_CKPT" \
  --data-config fourier_gr1_arms_waist \
  --dataset-path "${GR1_DATA[@]}" \
  --output "$OUT/gr1_1000demos_latent_summary.txt" \
  --max-samples 4096

# ---- additional tokenizers (deterministic: no VAE sampling, z == mu) ----

echo "############## GR1 100 DEMOS (v4 VAE) ##############"
# robocasa sim_100demos, same explicit fixed-val json as the v4-no-VAE / v3 gr1_100 runs.
python analysis/analyze_latents.py \
  --tag gr1_100demos_v4_vae \
  --checkpoint "/sjw_alinlab1/home/jungwook/Isaac-GR00T/checkpoints_action_tokenizer/gr1_100demos_v4_recon_dino_bn64_l1_mse_naiveln_vae/checkpoint-100000" \
  --data-config fourier_gr1_arms_waist \
  --dataset-path /storage1/sjw_dataset/dataset/robocasa_gr1_tabletop/sim_100demos \
  --fixed-val-path /sjw_alinlab1/home/jungwook/Isaac-GR00T/experiments/fixed_val_splits/gr1_100demos.json \
  --output "$OUT/gr1_100demos_v4_vae_latent_summary.txt" \
  --max-samples 4096

echo "############## GR1 100 DEMOS (v4, no VAE) ##############"
# robocasa sim_100demos, explicit fixed-val json shared across the v3/v4 gr1_100 runs.
python analysis/analyze_latents.py \
  --tag gr1_100demos_v4_novae \
  --checkpoint "/sjw_alinlab1/home/jungwook/Isaac-GR00T/checkpoints_action_tokenizer/gr1_100demos_v4_recon_dino_bn64_l1_mse_naiveln/checkpoint-100000" \
  --data-config fourier_gr1_arms_waist \
  --dataset-path /storage1/sjw_dataset/dataset/robocasa_gr1_tabletop/sim_100demos \
  --fixed-val-path /sjw_alinlab1/home/jungwook/Isaac-GR00T/experiments/fixed_val_splits/gr1_100demos.json \
  --output "$OUT/gr1_100demos_v4_novae_latent_summary.txt" \
  --max-samples 4096

echo "############## GR1 1000 DEMOS (v3, K=16) ##############"
# v3 is action-only (no DINO); same gr1_unified.* mixture + fixed-val as v4.
python analysis/analyze_latents.py \
  --tag gr1_1000demos_v3 \
  --checkpoint "/sjw_alinlab1/home/jungwook/Isaac-GR00T/checkpoints_action_tokenizer/gr1_1000demos_v3_recon_ln_bn16/checkpoint-100000" \
  --data-config fourier_gr1_arms_waist \
  --dataset-path "${GR1_DATA[@]}" \
  --output "$OUT/gr1_1000demos_v3_latent_summary.txt" \
  --max-samples 4096

echo "############## COMBINED LATENT TABLE ##############"
python analysis/combine_tables.py

# ===================================================================
# Clustering + t-SNE study (cluster input actions, color latent/decoded
# by the same action classes). Requires scikit-learn.
# ===================================================================
mkdir -p "$OUT/cluster"
echo "############## CLUSTER VIZ: dexjoco ##############"
python analysis/cluster_viz.py --tag dexjoco_dual_arm --data-config dexjoco_dual_arm_front \
  --checkpoint "$DEXJOCO_CKPT" --dataset-path "${DEXJOCO_DATA[@]}"
echo "############## CLUSTER VIZ: gr1_1000 v4vae ##############"
python analysis/cluster_viz.py --tag gr1_1000demos --data-config fourier_gr1_arms_waist \
  --checkpoint "$GR1_CKPT" --dataset-path "${GR1_DATA[@]}"
echo "############## CLUSTER VIZ: gr1_1000 v3 ##############"
python analysis/cluster_viz.py --tag gr1_1000demos_v3 --data-config fourier_gr1_arms_waist \
  --checkpoint "/sjw_alinlab1/home/jungwook/Isaac-GR00T/checkpoints_action_tokenizer/gr1_1000demos_v3_recon_ln_bn16/checkpoint-100000" \
  --dataset-path "${GR1_DATA[@]}"
echo "############## CLUSTER VIZ: gr1_100 v4vae ##############"
python analysis/cluster_viz.py --tag gr1_100demos_v4_vae --data-config fourier_gr1_arms_waist \
  --checkpoint "/sjw_alinlab1/home/jungwook/Isaac-GR00T/checkpoints_action_tokenizer/gr1_100demos_v4_recon_dino_bn64_l1_mse_naiveln_vae/checkpoint-100000" \
  --dataset-path /storage1/sjw_dataset/dataset/robocasa_gr1_tabletop/sim_100demos \
  --fixed-val-path /sjw_alinlab1/home/jungwook/Isaac-GR00T/experiments/fixed_val_splits/gr1_100demos.json
echo "############## CLUSTER VIZ: gr1_100 v4 noVAE ##############"
python analysis/cluster_viz.py --tag gr1_100demos_v4_novae --data-config fourier_gr1_arms_waist \
  --checkpoint "/sjw_alinlab1/home/jungwook/Isaac-GR00T/checkpoints_action_tokenizer/gr1_100demos_v4_recon_dino_bn64_l1_mse_naiveln/checkpoint-100000" \
  --dataset-path /storage1/sjw_dataset/dataset/robocasa_gr1_tabletop/sim_100demos \
  --fixed-val-path /sjw_alinlab1/home/jungwook/Isaac-GR00T/experiments/fixed_val_splits/gr1_100demos.json

echo "############## COMBINED CLUSTER TABLE ##############"
python analysis/combine_cluster.py

# ===================================================================
# DINO-decoder future-feature visualization (V4 tokenizers only; V3 has
# no DINO decoder). PCA-RGB + cosine maps + action-latent attention.
# ===================================================================
echo "############## DINO VIZ: dexjoco ##############"
python analysis/dino_decoder_viz.py --tag dexjoco_dual_arm --data-config dexjoco_dual_arm_front \
  --checkpoint "$DEXJOCO_CKPT" --dataset-path "${DEXJOCO_DATA[@]}" --tasks-max 5 --images-per-task 5
echo "############## DINO VIZ: gr1_1000 v4vae ##############"
python analysis/dino_decoder_viz.py --tag gr1_1000demos_v4_vae --data-config fourier_gr1_arms_waist \
  --checkpoint "$GR1_CKPT" --dataset-path "${GR1_DATA[@]}" --tasks-max 6 --images-per-task 5
echo "############## DINO VIZ: gr1_100 v4vae ##############"
python analysis/dino_decoder_viz.py --tag gr1_100demos_v4_vae --data-config fourier_gr1_arms_waist \
  --checkpoint "/sjw_alinlab1/home/jungwook/Isaac-GR00T/checkpoints_action_tokenizer/gr1_100demos_v4_recon_dino_bn64_l1_mse_naiveln_vae/checkpoint-100000" \
  --dataset-path /storage1/sjw_dataset/dataset/robocasa_gr1_tabletop/sim_100demos \
  --fixed-val-path /sjw_alinlab1/home/jungwook/Isaac-GR00T/experiments/fixed_val_splits/gr1_100demos.json \
  --tasks-max 1 --images-per-task 8
echo "############## DINO VIZ: gr1_100 v4 noVAE ##############"
python analysis/dino_decoder_viz.py --tag gr1_100demos_v4_novae --data-config fourier_gr1_arms_waist \
  --checkpoint "/sjw_alinlab1/home/jungwook/Isaac-GR00T/checkpoints_action_tokenizer/gr1_100demos_v4_recon_dino_bn64_l1_mse_naiveln/checkpoint-100000" \
  --dataset-path /storage1/sjw_dataset/dataset/robocasa_gr1_tabletop/sim_100demos \
  --fixed-val-path /sjw_alinlab1/home/jungwook/Isaac-GR00T/experiments/fixed_val_splits/gr1_100demos.json \
  --tasks-max 1 --images-per-task 8

echo "DONE. Reports in $OUT"
