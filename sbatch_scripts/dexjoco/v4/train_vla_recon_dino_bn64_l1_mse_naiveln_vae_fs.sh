#!/bin/bash
#SBATCH --job-name=full_v4_dexjoco_dual_arm_recon_dino_bn64_l1_mse_naiveln_vae_fs_fix_imagetransform
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b200:4
#SBATCH --nodes=1
#SBATCH --qos=preempt
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120
#SBATCH --time=72:00:00
#SBATCH --output=/NHNHOME/data/wook/action-tokenizer/slurm/logs_dexjoco/full_v4_dexjoco_dual_arm_recon_dino_bn64_l1_mse_naiveln_vae_fs_fix_imagetransform_%j.out
#SBATCH --error=/NHNHOME/data/wook/action-tokenizer/slurm/logs_dexjoco/full_v4_dexjoco_dual_arm_recon_dino_bn64_l1_mse_naiveln_vae_fs_fix_imagetransform_%j.err


set -ex  # -e: abort the whole job (skip Stage-2) if Stage-1 fails
export PATH="$HOME/.local/bin:$PATH"
export WANDB_PROJECT=Action-Tokenizer-DexJoCo-DualArm
export WANDB_API_KEY="66a73856614bc24a07523f3fbee42482fcbeffe3"
export HF_TOKEN="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"

BASE_DIR=/NHNHOME/data/wook/action-tokenizer
cd $BASE_DIR

source /NHNHOME/data/wook/miniconda3/bin/activate gr00t-actlat
export PATH="$CONDA_PREFIX/bin:$PATH"
hash -r
echo "[env-check] which python=$(which python)"
echo "[env-check] CONDA_PREFIX=$CONDA_PREFIX"
python -c "import sys, transformers; print('exe=', sys.executable, 'transformers=', transformers.__version__)"

# robocasa_gr1_tabletop is a GR1 embodiment (same action keys + single video.ego_view),
# so the same data-config (fourier_gr1_arms_waist) is reused; only the dataset path
# and per-dataset normalization stats differ.
DATA_DIR=("/NHNHOME/data/wook/dataset/dexjoco_lerobot/v20/bimanual/bimanual_assembly"
"/NHNHOME/data/wook/dataset/dexjoco_lerobot/v20/bimanual/bimanual_hanoi"
"/NHNHOME/data/wook/dataset/dexjoco_lerobot/v20/bimanual/bimanual_microwave_cook"
"/NHNHOME/data/wook/dataset/dexjoco_lerobot/v20/bimanual/bimanual_photograph"
"/NHNHOME/data/wook/dataset/dexjoco_lerobot/v20/bimanual/bimanual_unlock_ipad"
)
TOK_CKPT_DIR=checkpoints_action_tokenizer/dexjoco_dual_arm_v4_recon_dino_bn64_l1_mse_naiveln_vae
VLA_CKPT_DIR=checkpoints/vla_actlat_fm_dexjoco_dual_arm/v4_recon_dino_bn64_l1_mse_naiveln_vae_fs_fix_imagetransform
TOK_STEP=100000
ABS_TOK_CKPT="/NHNHOME/data/wook/action-tokenizer/$TOK_CKPT_DIR"


# Re-enable HF network for Stage-2 (GR00T base / eagle may need download).
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

# === Stage 2: VLA Training ===
# The wrapper auto-detects the VAE (and the naive final-norm) from the tokenizer
# checkpoint markers and rebuilds the matching encoder; the VLA target is the
# sampled latent z. Frames flow through unchanged.
python scripts/gr00t_finetune_actlat_fm.py \
    --dataset-path "${DATA_DIR[@]}" \
    --output-dir $VLA_CKPT_DIR \
    --data-config dexjoco_dual_arm \
    --embodiment-tag new_embodiment \
    --base-model-path "nvidia/GR00T-N1.5-3B" \
    --run-name "actlat_fm_v4_dexjoco_dual_arm_recon_dino_bn64_l1_mse_naiveln_vae_fs_fix_imagetransform" \
    --num-gpus 4 \
    --batch-size 16 \
    --max-steps 60000 \
    --save-steps 5000 \
    --save-total-limit 3 \
    --eval-steps 1000 \
    --actlat-tokenizer-path "$ABS_TOK_CKPT/checkpoint-$TOK_STEP" \
    --actlat-target-tokens "all" \
    --actlat-frames \
    --val-ratio 0.003 \
    --use-fixed-val \
    --no-load-action-head \
    --video-backend "decord"
