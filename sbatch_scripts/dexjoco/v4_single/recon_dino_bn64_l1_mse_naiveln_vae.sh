#!/bin/bash
#SBATCH --job-name=full_v4_dexjoco_single_arm_recon_dino_bn64_l1_mse_naiveln_vae
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b200:8
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --output=/NHNHOME/data/wook/action-tokenizer/slurm/logs_dexjoco/full_v4_dexjoco_single_arm_recon_dino_bn64_l1_mse_naiveln_vae_%j.out
#SBATCH --error=/NHNHOME/data/wook/action-tokenizer/slurm/logs_dexjoco/full_v4_dexjoco_single_arm_recon_dino_bn64_l1_mse_naiveln_vae_%j.err


set -ex  # -e: abort the whole job (skip Stage-2) if Stage-1 fails
export PATH="$HOME/.local/bin:$PATH"
export WANDB_PROJECT=Action-Tokenizer-DexJoCo-SingleArm
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

DATA_DIR=("/NHNHOME/data/wook/dataset/dexjoco_lerobot/v20/click_mouse"
"/NHNHOME/data/wook/dataset/dexjoco_lerobot/v20/hammer_nail"
"/NHNHOME/data/wook/dataset/dexjoco_lerobot/v20/water_plant"
"/NHNHOME/data/wook/dataset/dexjoco_lerobot/v20/fold_glasses"
"/NHNHOME/data/wook/dataset/dexjoco_lerobot/v20/pick_bucket"
"/NHNHOME/data/wook/dataset/dexjoco_lerobot/v20/pinch_tongs"
)
TOK_CKPT_DIR=checkpoints_action_tokenizer/dexjoco_single_arm_v4_recon_dino_bn64_l1_mse_naiveln_vae
VLA_CKPT_DIR=checkpoints/vla_actlat_fm_dexjoco_single_arm/v4_recon_dino_bn64_l1_mse_naiveln_vae
TOK_STEP=100000
ABS_TOK_CKPT="/NHNHOME/data/wook/action-tokenizer/$TOK_CKPT_DIR"

# === Stage 1: V4 Tokenizer Training — RLA-DINO hybrid, NAIVE final LayerNorm, SD-style VAE ===
# Identical to recon_dino_bn64_l1_mse_naiveln.sh EXCEPT the latent bottleneck is a
# SD-style VAE (--use-vae): the fusion output is the posterior mean μ, a logvar head
# is added, and the encoder returns a reparameterized sample z (so the Stage-2 VLA
# target is z, matching latent-diffusion practice). KL(N(0,I)) is weighted by a tiny
# --lambda-kl (1e-6, the SD regime) so reconstruction fidelity is preserved and the
# latent gets a mild scale regularization. The Stage-2 wrapper auto-detects the VAE
# from a checkpoint marker (_is_vae) and rebuilds the matching encoder.
#
# This codebase is hard-pinned to transformers==4.51.3 (GR00T eagle2.5 backbone),
# which predates DINOv3 — so we use the locally-cached facebook/dinov2-large
# (hidden_size=1024). Force HF offline so the cached weights load directly.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python scripts/train_action_latent_tokenizer_v4.py \
    --dataset-path "${DATA_DIR[@]}" \
    --output-dir $TOK_CKPT_DIR \
    --resume \
    --data-config dexjoco_single_arm_front \
    --embodiment-tag new_embodiment \
    --run-name "actlat_v4_dexjoco_single_arm_recon_dino_bn64_l1_mse_naiveln_vae" \
    --num-gpus 8 \
    --batch-size 64 \
    --max-steps $TOK_STEP \
    --save-steps 10000 \
    --save-total-limit 3 \
    --dataloader-num-workers 16 \
    --token-dim 64 \
    --encoder-depth 4 \
    --decoder-depth 4 \
    --dino-model "facebook/dinov2-large" \
    --dino-channels 1024 \
    --dino-final-norm naive \
    --fusion-width 1024 \
    --fusion-depth 6 \
    --dino-decoder-depth 6 \
    --lambda-recon 1.0 \
    --lambda-dino 0.1 \
    --use-vae \
    --lambda-kl 1e-6 \
    --recon-loss-type l1 \
    --dino-loss-type mse \
    --dino-w-l1 0.0 \
    --dino-w-mse 1.0 \
    --decoder-mode self_attention \
    --video-backend decord \
    --use-fixed-val \
    --wandb-project "Action-Tokenizer-dexjoco-single-arm-tokenizer" \
    --eval-steps 1000 \
    --report-to wandb


sbatch '/NHNHOME/data/wook/action-tokenizer/sbatch_scripts/dexjoco/v4_single/train_vla_recon_dino_bn64_l1_mse_naiveln_vae_fs.sh'
sbatch '/NHNHOME/data/wook/action-tokenizer/sbatch_scripts/dexjoco/v4_single/train_vla_recon_dino_bn64_l1_mse_naiveln_vae.sh'