#!/bin/bash
#SBATCH --job-name=gr00t_v2_headfix_recon_gr1
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=sjw_alinlab,h100
#SBATCH --output=out/%j-gr00t_v2_headfix_recon_gr1.out
#SBATCH --error=out/%j-gr00t_v2_headfix_recon_gr1.err
#SBATCH --comment "gr00t_v2_headfix_recon_gr1"

# Post-headfix VLA training. The tokenizer wrapper bug (head_dim auto-detect
# 32 instead of training default 64) was fixed in 2026-04-29
# (see experiments/verification/error_notes.md). All v2_headfix_* VLAs are
# retrained from scratch on top of correctly-loaded tokenizers.
#
# gr1 recon: gr1 has no V2 recon-only tokenizer on disk yet, so this script
# trains it (Stage 1) before VLA training (Stage 2). Other gr1 v2_headfix
# scripts reuse existing tokenizer checkpoints.

set -x
export PATH="$HOME/.local/bin:$PATH"
export WANDB_API_KEY="66a73856614bc24a07523f3fbee42482fcbeffe3"
export HF_TOKEN="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"

source /sjw_alinlab1/home/jungwook/miniconda3/bin/activate gr00t

DATA_DIR=/storage1/sjw_dataset/dataset/robocasa_gr1_tabletop/sim_100demos
TOK_CKPT_DIR=checkpoints_action_tokenizer/gr1_100demos_v2_base
VLA_CKPT_DIR=checkpoints/vla_actlat_fm_gr1_100demos/v2_headfix_recon
TOK_STEP=100000
ABS_TOK_CKPT="/sjw_alinlab1/home/jungwook/Isaac-GR00T/$TOK_CKPT_DIR"

# === Stage 1: Tokenizer Training (V2 base — recon only) ===
# Recon-only baseline: lambda-recon=1.0, all other losses = 0.
python scripts/train_action_latent_tokenizer_v2.py \
    --dataset-path $DATA_DIR \
    --output-dir $TOK_CKPT_DIR \
    --no-resume \
    --data-config fourier_gr1_arms_waist \
    --embodiment-tag new_embodiment \
    --run-name "actlat_v2_base_gr1_100demos" \
    --num-gpus 2 \
    --batch-size 1024 \
    --max-steps $TOK_STEP \
    --save-steps 5000 \
    --num-global-tokens 0 \
    --num-hand-tokens 0 \
    --lambda-recon 1.0 \
    --recon-loss-type mse \
    --decoder-mode self_attention \
    --wandb-project "action-latent-tokenizer-v2" \
    --eval-steps 1000 \
    --report-to wandb

# === Stage 2: VLA Training ===
# num-hand-tokens=0 → target_tokens="all" effectively means time tokens only.
python scripts/gr00t_finetune_actlat_fm.py \
    --dataset-path $DATA_DIR \
    --output-dir $VLA_CKPT_DIR \
    --data-config fourier_gr1_arms_waist_actlat_fm \
    --embodiment-tag new_embodiment \
    --base-model-path "nvidia/GR00T-N1.5-3B" \
    --run-name "actlat_fm_v2_headfix_recon_gr1_100demos" \
    --num-gpus 2 \
    --batch-size 32 \
    --max-steps 60000 \
    --save-steps 10000 \
    --eval-steps 1000 \
    --actlat-tokenizer-path "$ABS_TOK_CKPT/checkpoint-$TOK_STEP" \
    --actlat-target-tokens "all" \
    --val-ratio 0.003 \
    --video-backend "decord"
