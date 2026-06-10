#!/bin/bash
#SBATCH --job-name=full_v4_gr1_recon_dino_bn64
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b200:4
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --output=/NHNHOME/data/wook/action-tokenizer/slurm/logs/full_v4_gr1_recon_dino_bn64%j.out
#SBATCH --error=/NHNHOME/data/wook/action-tokenizer/slurm/logs/full_v4_gr1_recon_dino_bn64%j.err


set -ex  # -e: abort the whole job (skip Stage-2) if Stage-1 fails
export PATH="$HOME/.local/bin:$PATH"
export WANDB_PROJECT=Action-Tokenizer-GR1-1000demos
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

DATA_DIR=(/NHNHOME/data/wook/dataset/gr00t_unified/gr1_unified.*)
TOK_CKPT_DIR=checkpoints_action_tokenizer/gr1_1000demos_v4_recon_dino_bn64
VLA_CKPT_DIR=checkpoints/vla_actlat_fm_gr1_1000demos/v4_recon_dino_bn64_fs
TOK_STEP=100000
ABS_TOK_CKPT="/NHNHOME/data/wook/action-tokenizer/$TOK_CKPT_DIR"

# === Stage 1: V4 Tokenizer Training — RLA-DINO hybrid ===
# V3 action encoder (256) produces action latents that replace RLA's learnable
# query tokens; they are fused with DINO feature-difference (x1 - x0) tokens in an
# RLA SimpleTokenTransformer whose out_layer bottlenecks to token_dim=64. The 64-dim
# latent feeds BOTH an action recon decoder and a DINO future-feature decoder.
#   - lambda_recon / lambda_dino : per-loss weights (default 1.0 each)
#   - dino_loss_type             : l1 | mse | cosine | "l1+cosine" ...
#   - DINOv3-Large (1024) frozen, extracted on-the-fly (--no-cache-dataset)
# batch-size reduced vs v3 (512) due to DINO memory.
python scripts/train_action_latent_tokenizer_v4.py \
    --dataset-path "${DATA_DIR[@]}" \
    --output-dir $TOK_CKPT_DIR \
    --no-resume \
    --data-config fourier_gr1_arms_waist \
    --embodiment-tag new_embodiment \
    --run-name "actlat_v4_gr1_recon_dino_bn64_1000demos" \
    --num-gpus 4 \
    --batch-size 64 \
    --max-steps $TOK_STEP \
    --save-steps 10000 \
    --save-total-limit 3 \
    --dataloader-num-workers 16 \
    --token-dim 64 \
    --dino-model "facebook/dinov3-vitl16-pretrain-lvd1689m" \
    --dino-channels 1024 \
    --fusion-width 1024 \
    --fusion-depth 12 \
    --dino-decoder-depth 12 \
    --lambda-recon 1.0 \
    --lambda-dino 1.0 \
    --recon-loss-type mse \
    --dino-loss-type l1+mse \
    --dino-w-l1 1.0 \
    --dino-w-mse 1.0 \
    --decoder-mode self_attention \
    --video-backend decord \
    --use-fixed-val \
    --wandb-project "Action-Tokenizer-GR1-1000demos-tokenizer" \
    --eval-steps 1000 \
    --report-to wandb

# === Stage 2: VLA Training ===
# V4 latent targets are DINO-dependent, so --actlat-frames makes the dataset
# yield (frame_x0, frame_x1) and the model thread them into get_latent_target.
# The wrapper owns a frozen DINO extractor (built from the tokenizer's dino_dim).
# Inference is decode-only and needs no frames.
python scripts/gr00t_finetune_actlat_fm.py \
    --dataset-path "${DATA_DIR[@]}" \
    --output-dir $VLA_CKPT_DIR \
    --data-config fourier_gr1_arms_waist_actlat_fm \
    --embodiment-tag new_embodiment \
    --base-model-path "nvidia/GR00T-N1.5-3B" \
    --run-name "actlat_fm_v4_gr1_recon_dino_bn64_fs_1000demos" \
    --num-gpus 4 \
    --batch-size 128 \
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
