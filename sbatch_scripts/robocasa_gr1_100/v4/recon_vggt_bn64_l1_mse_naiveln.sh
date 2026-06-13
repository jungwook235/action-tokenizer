#!/bin/bash
#SBATCH --job-name=full_v4_gr1_recon_vggt_bn64_l1_mse_naiveln
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b200:4
#SBATCH --nodes=1
#SBATCH --qos=preempt
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120
#SBATCH --time=72:00:00
#SBATCH --output=/NHNHOME/data/wook/action-tokenizer/slurm/logs/full_v4_gr1_recon_vggt_bn64_l1_mse_naiveln_%j.out
#SBATCH --error=/NHNHOME/data/wook/action-tokenizer/slurm/logs/full_v4_gr1_recon_vggt_bn64_l1_mse_naiveln_%j.err


set -ex  # -e: abort the whole job (skip Stage-2) if Stage-1 fails
# Unbuffer stdout/stderr so HF Trainer's loss logs flush to the Slurm .out file in
# real time. Without this, stdout is block-buffered when redirected to a file, so
# loss lines only appear in large delayed bursts (~every 550 steps).
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"
export WANDB_PROJECT=Action-Tokenizer-GR1-100demos
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
DATA_DIR=/NHNHOME/data/wook/dataset/robocasa_gr1_tabletop/sim_100demos
TOK_CKPT_DIR=checkpoints_action_tokenizer/gr1_100demos_v4_recon_vggt_bn64_l1_mse_naiveln
VLA_CKPT_DIR=checkpoints/vla_actlat_fm_gr1_100demos/v4_recon_vggt_bn64_l1_mse_naiveln_fs
TOK_STEP=100000
ABS_TOK_CKPT="/NHNHOME/data/wook/action-tokenizer/$TOK_CKPT_DIR"

# === Stage 1: V4 Tokenizer Training — RLA-VGGT hybrid, L1 recon, NAIVE final LayerNorm ===
# Same as recon_vggt_bn64_l1_mse.sh, except the VGGT patch features get an extra
# NON-AFFINE final LayerNorm (--vggt-final-norm naive): a plain (x-mean)/std (no
# learned γ/β) is applied on top of the final VGGT token features. The Stage-2
# wrapper auto-detects this from a checkpoint marker and rebuilds a matching extractor.
#
# This variant replaces the DINO visual feature with VGGT patch tokens
# (--feature-source vggt). The token source is the point head's DPT layer-2
# feature at patch-grid resolution (--vggt-token-source dpt_out2 → 1024-d), so
# --dino-channels / --fusion-width stay 1024 (1:1 shape swap for DINO-large).
# Alternative: --vggt-token-source aggregator (set --dino-channels 2048).
#
# VGGT weights are loaded from the in-repo vggt/ source via huggingface_hub
# (VGGT.from_pretrained). The checkpoint (facebook/VGGT-1B, ~5GB) must be
# downloadable or already cached — so we do NOT force HF offline here (unlike the
# DINO variant, which used cached dinov2-large). If VGGT-1B is gated, use
# facebook/VGGT-1B-Commercial.
python scripts/train_action_latent_tokenizer_v4.py \
    --dataset-path "$DATA_DIR" \
    --output-dir $TOK_CKPT_DIR \
    --no-resume \
    --data-config fourier_gr1_arms_waist \
    --embodiment-tag new_embodiment \
    --run-name "actlat_v4_gr1_recon_vggt_bn64_l1_mse_naiveln_100demos" \
    --num-gpus 4 \
    --batch-size 64 \
    --max-steps $TOK_STEP \
    --save-steps 10000 \
    --save-total-limit 3 \
    --dataloader-num-workers 16 \
    --token-dim 64 \
    --encoder-depth 4 \
    --decoder-depth 4 \
    --feature-source vggt \
    --vggt-token-source dpt_out2 \
    --vggt-model "facebook/VGGT-1B" \
    --vggt-image-size 224 \
    --vggt-final-norm naive \
    --dino-channels 1024 \
    --fusion-width 1024 \
    --fusion-depth 6 \
    --dino-decoder-depth 6 \
    --lambda-recon 1.0 \
    --lambda-dino 1.0 \
    --recon-loss-type l1 \
    --dino-loss-type mse \
    --dino-w-l1 0.0 \
    --dino-w-mse 1.0 \
    --decoder-mode self_attention \
    --video-backend decord \
    --use-fixed-val \
    --wandb-project "Action-Tokenizer-GR1-100demos-tokenizer" \
    --eval-steps 1000 \
    --report-to wandb

# === Stage 2: VLA Training ===
# The wrapper auto-detects the VGGT feature source AND the naive final-norm from the
# tokenizer checkpoint (recorded markers) and rebuilds the matching frozen VGGT
# extractor; frames flow through unchanged.
python scripts/gr00t_finetune_actlat_fm.py \
    --dataset-path "$DATA_DIR" \
    --output-dir $VLA_CKPT_DIR \
    --data-config fourier_gr1_arms_waist_actlat_fm \
    --embodiment-tag new_embodiment \
    --base-model-path "nvidia/GR00T-N1.5-3B" \
    --run-name "actlat_fm_v4_gr1_recon_vggt_bn64_l1_mse_naiveln_fs_100demos" \
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
