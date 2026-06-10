#!/bin/bash
#SBATCH --job-name=full_v4_gr1_recon_dino_bn64_cos
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b200:4
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --output=/NHNHOME/data/wook/action-tokenizer/slurm/logs/full_v4_gr1_recon_dino_bn64_cos_%j.out
#SBATCH --error=/NHNHOME/data/wook/action-tokenizer/slurm/logs/full_v4_gr1_recon_dino_bn64_cos_%j.err


set -ex  # -e: abort the whole job (skip Stage-2) if Stage-1 fails
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
TOK_CKPT_DIR=checkpoints_action_tokenizer/gr1_100demos_v4_recon_dino_bn64_cos
VLA_CKPT_DIR=checkpoints/vla_actlat_fm_gr1_100demos/v4_recon_dino_bn64_cos_fs
TOK_STEP=100000
ABS_TOK_CKPT="/NHNHOME/data/wook/action-tokenizer/$TOK_CKPT_DIR"

# === Stage 1: V4 Tokenizer Training — RLA-DINO hybrid (robocasa) ===
# NOTE: This codebase is hard-pinned to transformers==4.51.3 (GR00T eagle2.5
# backbone), which predates DINOv3 support — so we use the locally-cached
# facebook/dinov2-large (also hidden_size=1024). Force HF offline so the cached
# weights load directly (no network / token). To switch to real DINOv3 later you
# need a transformers>=4.57 env that does NOT also run GR00T (eagle breaks there).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python scripts/train_action_latent_tokenizer_v4.py \
    --dataset-path "$DATA_DIR" \
    --output-dir $TOK_CKPT_DIR \
    --no-resume \
    --data-config fourier_gr1_arms_waist \
    --embodiment-tag new_embodiment \
    --run-name "actlat_v4_gr1_recon_dino_bn64_cos_100demos" \
    --num-gpus 4 \
    --batch-size 128 \
    --max-steps $TOK_STEP \
    --save-steps 10000 \
    --save-total-limit 3 \
    --dataloader-num-workers 16 \
    --token-dim 64 \
    --encoder-depth 4 \
    --decoder-depth 4 \
    --dino-model "facebook/dinov2-large" \
    --dino-channels 1024 \
    --fusion-width 1024 \
    --fusion-depth 6 \
    --dino-decoder-depth 6 \
    --lambda-recon 1.0 \
    --lambda-dino 1.0 \
    --recon-loss-type mse \
    --dino-loss-type cosine \
    --dino-w-l1 0.0 \
    --dino-w-mse 0.0 \
    --dino-w-cosine 1.0 \
    --decoder-mode self_attention \
    --video-backend decord \
    --use-fixed-val \
    --wandb-project "Action-Tokenizer-GR1-100demos-tokenizer" \
    --eval-steps 1000 \
    --report-to wandb

# Re-enable HF network for Stage-2 (GR00T base / eagle may need download).
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

# === Stage 2: VLA Training ===
python scripts/gr00t_finetune_actlat_fm.py \
    --dataset-path "$DATA_DIR" \
    --output-dir $VLA_CKPT_DIR \
    --data-config fourier_gr1_arms_waist_actlat_fm \
    --embodiment-tag new_embodiment \
    --base-model-path "nvidia/GR00T-N1.5-3B" \
    --run-name "actlat_fm_v4_gr1_recon_dino_bn64_cos_fs_100demos" \
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
