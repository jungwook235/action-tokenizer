#!/bin/bash
#SBATCH --job-name=full_v4_gr1_recon_vggt_bn64_l1_mse_naiveln_vae_1000tok_ft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b200:8
#SBATCH --nodes=1
#SBATCH --qos=preempt
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120
#SBATCH --time=72:00:00
#SBATCH --output=/NHNHOME/data/wook/action-tokenizer/slurm/logs/full_v4_gr1_recon_vggt_bn64_l1_mse_naiveln_vae_1000tok_ft_%j.out
#SBATCH --error=/NHNHOME/data/wook/action-tokenizer/slurm/logs/full_v4_gr1_recon_vggt_bn64_l1_mse_naiveln_vae_1000tok_ft_%j.err


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
# Override state/action normalization stats with the unified-dataset stats instead
# of computing them from DATA_DIR. Picked up by gr00t/data/dataset.py::_get_metadata.
export GR00T_STATS_OVERRIDE=/NHNHOME/data/wook/dataset/gr00t_unified/stats.json
TOK_CKPT_DIR=checkpoints_action_tokenizer/gr1_1000demos_v4_recon_dino_bn64_l1_mse_naiveln_vae
VLA_CKPT_DIR=checkpoints/vla_actlat_fm_gr1_100demos/v4_recon_vggt_bn64_l1_mse_naiveln_vae_1000tok_ft
TOK_STEP=100000
ABS_TOK_CKPT="/NHNHOME/data/wook/action-tokenizer/$TOK_CKPT_DIR"


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
    --run-name "actlat_fm_v4_gr1_recon_vggt_bn64_l1_mse_naiveln_vae_1000tok_ft_100demos" \
    --num-gpus 8 \
    --batch-size 8 \
    --max-steps 60000 \
    --save-steps 5000 \
    --save-total-limit 3 \
    --eval-steps 1000 \
    --actlat-tokenizer-path "$ABS_TOK_CKPT/checkpoint-$TOK_STEP" \
    --actlat-target-tokens "all" \
    --actlat-frames \
    --val-ratio 0.003 \
    --use-fixed-val \
    --video-backend "decord"
