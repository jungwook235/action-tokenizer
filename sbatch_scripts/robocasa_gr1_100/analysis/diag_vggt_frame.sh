#!/bin/bash
#SBATCH --job-name=diag_vggt_frame_diff_no_ln
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --nodes=1
#SBATCH --qos=preempt
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120
#SBATCH --time=72:00:00
#SBATCH --output=/NHNHOME/data/wook/action-tokenizer/slurm/logs/diag_vggt_frame_diff_no_ln%j.out
#SBATCH --error=/NHNHOME/data/wook/action-tokenizer/slurm/logs/diag_vggt_frame_diff_no_ln%j.err

set -x
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

PYTHONUNBUFFERED=1 python scripts/diag_vggt_frame_diff.py \
    --dataset-path /NHNHOME/data/wook/dataset/robocasa_gr1_tabletop/sim_100demos \
    --data-config fourier_gr1_arms_waist \
    --embodiment-tag new_embodiment \
    --vggt-final-norm none \
    --num-samples 512 --batch-size 32