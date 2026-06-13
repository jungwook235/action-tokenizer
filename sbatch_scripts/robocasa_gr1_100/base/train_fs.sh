#!/bin/bash
#SBATCH --job-name=nactlat_fm_gr1_100demos_base_sbatch_fs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b200:2
#SBATCH --nodes=1
#SBATCH --qos=preempt
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120
#SBATCH --time=72:00:00
#SBATCH --output=/NHNHOME/data/wook/action-tokenizer/slurm/logs/nactlat_fm_gr1_100demos_base_sbatch_fs%j.out
#SBATCH --error=/NHNHOME/data/wook/action-tokenizer/slurm/logs/nactlat_fm_gr1_100demos_base_sbatch_fs%j.err

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

CKPT_DIR="checkpoints/vla_nactlat_fm_gr1_100demos/base_fs"
# Glob expands to all 24 gr1_unified.* dataset dirs (each a LeRobot dataset root)

python scripts/gr00t_finetune_actlat_fm.py \
    --dataset-path "/NHNHOME/data/wook/dataset/robocasa_gr1_tabletop/sim_100demos" \
    --output-dir $CKPT_DIR \
    --data-config fourier_gr1_arms_waist_actlat_fm \
    --embodiment-tag new_embodiment \
    --base-model-path "nvidia/GR00T-N1.5-3B" \
    --run-name "nactlat_baseline_gr1_100demos_fs" \
    --mode "vla" \
    --num-gpus 2 \
    --batch-size 32 \
    --max-steps 60000 \
    --save-steps 5000 \
    --save-total-limit 3 \
    --eval-steps 1000 \
    --val-ratio 0.003 \
    --use-fixed-val \
    --no-load-action-head \
    --video-backend "decord"
