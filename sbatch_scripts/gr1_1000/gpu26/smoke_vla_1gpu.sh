#!/bin/bash
#SBATCH --job-name=smoke_gr1_vla_1gpu
#SBATCH --partition=a100
#SBATCH --qos=background
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --time=1:30:00
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120
#SBATCH --output=/home/wook/action-tokenizer/slurm/logs_gr1_gpu26/%x_%j.out
#SBATCH --error=/home/wook/action-tokenizer/slurm/logs_gr1_gpu26/%x_%j.err

# 1-GPU smoke test for the gpu26 GR1 VLA pipeline (train_vla_base.sh port).
# Verifies: /s3data tar extraction -> /scratch, GR00T base download, dataloading,
# a few training steps, and HF Trainer checkpoint save directly onto /s3ckpt.

set -x

# --- storage-policy paths (checked by job_submit.lua) ---
S3_DATA_ROOT=/s3data/gr1-unified-lerobot/v1
CKPT_ROOT=/s3ckpt/$USER/smoke_gr1_vla_1gpu_0816_r5
SCRATCH_DATA=/scratch/$USER/gr1-unified-lerobot_v1

# gpu26/AWS S3 output: single switch for all S3-compat behaviors (ckpt staging,
# data-only copies, WANDB_DIR respect). Other servers never set this -> stock code.
export GR00T_S3_COMPAT=1
# scratch, then copytree to /s3ckpt (trainer honors this env; unset = stock path)
export GR00T_CKPT_STAGE_DIR=/scratch/$USER/ckpt_stage_smoke
# wandb append-writes must not land on /s3ckpt (script honors pre-set WANDB_DIR)
export WANDB_DIR=$HOME/wandb_runs
mkdir -p "$WANDB_DIR"

DS=gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000

export PATH="$HOME/.local/bin:$PATH"
source /home/wook/miniconda3/bin/activate gr00t-actlat
export PATH="$CONDA_PREFIX/bin:$PATH"
hash -r
echo "[env-check] which python=$(which python)"
python -c "import sys, torch; print('exe=', sys.executable, 'cuda=', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"

# smoke: no wandb account needed; offline run dir lands in home (allowed)
export WANDB_MODE=offline
export WANDB_PROJECT=Action-Tokenizer-GR1-gpu26-smoke

# --- Stage 0: extract one dataset's shards to node-local scratch (idempotent) ---
mkdir -p "$SCRATCH_DATA"
for part in data_meta videos; do
    tarpath="$S3_DATA_ROOT/$DS.$part.tar"
    marker="$SCRATCH_DATA/.done.$DS.$part"
    [ -f "$marker" ] && continue
    [ -f "$tarpath.ok" ] || { echo "[extract] missing .ok for $DS.$part — aborting"; exit 1; }
    time tar -xf "$tarpath" -C "$SCRATCH_DATA" && touch "$marker" \
        || { echo "[extract] FAILED: $DS.$part"; exit 1; }
    echo "[extract] done: $DS.$part"
done
ls "$SCRATCH_DATA/$DS" || exit 1

BASE_DIR=/home/wook/action-tokenizer
cd "$BASE_DIR"
mkdir -p "$CKPT_ROOT"

python scripts/gr00t_finetune_actlat_fm.py \
    --dataset-path "$SCRATCH_DATA/$DS" \
    --output-dir "$CKPT_ROOT" \
    --data-config fourier_gr1_arms_waist_actlat_fm \
    --embodiment-tag new_embodiment \
    --base-model-path "nvidia/GR00T-N1.5-3B" \
    --run-name "smoke_gr1_vla_1gpu_0816" \
    --mode "vla" \
    --num-gpus 1 \
    --batch-size 4 \
    --max-steps 30 \
    --save-steps 10 \
    --eval-steps 1000 \
    --val-ratio 0.003 \
    --use-fixed-val \
    --dataloader-num-workers 8 \
    --resume \
    --video-backend "decord"

echo "[smoke] training exited with $?"
echo "[smoke] checkpoints on /s3ckpt:"
ls -R "$CKPT_ROOT" | head -40
