#!/bin/bash
#SBATCH --job-name=gr1_1000_vla_base
#SBATCH --partition=h200
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120
#SBATCH --output=/home/wook/action-tokenizer/slurm/logs_gr1_gpu26/%x_%j.out
#SBATCH --error=/home/wook/action-tokenizer/slurm/logs_gr1_gpu26/%x_%j.err

# gpu26 port of mlxp gr1_1000/base/train.sh (VLA baseline, --mode vla).
# Storage contract: read tar shards from /s3data, extract to node-local /scratch,
# write checkpoints to /s3ckpt/$USER (submit filter requires both path literals).

set -x

# --- storage-policy paths (checked by job_submit.lua) ---
S3_DATA_ROOT=/s3data/gr1-unified-lerobot/v1
CKPT_ROOT=/s3ckpt/$USER/gr1_1000_vla_base          # unique experiment name = dir name
SCRATCH_DATA=/scratch/$USER/gr1-unified-lerobot_v1

# gpu26/AWS S3 output: single switch for all S3-compat behaviors (ckpt staging,
# data-only copies, WANDB_DIR respect). Other servers never set this -> stock code.
export GR00T_S3_COMPAT=1
# node-local scratch, then data-only copy to /s3ckpt (verified: smoke job 48)
export GR00T_CKPT_STAGE_DIR=/scratch/$USER/ckpt_stage_gr1_vla_base
# wandb append-writes must not land on /s3ckpt (script honors pre-set WANDB_DIR)
export WANDB_DIR=$HOME/wandb_runs
mkdir -p "$WANDB_DIR"

export PATH="$HOME/.local/bin:$PATH"
source /home/wook/miniconda3/bin/activate gr00t-actlat
export PATH="$CONDA_PREFIX/bin:$PATH"
hash -r
echo "[env-check] which python=$(which python)"
python -c "import sys, transformers; print('exe=', sys.executable, 'transformers=', transformers.__version__)"

# secrets (wandb) live in home, not in this script; logs stay in home (no /s3ckpt appends)
[ -f "$HOME/.secrets.env" ] && source "$HOME/.secrets.env"
export WANDB_PROJECT=Action-Tokenizer-GR1-1000demos

# --- Stage 0: extract /s3data tar shards -> /scratch (idempotent; markers survive
# requeue on the same node, and the s3 NVMe read-cache makes re-extraction cheap) ---
mkdir -p "$SCRATCH_DATA"
for tarpath in "$S3_DATA_ROOT"/*.tar; do
    base=$(basename "$tarpath" .tar)
    marker="$SCRATCH_DATA/.done.$base"
    [ -f "$marker" ] && continue
    [ -f "$tarpath.ok" ] || { echo "[extract] missing .ok for $base — aborting"; exit 1; }
    tar -xf "$tarpath" -C "$SCRATCH_DATA" && touch "$marker" \
        || { echo "[extract] FAILED: $base"; exit 1; }
    echo "[extract] done: $base"
done

DATA_DIR=("$SCRATCH_DATA"/gr1_unified.*/)
echo "[data-check] ${#DATA_DIR[@]} dataset roots under $SCRATCH_DATA (expect 24)"
[ "${#DATA_DIR[@]}" -eq 24 ] || { echo "[data-check] unexpected dataset count"; exit 1; }

BASE_DIR=/home/wook/action-tokenizer
cd "$BASE_DIR"
mkdir -p "$CKPT_ROOT"

# --- Stage 2 only (VLA baseline; no tokenizer in the base experiment) ---
python scripts/gr00t_finetune_actlat_fm.py \
    --dataset-path "${DATA_DIR[@]}" \
    --output-dir "$CKPT_ROOT" \
    --data-config fourier_gr1_arms_waist_actlat_fm \
    --embodiment-tag new_embodiment \
    --base-model-path "nvidia/GR00T-N1.5-3B" \
    --run-name "gr1_1000_vla_base_gpu26" \
    --mode "vla" \
    --num-gpus 4 \
    --batch-size 128 \
    --max-steps 60000 \
    --save-steps 5000 \
    --eval-steps 1000 \
    --val-ratio 0.003 \
    --use-fixed-val \
    --resume \
    --video-backend "decord"

# NOTE (vs mlxp original): --save-total-limit 3 removed on purpose — checkpoint
# rotation deletes old dirs (shutil.rmtree), and /s3ckpt forbids deletion, so it
# would crash the run. Old checkpoints simply accumulate (cleanup = operator request).
