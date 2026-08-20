#!/bin/bash
#SBATCH --job-name=gr1_sam3_split
#SBATCH --partition=a100
#SBATCH --qos=background
#SBATCH --array=2-9,16-47%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@300
#SBATCH --output=/home/wook/logs_gpu26/%x_%A_%a.out
#SBATCH --error=/home/wook/logs_gpu26/%x_%A_%a.err

# GR1 unified SAM3 masking with role-split npz (robot_mask / object_mask /
# prompt_roles), 48 shards (episode j -> shard j % 48). Same prompt combo as the
# existing ..._sam3_D_parts_nouns_norobot output, so the union mask — and therefore
# the cutout — is unchanged; only the npz gains the role split.
#
# --skip-videos: the cutout/overlay mp4s already exist from the previous run, so
# this pass writes ONLY the npz. That removes the entire mp4 encoding path (2 cv2
# writes + 2 ffmpeg re-encodes = ~9.8 ms/frame, measured) and makes the skip check
# look at the npz alone, which keeps resume exact.
#
# a100 + background QoS only (wook's guaranteed partition is h200).
#
# 2026-08-20 respin of job 670 at the user's request: leave more a100 capacity for
# the other researchers, so the throttle drops 8 -> %2 (at most 2 concurrent tasks).
# Job 670 kept tasks 16/17 running and had 18/19 + all pending tasks cancelled; this
# array covers every shard that was NOT verified COMPLETED under 670.
#   COMPLETED under 670 (excluded here): 0, 1, 10-15
#   included: 2-9 (never started), 16-47 (16/17 as a no-op safety net in case they
#             do not finish cleanly, 18/19 partial, 20-47 never started)
# Submit AFTER 670_16/670_17 drain, otherwise 4 GPUs stay busy:
#   sbatch --dependency=afterany:670_16:670_17 sam3_gr1_split_array_t2.sh
#
# NUM_SHARDS stays 48 -- it defines the episode->shard map (j % 48). Changing it
# would repartition every shard and silently break resume coverage.

export PYTHONUNBUFFERED=1
set -euxo pipefail

NUM_SHARDS=48
SHARD_ID=${SLURM_ARRAY_TASK_ID:?this script must run as a job array}

# --- storage contract (submit filter checks these literals) ---
S3_DATA_ROOT=/s3data/gr1-unified-lerobot/v1
SCRATCH_DATA=/scratch/$USER/gr1_unified      # same path+markers the udec training
OUT_ROOT=/ckpt/$USER/gr1_sam3_split          # jobs use, so an existing extraction
                                             # on this node is reused as-is

# --- SAM3 env (shared with the ActionNet run: transformers 5.x venv) ---
SAM3_VENV=/home/wook/venv_sam3
SAM3_PY=$SAM3_VENV/bin/python
export HF_HOME=/home/wook/.cache/huggingface
export HF_HUB_OFFLINE=1
# ffmpeg is not needed with --skip-videos, but keep it reachable so a future run
# without the flag does not silently fall back to mp4v.
export PATH="$PATH:/home/wook/miniconda3/envs/gr00t-actlat/bin"

BASE_DIR=/home/wook/action-tokenizer
cd "$BASE_DIR"
SAM3_SCRIPT=analysis/sam3_masking/batch_sam3_gr1_unified.py
[ -f "$SAM3_SCRIPT" ] || { echo "[x] missing $SAM3_SCRIPT"; exit 1; }

echo "[env] $($SAM3_PY -c 'import torch,transformers;print("torch",torch.__version__,"tf",transformers.__version__,"cuda",torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")')"
echo "[disk] /ckpt: $(df -h /ckpt | tail -1)"
echo "[disk] /scratch: $(df -h /scratch | tail -1)"

# --- Stage 0: extract the 48 GR1 tars ONCE per node (39 GB; shard boundaries do
# not align with tars, so every shard needs the whole set locally) ---
mkdir -p "$SCRATCH_DATA"
(
  flock -x 200
  for tarpath in "$S3_DATA_ROOT"/*.tar; do
    base=$(basename "$tarpath" .tar)
    marker="$SCRATCH_DATA/.done.$base"
    [ -f "$marker" ] && continue
    [ -f "$tarpath.ok" ] || { echo "[extract] missing .ok for $base"; exit 1; }
    tar -xf "$tarpath" -C "$SCRATCH_DATA"
    touch "$marker"
    echo "[extract] done: $base"
  done
) 200>"$SCRATCH_DATA/.extract.lock"

NDS=$(find "$SCRATCH_DATA" -maxdepth 1 -type d -name 'gr1_unified.*' | wc -l)
NVID=$(find "$SCRATCH_DATA" -name '*.mp4' | wc -l)
echo "[data-check] $NDS dataset dirs, $NVID episode videos (expect 24 / 24000)"
[ "$NDS" -eq 24 ] && [ "$NVID" -eq 24000 ]

mkdir -p "$OUT_ROOT"

# --- SAM3 role-split masking for this shard (npz only) ---
srun --unbuffered "$SAM3_PY" "$SAM3_SCRIPT" \
    --data-glob "$SCRATCH_DATA/gr1_unified.*" \
    --out-root "$OUT_ROOT" \
    --model jetjodh/sam3 \
    --num-shards "$NUM_SHARDS" \
    --shard-id "$SHARD_ID" \
    --max-nouns 3 \
    --skip-videos \
    --no-prompt-masks

echo "[done] gr1_split shard $SHARD_ID/$NUM_SHARDS"
