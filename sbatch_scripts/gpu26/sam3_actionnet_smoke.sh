#!/bin/bash
#SBATCH --job-name=an_sam3_smoke
#SBATCH --partition=a100
#SBATCH --qos=background
#SBATCH --array=0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@300
#SBATCH --output=/home/wook/logs_gpu26/%x_%A_%a.out
#SBATCH --error=/home/wook/logs_gpu26/%x_%A_%a.err

# ActionNet SAM3 masking, 48 shards (episode j -> shard j % 48).
# a100 + background QoS (wook's guaranteed partition is h200, so a100 can only be
# background); the array throttle %8 matches the per-user background cap of 8 GPUs,
# so 8 shards run at a time and the rest queue.
#
# Resumability: a shard skips every episode whose three artifacts already exist, so
# the SAME array can be resubmitted after a preemption or the 24 h TIMEOUT (which
# Slurm does NOT requeue for background) and it continues where it stopped.
#
# Storage contract: read /s3data (read-only tars) -> extract once per node into
# /scratch (node-local, flock-guarded) -> write /ckpt/$USER (real POSIX FS,
# auto-archived to S3). Logs stay in home.

export PYTHONUNBUFFERED=1
set -euxo pipefail

NUM_SHARDS=48
SHARD_ID=${SLURM_ARRAY_TASK_ID:?this script must run as a job array}

# --- storage contract (submit filter checks these literals) ---
S3_DATA_ROOT=/s3data/gr1-actionnet-lerobot-15fps/v1
SCRATCH_DATA=/scratch/$USER/actionnet_15fps
OUT_ROOT=/ckpt/$USER/actionnet_sam3_smoke
DS_DIR_NAME=gr1_actionnet_lerobot_15fps

# --- SAM3 env (own venv: transformers 5.x; the training env is pinned to 4.51.3) ---
SAM3_VENV=/home/wook/venv_sam3
SAM3_PY=$SAM3_VENV/bin/python
export HF_HOME=/home/wook/.cache/huggingface
export HF_HUB_OFFLINE=1          # weights pre-cached: 48 shards must not hit the hub
# run_sam3.write_video() shells out to ffmpeg (mp4v -> h264). No sudo here, so use
# the ffmpeg that ships in the training conda env; APPENDED so nothing is shadowed.
export PATH="$PATH:/home/wook/miniconda3/envs/gr00t-actlat/bin"
command -v ffmpeg >/dev/null || { echo "[x] ffmpeg not on PATH"; exit 1; }

BASE_DIR=/home/wook/action-tokenizer
cd "$BASE_DIR"
SAM3_SCRIPT="${SAM3_SCRIPT:-analysis/sam3_masking/batch_sam3_actionnet.py}"
[ -f "$SAM3_SCRIPT" ] || { echo "[x] missing $SAM3_SCRIPT (set SAM3_SCRIPT=...)"; exit 1; }

echo "[env] $($SAM3_PY -c 'import torch,transformers;print("torch",torch.__version__,"tf",transformers.__version__,"cuda",torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")')"
echo "[disk] /ckpt: $(df -h /ckpt | tail -1)"
echo "[disk] /scratch: $(df -h /scratch | tail -1)"

# --- Stage 0: extract the 15 data tars ONCE per node (all 8 shards share them) ---
# Shard boundaries (j % 48) do not align with tar boundaries, so every shard needs
# the whole dataset locally. flock serializes the first arrival; the rest wait and
# then skip via the .done markers.
mkdir -p "$SCRATCH_DATA"
(
  flock -x 200
  for tarpath in "$S3_DATA_ROOT"/*.tar; do
    base=$(basename "$tarpath" .tar)
    marker="$SCRATCH_DATA/.done.$base"
    [ -f "$marker" ] && continue
    tar -xf "$tarpath" -C "$SCRATCH_DATA"
    touch "$marker"
    echo "[extract] done: $base"
  done
) 200>"$SCRATCH_DATA/.extract.lock"

DATA_ROOT="$SCRATCH_DATA/$DS_DIR_NAME"
[ -d "$DATA_ROOT/videos" ] || { echo "[x] extraction incomplete: no $DATA_ROOT/videos"; exit 1; }
NVID=$(find "$DATA_ROOT/videos" -name '*.mp4' | wc -l)
echo "[data-check] $NVID episode videos under $DATA_ROOT (expect 29968)"
[ "$NVID" -eq 29968 ]

mkdir -p "$OUT_ROOT"

# --- SAM3 masking for this shard ---
# Flags mirror batch_sam3_robot_task.py's production CLI; adjust only if the
# ActionNet script renames them.
srun --unbuffered "$SAM3_PY" "$SAM3_SCRIPT" \
    --root "$DATA_ROOT" \
    --out-root "$OUT_ROOT" \
    --model jetjodh/sam3 \
    --num-shards "$NUM_SHARDS" \
    --shard-id "$SHARD_ID" \
    --max-nouns 3 \
    --cutout-from union \
    --bg green \
    --no-prompt-masks \
    --episodes 8084 15316 19262

nvidia-smi --query-gpu=memory.used,memory.total --format=csv || true
du -sh "$OUT_ROOT" 2>/dev/null || true
find "$OUT_ROOT" -type f -printf "%s %p\n" 2>/dev/null | sort -n | tail -9
echo "[done] shard $SHARD_ID/$NUM_SHARDS"
