#!/bin/bash
#SBATCH --job-name=gr1_sam3_split_smoke
#SBATCH --partition=a6000
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@300
#SBATCH --output=/home/wook/logs_alin/%x_%j.out
#SBATCH --error=/home/wook/logs_alin/%x_%j.err

# Smoke test for the alin-slurm GR1 SAM3 masking port: 1 GPU, LIMIT episodes
# (default 2), writing to a throwaway OUT_ROOT. Generated from
# sam3_gr1_split_array.sh by the sed recipe in this directory's git history, so the
# env, the flags and the data checks cannot drift from the real thing -- only the
# episode budget, the output root and the shard source differ.
#
#   sbatch sam3_gr1_split_smoke.sh                 # shard 0, 2 episodes
#   SHARD_ID=7 LIMIT=1 sbatch sam3_gr1_split_smoke.sh
#
# No --qos (only default_qos is granted to us) and no --array: this is one task.
# NUM_SHARDS stays 48 so the episode -> shard map matches the real run exactly,
# which means a smoke on shard N previews precisely what shard N will process.

export PYTHONUNBUFFERED=1
set -euxo pipefail

[ "$(hostname)" = "master" ] && { echo "[x] refusing to run on the login node"; exit 1; }

NUM_SHARDS=${NUM_SHARDS:-48}
SHARD_ID=${SHARD_ID:-${SLURM_ARRAY_TASK_ID:-0}}

# --- storage: everything under $HOME (lustre, shared with the compute nodes) ---
DATA_ROOT=${DATA_ROOT:-/home/wook/data/gr1_unified}
TAR_ROOT=${TAR_ROOT:-/home/wook/data/gr1_unified_tars}
OUT_ROOT=${OUT_ROOT:-/home/wook/outputs/gr1_sam3_split_smoke}

# --- SAM3 env (uv venv, transformers 5.x; built by analysis/sam3_masking/setup_sam3_env.sh) ---
SAM3_VENV=${SAM3_VENV:-/home/wook/venv_sam3}
SAM3_PY=$SAM3_VENV/bin/python
export HF_HOME=${HF_HOME:-/home/wook/.cache/huggingface}
export HF_HUB_OFFLINE=1
# ffmpeg is NOT uniformly installed on this cluster: of the 8 a6000 nodes only
# node3 and node6 have it (surveyed 2026-08-21; PATH is identical on all of them,
# the binary is simply absent on the other six). run_sam3.write_video() falls back
# to mp4v when ffmpeg is missing, so a videos-enabled run would silently write
# h264 on two nodes and mp4v on six -- heterogeneous output that nothing
# downstream would flag. Under --skip-videos no video is written at all, so it is
# moot; this guards the case where someone drops that flag.
SKIP_VIDEOS=${SKIP_VIDEOS:-1}
VIDEO_FLAGS=()
if [ "$SKIP_VIDEOS" = "1" ]; then
  VIDEO_FLAGS+=(--skip-videos)
elif ! command -v ffmpeg >/dev/null; then
  echo "[x] $(hostname) has no ffmpeg but SKIP_VIDEOS=0: write_video would silently"
  echo "    fall back to mp4v. Constrain the job to a node that has it (-w node3,node6)"
  echo "    or install ffmpeg into the venv first."
  exit 1
fi

BASE_DIR=/home/wook/action-tokenizer
cd "$BASE_DIR"
SAM3_SCRIPT=analysis/sam3_masking/batch_sam3_gr1_unified.py
[ -f "$SAM3_SCRIPT" ] || { echo "[x] missing $SAM3_SCRIPT"; exit 1; }

"$SAM3_PY" - <<'PYENV'
import torch, transformers
print("[env] torch", torch.__version__, "tf", transformers.__version__,
      "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
PYENV
echo "[disk] $HOME: $(df -h "$HOME" | tail -1)"
lfs quota -h -u "$USER" /mnt/lustre 2>/dev/null | sed -n 3p || true

# --- Stage 0: data in place. Extract only when the dataset dirs are absent and
# tars are present; on lustre that happens once globally, not once per node. ---
mkdir -p "$DATA_ROOT"
if [ -z "$(find "$DATA_ROOT" -maxdepth 1 -type d -name 'gr1_unified.*' -print -quit)" ] \
   && [ -d "$TAR_ROOT" ] && [ -n "$(find "$TAR_ROOT" -maxdepth 1 -name '*.tar' -print -quit)" ]; then
  (
    flock -x 200
    for tarpath in "$TAR_ROOT"/*.tar; do
      base=$(basename "$tarpath" .tar)
      marker="$DATA_ROOT/.done.$base"
      [ -f "$marker" ] && continue
      tar -xf "$tarpath" -C "$DATA_ROOT"
      touch "$marker"
      echo "[extract] done: $base"
    done
  ) 200>"$DATA_ROOT/.extract.lock"
fi

NDS=$(find "$DATA_ROOT" -maxdepth 1 -type d -name 'gr1_unified.*' | wc -l)
NVID=$(find "$DATA_ROOT" -name '*.mp4' ! -name '*.partial.mp4' | wc -l)
echo "[data-check] $NDS dataset dirs, $NVID episode videos (gpu26 sees 24 / 24000)"
[ "$NDS" -gt 0 ] || { echo "[x] no gr1_unified.* dirs under $DATA_ROOT"; exit 1; }
[ "$NVID" -gt 0 ] || { echo "[x] no episode videos under $DATA_ROOT"; exit 1; }

mkdir -p "$OUT_ROOT"

# --- SAM3 role-split masking for this shard (npz only) ---
srun --unbuffered "$SAM3_PY" "$SAM3_SCRIPT" \
    --data-glob "$DATA_ROOT/gr1_unified.*" \
    --out-root "$OUT_ROOT" \
    --model jetjodh/sam3 \
    --num-shards "$NUM_SHARDS" \
    --shard-id "$SHARD_ID" \
    --max-nouns 3 \
    --no-prompt-masks \
    --limit "${LIMIT:-2}" \
    ${VIDEO_FLAGS[@]+"${VIDEO_FLAGS[@]}"}

# The npz temp suffix is '.partial' (not '.npz'), so *.npz is already an honest
# progress count. Were videos re-enabled, count mp4 as: ! -name '*.partial.mp4'.
echo "[count] npz so far: $(find "$OUT_ROOT" -name '*.npz' | wc -l)"
echo "[done] gr1_split shard $SHARD_ID/$NUM_SHARDS"
