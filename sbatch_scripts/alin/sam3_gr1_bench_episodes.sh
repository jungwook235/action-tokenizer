#!/bin/bash
#SBATCH --job-name=gr1_sam3_bench
#SBATCH --partition=a6000
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --output=/home/wook/logs_alin/%x_%j.out
#SBATCH --error=/home/wook/logs_alin/%x_%j.err

# Throughput/memory benchmark for GR1 SAM3 masking on a6000, to size how many
# shards this cluster should take off gpu26.
#
# Explicitly NOT a shard run. It uses --episodes (which makes
# batch_sam3_gr1_unified.py ignore sharding entirely) against ONE dataset dir,
# because the transferred tree is still incomplete: the global episode
# enumeration that `episode j -> shard j % 48` depends on would not agree with
# gpu26 until every dataset has landed, so a shard-mode run now would mask the
# wrong episodes.
#
# Flags match the gpu26 production run (--max-nouns 3 --skip-videos
# --no-prompt-masks) so the seconds-per-frame are comparable to its 0.232 s/frame
# on a100. Output goes to a throwaway root keyed by job id -- it must never touch
# the real output tree.

export PYTHONUNBUFFERED=1
set -euxo pipefail

[ "$(hostname)" = "master" ] && { echo "[x] refusing to run on the login node"; exit 1; }

DATASET=${DATASET:?set DATASET to one complete gr1_unified.* dir}
EPISODES=${EPISODES:-0 1 2}
OUT_ROOT=${OUT_ROOT:-/home/wook/outputs/smoke_${SLURM_JOB_ID:-manual}}

SAM3_VENV=${SAM3_VENV:-/home/wook/venv_sam3}
SAM3_PY=$SAM3_VENV/bin/python
export HF_HOME=${HF_HOME:-/home/wook/.cache/huggingface}
export HF_HUB_OFFLINE=1

BASE_DIR=/home/wook/action-tokenizer
cd "$BASE_DIR"
SAM3_SCRIPT=analysis/sam3_masking/batch_sam3_gr1_unified.py

[ -d "$DATASET" ] || { echo "[x] no such dataset dir: $DATASET"; exit 1; }
[ -f "$DATASET/meta/episodes.jsonl" ] || { echo "[x] $DATASET has no meta/episodes.jsonl"; exit 1; }

# Completeness gate. Count with a glob, NOT `find -name '*.mp4'`: the transfer
# leaves macOS AppleDouble sidecars (._episode_NNNNNN.mp4, 163 B) next to every
# video, which find counts and ls/python-glob do not -- so find reports exactly
# double and a naive check would call a complete dir 2000/1000.
NEP=$(grep -c . "$DATASET/meta/episodes.jsonl")
NVID=$(cd "$DATASET/videos/chunk-000/observation.images.ego_view" && ls -1 episode_*.mp4 2>/dev/null | wc -l)
echo "[data-check] $(basename "$DATASET"): $NVID ego_view mp4 / $NEP episodes in meta"
[ "$NVID" -eq "$NEP" ] || { echo "[x] incomplete dataset ($NVID != $NEP) - pick another"; exit 1; }

mkdir -p "$OUT_ROOT"

# --- GPU memory sampler: per-process usage is what decides whether two of these
# fit on one 49140 MiB a6000. Runs until the masking exits. ---
MEMLOG=$OUT_ROOT/gpu_mem.csv
( while true; do
    nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null \
      | sed "s/^/$(date +%s),/" >> "$MEMLOG" || true
    sleep 2
  done ) &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null || true' EXIT

"$SAM3_PY" -c 'import torch;print("[env] torch",torch.__version__,"|",torch.cuda.get_device_name(0),"|",round(torch.cuda.get_device_properties(0).total_memory/2**20),"MiB")'

# shellcheck disable=SC2086  # EPISODES is an intentional word-split list
srun --unbuffered "$SAM3_PY" "$SAM3_SCRIPT" \
    --data-glob "$DATASET" \
    --out-root "$OUT_ROOT" \
    --model jetjodh/sam3 \
    --episodes $EPISODES \
    --max-nouns 3 \
    --skip-videos \
    --no-prompt-masks

kill "$SAMPLER" 2>/dev/null || true

echo "[mem] peak per-process MiB: $(awk -F, 'NF>=3{gsub(/ /,"",$3); if($3+0>m)m=$3+0} END{print m+0}' "$MEMLOG")"
echo "[mem] samples: $(wc -l < "$MEMLOG")"
echo "[count] npz: $(find "$OUT_ROOT" -name '*.npz' | wc -l)   .partial leftovers: $(find "$OUT_ROOT" -name '*.partial' | wc -l)"
echo "[done] bench on $(basename "$DATASET") episodes: $EPISODES"
