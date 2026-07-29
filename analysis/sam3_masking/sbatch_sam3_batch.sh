#!/bin/bash
#SBATCH --job-name=sam3_mask_gr1_unified_robot_task_cutout_overlay_maskgen_per_gpu_shards
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --partition=background
#SBATCH --wckey=project-short-name:sub_human
#SBATCH --output=/sjw_alinlab1/home/jungwook/action_tokenizer/analysis/sam3_masking/logs/%j-sam3_gr1_mask.out
#SBATCH --error=/sjw_alinlab1/home/jungwook/action_tokenizer/analysis/sam3_masking/logs/%j-sam3_gr1_mask.err

# Batch SAM3 masking over GR-1 unified, one shard per GPU.
#
# Single node (8 GPUs):
#   sbatch analysis/sam3_masking/sbatch_sam3_batch.sh
# Multi node via job array (e.g. 3 nodes x 8 GPUs = 24 shards):
#   sbatch --array=0-2 analysis/sam3_masking/sbatch_sam3_batch.sh
# Smoke test (2 episodes per shard):
#   sbatch --gpus=2 analysis/sam3_masking/sbatch_sam3_batch.sh --limit 2
#
# Global shard layout: total = array_count * gpus_per_node; this task owns
# shards [array_id*gpus, (array_id+1)*gpus). Episodes are enumerated in a fixed
# sorted order and episode j goes to shard (j % total), so the split is
# deterministic. Already-finished episodes are skipped -> fully resumable, and
# re-running with a different node/GPU count is safe.
#
# Extra args after the script name are passed through to batch_sam3_robot_task.py
# (e.g. --limit N, --out-root ..., --data-glob ...).

set -uo pipefail
SAM3_DIR=/sjw_alinlab1/home/jungwook/action_tokenizer/analysis/sam3_masking

# 클러스터 제출 필터가 요구하는 출력 디렉토리 (env로 덮어쓰기 가능).
MODEL_OUTPUT_DIR=${MODEL_OUTPUT_DIR:-/rlwrld-unified-checkpoints/jungwook/action_tokenizer/data}
export MODEL_OUTPUT_DIR

# 저장 경로: MODEL_OUTPUT_DIR를 루트로 사용 (cutout/overlay/masks가 이 아래에 데이터셋별로 생성됨).
# 제출 시 OUT_ROOT=... sbatch ... 로 덮어쓸 수 있음.
OUT_ROOT=${OUT_ROOT:-${MODEL_OUTPUT_DIR}/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim_sam3_robot_task}

cd "$SAM3_DIR"
mkdir -p logs

NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
ARRAY_ID=${SLURM_ARRAY_TASK_ID:-0}
ARRAY_CNT=${SLURM_ARRAY_TASK_COUNT:-1}
TOTAL=$(( ARRAY_CNT * NGPU ))

echo "node $(hostname): array ${ARRAY_ID}/${ARRAY_CNT}, ${NGPU} GPUs -> shards $(( ARRAY_ID * NGPU ))..$(( ARRAY_ID * NGPU + NGPU - 1 )) of ${TOTAL}"
echo "out-root: ${OUT_ROOT}"

pids=()
for i in $(seq 0 $(( NGPU - 1 ))); do
    shard=$(( ARRAY_ID * NGPU + i ))
    CUDA_VISIBLE_DEVICES=$i "$SAM3_DIR/venv_sam3/bin/python" -u batch_sam3_robot_task.py \
        --num-shards "$TOTAL" --shard-id "$shard" --out-root "$OUT_ROOT" "$@" \
        > "logs/${SLURM_JOB_ID:-nojob}_shard_${shard}_of_${TOTAL}.log" 2>&1 &
    pids+=($!)
    echo "shard ${shard}/${TOTAL} -> GPU $i, pid $!"
done

fail=0
for p in "${pids[@]}"; do
    wait "$p" || fail=1
done
echo "all shards finished (fail=$fail)"
exit $fail
