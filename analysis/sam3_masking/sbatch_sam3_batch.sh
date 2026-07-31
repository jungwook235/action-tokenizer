#!/bin/bash
#SBATCH --job-name=sam3_mask_gr1_unified_robot_task_cutout_overlay_maskgen_per_gpu_shards
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --partition=background
#SBATCH --wckey=project-short-name:sub_human
#SBATCH --output=/sjw_alinlab1/home/jungwook/action_tokenizer/analysis/sam3_masking/logs/%j-sam3_gr1_mask.out
#SBATCH --error=/sjw_alinlab1/home/jungwook/action_tokenizer/analysis/sam3_masking/logs/%j-sam3_gr1_mask.err

# Batch SAM3 masking over GR-1 unified: 4 jobs x 4 GPUs, 6 tasks (dataset dirs) per job.
#
#   sbatch --array=0-3 analysis/sam3_masking/sbatch_sam3_batch.sh
# Smoke test (1 episode per shard):
#   sbatch --array=0-3 analysis/sam3_masking/sbatch_sam3_batch.sh --limit 1
#
# Work split — TASK level across jobs, EPISODE level across GPUs:
#   the 24 gr1_unified.* dirs are listed in a fixed sorted order and job k owns
#   dirs [k*TASKS_PER_JOB, (k+1)*TASKS_PER_JOB) = 6 dirs. The job walks its 6 dirs
#   one at a time and, per dir, runs one process per GPU with --num-shards 4
#   --shard-id <gpu>, so episode j of that dir goes to GPU (j % 4) — 250 of the
#   1000 episodes each. Already-finished episodes are skipped, so the whole thing
#   is resumable and re-running with a different job/GPU count is safe.
#
# Extra args after the script name are passed through to batch_sam3_robot_task.py
# (e.g. --limit N, --out-root ..., --data-glob ...).

set -uo pipefail
# Resolve from the script's own location so this works on any cluster/mount.
SAM3_DIR=${SAM3_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}

# 클러스터 제출 필터가 요구하는 출력 디렉토리 (env로 덮어쓰기 가능).
MODEL_OUTPUT_DIR=${MODEL_OUTPUT_DIR:-/rlwrld-unified-checkpoints/jungwook/action_tokenizer/data}
export MODEL_OUTPUT_DIR

# 저장 경로: MODEL_OUTPUT_DIR를 루트로 사용 (cutout/overlay/masks가 이 아래에 데이터셋별로 생성됨).
# 제출 시 OUT_ROOT=... sbatch ... 로 덮어쓸 수 있음.
OUT_ROOT=${OUT_ROOT:-${MODEL_OUTPUT_DIR}/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim_sam3_robot_task}
DATA_GLOB=${DATA_GLOB:-/storage1/sjw_dataset/dataset/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.*}

cd "$SAM3_DIR"
mkdir -p logs

NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
JOB_ID=${SLURM_ARRAY_TASK_ID:-0}
JOB_CNT=${SLURM_ARRAY_TASK_COUNT:-1}
TASKS_PER_JOB=${TASKS_PER_JOB:-6}

# Fixed sorted dataset order — identical in every job, so the split is deterministic.
mapfile -t ALL_DS < <(ls -d $DATA_GLOB 2>/dev/null | sort)
[ "${#ALL_DS[@]}" -gt 0 ] || { echo "[x] no dataset dirs match $DATA_GLOB" >&2; exit 1; }
MY_DS=("${ALL_DS[@]:$(( JOB_ID * TASKS_PER_JOB )):${TASKS_PER_JOB}}")
[ "${#MY_DS[@]}" -gt 0 ] || { echo "[x] job ${JOB_ID}: no tasks in its slice (${#ALL_DS[@]} dirs total)" >&2; exit 1; }

echo "node $(hostname): job ${JOB_ID}/${JOB_CNT}, ${NGPU} GPUs, ${#MY_DS[@]}/${#ALL_DS[@]} tasks"
echo "out-root: ${OUT_ROOT}"
COVERED=$(( JOB_CNT * TASKS_PER_JOB ))
if [ "$COVERED" -lt "${#ALL_DS[@]}" ]; then
    echo "[!] ${JOB_CNT} jobs x ${TASKS_PER_JOB} tasks = ${COVERED} < ${#ALL_DS[@]} dirs -> $(( ${#ALL_DS[@]} - COVERED )) task(s) UNASSIGNED"
fi
for d in "${MY_DS[@]}"; do echo "  task: $(basename "$d")"; done

# One dataset at a time; within a dataset one process per GPU. Sequential over
# datasets keeps the model resident per process for a whole dataset and keeps the
# resumable per-episode skip logic untouched.
fail=0
for t in "${!MY_DS[@]}"; do
    ds="${MY_DS[$t]}"
    name=$(basename "$ds")
    echo "=== [job ${JOB_ID}] task $(( t + 1 ))/${#MY_DS[@]}: ${name} ==="
    pids=()
    for i in $(seq 0 $(( NGPU - 1 ))); do
        CUDA_VISIBLE_DEVICES=$i "$SAM3_DIR/venv_sam3/bin/python" -u batch_sam3_robot_task.py \
            --data-glob "$ds" --num-shards "$NGPU" --shard-id "$i" --out-root "$OUT_ROOT" "$@" \
            > "logs/${SLURM_JOB_ID:-nojob}_job${JOB_ID}_${name}_gpu${i}.log" 2>&1 &
        pids+=($!)
        echo "  ${name} shard ${i}/${NGPU} -> GPU $i, pid $!"
    done
    for p in "${pids[@]}"; do
        wait "$p" || fail=1
    done
    echo "=== [job ${JOB_ID}] ${name} done (fail=$fail) ==="
done
echo "job ${JOB_ID} finished all ${#MY_DS[@]} tasks (fail=$fail)"
exit $fail
