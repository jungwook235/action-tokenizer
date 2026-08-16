#!/bin/bash
# Batch SAM3 masking over GR-1 unified — shared runner behind the 4 per-job
# launchers (run_sam3_gr1_1.sh … run_sam3_gr1_4.sh), one k8s Job each with 4 GPUs.
# k8s replacement for sbatch_sam3_batch.sh (this cluster has no SLURM; a pod just
# runs a script). Work split: 4 jobs x 4 GPUs, 6 tasks (dataset dirs) per job.
#
# Normally invoked through a launcher, not directly:
#   ./run_sam3_gr1_1.sh                          # job 1/4 -> tasks 0-5
#   SAM3_JOB_IDX=0 ./run_sam3_batch_k8s.sh       # same thing, spelled out
#   ./run_sam3_gr1_1.sh --limit 1                # smoke test: 1 episode per shard
#   NOHUP=1 ./run_sam3_gr1_1.sh                  # detach, survives the shell
#
# Knobs: SAM3_JOB_IDX (0-based, default 0), SAM3_JOB_COUNT (default 4),
# TASKS_PER_JOB (default 6), NUM_GPUS (default = all visible GPUs).
# NOTE: the index is SAM3_JOB_IDX, not JOB_ID — the k8s yamls already use JOB_ID
# for the Job's name string (JOB_ID=jungwook-data-sam3-gr1-1), so reusing it here
# would feed a non-numeric value into the shard arithmetic.
#
# Work split — TASK level across jobs, EPISODE level across GPUs:
#   the 24 gr1_unified.* dirs are listed in a fixed sorted order and job k owns
#   dirs [k*TASKS_PER_JOB, (k+1)*TASKS_PER_JOB) = 6 dirs. The job walks its 6 dirs
#   one at a time and, per dir, runs one process per GPU with --num-shards 4
#   --shard-id <gpu>, so episode j of that dir goes to GPU (j % 4) — 250 of the
#   1000 episodes each. Finished episodes are skipped -> fully resumable, and
#   re-running with a different job/GPU count is safe.
#
# Extra args are passed through to batch_sam3_robot_task.py (--limit N, ...).
set -uo pipefail

SAM3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SAM3_DIR/sam3_env.sh"

# --- env bootstrap ----------------------------------------------------------
# A fresh pod has an empty root fs: no uv, no venv (only /data persists). Bring
# the env up first — a no-op once requirements_sam3.txt is already in sync.
if [ "${SKIP_SETUP:-0}" != "1" ]; then
    "$SAM3_DIR/setup_sam3_env.sh" --venv-only || { echo "[x] env setup failed" >&2; exit 1; }
fi
[ -x "$SAM3_PY" ] || { echo "[x] no venv python at $SAM3_PY (run ./setup_sam3_env.sh)" >&2; exit 1; }

cd "$SAM3_DIR"
mkdir -p logs

# --- weights must be in the cache; then pin the run offline ------------------
# The shards pass a repo id (jetjodh/sam3), so a missing cache would mean 4 pods
# hitting the hub at once — or hanging, if the pod has no egress. Check up front
# and then forbid network access outright, so a run either uses the local
# snapshot or fails immediately with a clear message.
if [ ! -d "$HF_HOME/hub/models--${SAM3_HF_REPO//\//--}" ]; then
    echo "[x] $SAM3_HF_REPO not in $HF_HOME/hub — run ./setup_sam3_env.sh first" >&2
    exit 1
fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# --- job / GPU layout -------------------------------------------------------
# nvidia-smi is normally injected by the container runtime, but fall back to
# torch so a missing binary does not kill an otherwise healthy 4-GPU pod.
if command -v nvidia-smi >/dev/null 2>&1; then
    VISIBLE=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
else
    VISIBLE=$("$SAM3_PY" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
    echo "[!] nvidia-smi not found; torch reports ${VISIBLE} GPU(s)"
fi
NUM_GPUS=${NUM_GPUS:-$VISIBLE}
[ "$NUM_GPUS" -gt 0 ] || { echo "[x] no GPU visible in this pod (detected $VISIBLE)" >&2; exit 1; }
JOB_IDX=${SAM3_JOB_IDX:-0}
JOB_COUNT=${SAM3_JOB_COUNT:-4}
TASKS_PER_JOB=${TASKS_PER_JOB:-6}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}

# Fixed sorted dataset order — identical in every job, so the split is deterministic.
mapfile -t ALL_DS < <(ls -d $SAM3_DATA_GLOB 2>/dev/null | sort)
[ "${#ALL_DS[@]}" -gt 0 ] || { echo "[x] no dataset dirs match $SAM3_DATA_GLOB" >&2; exit 1; }
MY_DS=("${ALL_DS[@]:$(( JOB_IDX * TASKS_PER_JOB )):${TASKS_PER_JOB}}")
[ "${#MY_DS[@]}" -gt 0 ] || { echo "[x] job ${JOB_IDX}: no tasks in its slice (${#ALL_DS[@]} dirs total)" >&2; exit 1; }

echo "pod $(hostname): job ${JOB_IDX}/${JOB_COUNT}, ${NUM_GPUS} GPUs, ${#MY_DS[@]}/${#ALL_DS[@]} tasks"
echo "python   : $SAM3_PY"
echo "model    : $SAM3_HF_REPO (from $HF_HOME/hub)"
echo "data     : $SAM3_DATA_GLOB"
echo "out-root : $SAM3_OUT_ROOT"
echo "extra    : $*"
COVERED=$(( JOB_COUNT * TASKS_PER_JOB ))
if [ "$COVERED" -lt "${#ALL_DS[@]}" ]; then
    echo "[!] ${JOB_COUNT} jobs x ${TASKS_PER_JOB} tasks = ${COVERED} < ${#ALL_DS[@]} dirs -> $(( ${#ALL_DS[@]} - COVERED )) task(s) UNASSIGNED"
fi
for d in "${MY_DS[@]}"; do echo "  task: $(basename "$d")"; done
mkdir -p "$SAM3_OUT_ROOT" || { echo "[x] cannot create $SAM3_OUT_ROOT" >&2; exit 1; }

# One dataset at a time; within a dataset one process per GPU. NOTE: all jobs
# append to the same $SAM3_OUT_ROOT/manifest_shard{0..NUM_GPUS-1}.jsonl (the
# filename comes from --shard-id), so those logs interleave records from the 4
# jobs — the per-episode outputs themselves never collide, since jobs own
# disjoint dataset dirs.
launch() {
    local fail=0
    for t in "${!MY_DS[@]}"; do
        local ds="${MY_DS[$t]}"
        local name; name=$(basename "$ds")
        echo "=== [job ${JOB_IDX}] task $(( t + 1 ))/${#MY_DS[@]}: ${name} ==="
        local pids=()
        for i in $(seq 0 $(( NUM_GPUS - 1 ))); do
            local log="logs/${RUN_ID}_job${JOB_IDX}_${name}_gpu${i}.log"
            CUDA_VISIBLE_DEVICES=$i "$SAM3_PY" -u batch_sam3_robot_task.py \
                --data-glob "$ds" --num-shards "$NUM_GPUS" --shard-id "$i" \
                --out-root "$SAM3_OUT_ROOT" --model "$SAM3_HF_REPO" "$@" \
                > "$log" 2>&1 &
            pids+=($!)
            echo "  ${name} shard ${i}/${NUM_GPUS} -> GPU $i, pid $!, log $log"
        done
        for p in "${pids[@]}"; do wait "$p" || fail=1; done
        echo "=== [job ${JOB_IDX}] ${name} done (fail=$fail) ==="
    done
    echo "job ${JOB_IDX} finished all ${#MY_DS[@]} tasks (fail=$fail)"
    return $fail
}

if [ "${NOHUP:-0}" = "1" ]; then
    # re-exec detached; the child re-enters with NOHUP=0 and does the real work
    NOHUP=0 SKIP_SETUP=1 RUN_ID="$RUN_ID" SAM3_JOB_IDX="$JOB_IDX" SAM3_JOB_COUNT="$JOB_COUNT" \
        TASKS_PER_JOB="$TASKS_PER_JOB" NUM_GPUS="$NUM_GPUS" \
        nohup "$0" "$@" > "logs/${RUN_ID}_job${JOB_IDX}_driver.log" 2>&1 &
    echo "detached: pid $!, driver log logs/${RUN_ID}_job${JOB_IDX}_driver.log"
    echo "monitor : tail -f $SAM3_DIR/logs/${RUN_ID}_job${JOB_IDX}_*.log"
    exit 0
fi

launch "$@"
rc=$?
echo "progress: wc -l ${SAM3_OUT_ROOT}/manifest_shard*.jsonl"
exit $rc
