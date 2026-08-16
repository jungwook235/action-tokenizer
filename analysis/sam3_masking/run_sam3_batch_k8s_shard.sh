#!/bin/bash
# Batch SAM3 masking over GR-1 unified — variant of run_sam3_batch_k8s.sh for
# pods with FEWER GPUs than the episode split. Same work, smaller pods.
#
# run_sam3_batch_k8s.sh ties the episode split to the pod: --num-shards is the
# pod's GPU count and --shard-id is the GPU index, so one task slice needs one
# 4-GPU pod. Here the two are decoupled:
#   SHARD_TOTAL  the episode split the shard ids are drawn from (default 4)
#   SHARD_BASE   the first shard id this pod owns (default 0)
# A pod runs shards [SHARD_BASE, SHARD_BASE + NUM_GPUS), one per GPU. Two 2-GPU
# pods with SHARD_TOTAL=4 and SHARD_BASE=0 / 2 therefore reproduce exactly what
# one 4-GPU pod did: episode j of a task still goes to shard j % 4, and the
# per-episode outputs and manifest_shard{0..3}.jsonl files are the same ones.
# Mixing layouts is safe — finished episodes are skipped either way.
#
# Task-level split across jobs is unchanged: job k owns dirs
# [k*TASKS_PER_JOB, (k+1)*TASKS_PER_JOB) of the 24 sorted gr1_unified.* dirs.
#
# Normally invoked through a launcher, not directly:
#   ./run_sam3_gr1_1_1_1.sh                 # job 1-1, GPUs for shards 0-1
#   ./run_sam3_gr1_1_1_2.sh                 # job 1-1, GPUs for shards 2-3
#   ./run_sam3_gr1_1_1_1.sh --limit 1       # smoke test: 1 episode per shard
#   NOHUP=1 ./run_sam3_gr1_1_1_1.sh         # detach, survives the shell
#
# Knobs: SAM3_JOB_IDX (0-based, default 0), SAM3_JOB_COUNT (default 8),
# TASKS_PER_JOB (default 3), SHARD_TOTAL (default 4), SHARD_BASE (default 0),
# NUM_GPUS (default = all visible GPUs).
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
# The shards pass a repo id (jetjodh/sam3), so a missing cache would mean every
# pod hitting the hub at once — or hanging, if the pod has no egress. Check up
# front and then forbid network access outright, so a run either uses the local
# snapshot or fails immediately with a clear message.
if [ ! -d "$HF_HOME/hub/models--${SAM3_HF_REPO//\//--}" ]; then
    echo "[x] $SAM3_HF_REPO not in $HF_HOME/hub — run ./setup_sam3_env.sh first" >&2
    exit 1
fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# --- job / GPU layout -------------------------------------------------------
# nvidia-smi is normally injected by the container runtime, but fall back to
# torch so a missing binary does not kill an otherwise healthy pod.
if command -v nvidia-smi >/dev/null 2>&1; then
    VISIBLE=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
else
    VISIBLE=$("$SAM3_PY" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
    echo "[!] nvidia-smi not found; torch reports ${VISIBLE} GPU(s)"
fi
NUM_GPUS=${NUM_GPUS:-$VISIBLE}
[ "$NUM_GPUS" -gt 0 ] || { echo "[x] no GPU visible in this pod (detected $VISIBLE)" >&2; exit 1; }
JOB_IDX=${SAM3_JOB_IDX:-0}
JOB_COUNT=${SAM3_JOB_COUNT:-8}
TASKS_PER_JOB=${TASKS_PER_JOB:-3}
SHARD_TOTAL=${SHARD_TOTAL:-4}
SHARD_BASE=${SHARD_BASE:-0}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}

# A pod that claims shards past SHARD_TOTAL would silently drop episodes (the
# shard filter j % SHARD_TOTAL can never match an id >= SHARD_TOTAL), so refuse.
if [ "$(( SHARD_BASE + NUM_GPUS ))" -gt "$SHARD_TOTAL" ]; then
    echo "[x] SHARD_BASE=${SHARD_BASE} + NUM_GPUS=${NUM_GPUS} > SHARD_TOTAL=${SHARD_TOTAL}" >&2
    echo "    this pod has more GPUs than the shards it was given; set NUM_GPUS or SHARD_TOTAL" >&2
    exit 1
fi

# Fixed sorted dataset order — identical in every job, so the split is deterministic.
mapfile -t ALL_DS < <(ls -d $SAM3_DATA_GLOB 2>/dev/null | sort)
[ "${#ALL_DS[@]}" -gt 0 ] || { echo "[x] no dataset dirs match $SAM3_DATA_GLOB" >&2; exit 1; }
MY_DS=("${ALL_DS[@]:$(( JOB_IDX * TASKS_PER_JOB )):${TASKS_PER_JOB}}")
[ "${#MY_DS[@]}" -gt 0 ] || { echo "[x] job ${JOB_IDX}: no tasks in its slice (${#ALL_DS[@]} dirs total)" >&2; exit 1; }

LAST_SHARD=$(( SHARD_BASE + NUM_GPUS - 1 ))
echo "pod $(hostname): job ${JOB_IDX}/${JOB_COUNT}, ${NUM_GPUS} GPUs, ${#MY_DS[@]}/${#ALL_DS[@]} tasks"
echo "shards   : ${SHARD_BASE}-${LAST_SHARD} of ${SHARD_TOTAL}"
echo "python   : $SAM3_PY"
echo "model    : $SAM3_HF_REPO (from $HF_HOME/hub)"
echo "data     : $SAM3_DATA_GLOB"
echo "out-root : $SAM3_OUT_ROOT"
echo "extra    : $*"
COVERED=$(( JOB_COUNT * TASKS_PER_JOB ))
if [ "$COVERED" -lt "${#ALL_DS[@]}" ]; then
    echo "[!] ${JOB_COUNT} jobs x ${TASKS_PER_JOB} tasks = ${COVERED} < ${#ALL_DS[@]} dirs -> $(( ${#ALL_DS[@]} - COVERED )) task(s) UNASSIGNED"
fi
if [ "$NUM_GPUS" -lt "$SHARD_TOTAL" ]; then
    echo "[!] shards $(( SHARD_TOTAL - NUM_GPUS )) of ${SHARD_TOTAL} are NOT run here — the sibling pod(s) must cover them"
fi
for d in "${MY_DS[@]}"; do echo "  task: $(basename "$d")"; done
mkdir -p "$SAM3_OUT_ROOT" || { echo "[x] cannot create $SAM3_OUT_ROOT" >&2; exit 1; }

# One dataset at a time; within a dataset one process per GPU. NOTE: every job
# appends to the same $SAM3_OUT_ROOT/manifest_shard{K}.jsonl (the filename comes
# from --shard-id), so those logs interleave records from all jobs — the
# per-episode outputs themselves never collide, since jobs own disjoint dataset
# dirs and, within a dir, disjoint shard ids.
launch() {
    local fail=0
    for t in "${!MY_DS[@]}"; do
        local ds="${MY_DS[$t]}"
        local name; name=$(basename "$ds")
        echo "=== [job ${JOB_IDX} shards ${SHARD_BASE}-${LAST_SHARD}] task $(( t + 1 ))/${#MY_DS[@]}: ${name} ==="
        local pids=()
        for i in $(seq 0 $(( NUM_GPUS - 1 ))); do
            local sid=$(( SHARD_BASE + i ))
            # shard id, not GPU index, names the log: sibling pods both use
            # local GPU 0..N-1, so gpu${i} would collide in the shared logs dir.
            local log="logs/${RUN_ID}_job${JOB_IDX}_${name}_shard${sid}.log"
            CUDA_VISIBLE_DEVICES=$i "$SAM3_PY" -u batch_sam3_robot_task.py \
                --data-glob "$ds" --num-shards "$SHARD_TOTAL" --shard-id "$sid" \
                --out-root "$SAM3_OUT_ROOT" --model "$SAM3_HF_REPO" "$@" \
                > "$log" 2>&1 &
            pids+=($!)
            echo "  ${name} shard ${sid}/${SHARD_TOTAL} -> GPU $i, pid $!, log $log"
        done
        for p in "${pids[@]}"; do wait "$p" || fail=1; done
        echo "=== [job ${JOB_IDX}] ${name} done (fail=$fail) ==="
    done
    echo "job ${JOB_IDX} shards ${SHARD_BASE}-${LAST_SHARD} finished all ${#MY_DS[@]} tasks (fail=$fail)"
    return $fail
}

if [ "${NOHUP:-0}" = "1" ]; then
    # re-exec detached; the child re-enters with NOHUP=0 and does the real work
    DRIVER_LOG="logs/${RUN_ID}_job${JOB_IDX}_s${SHARD_BASE}_driver.log"
    NOHUP=0 SKIP_SETUP=1 RUN_ID="$RUN_ID" SAM3_JOB_IDX="$JOB_IDX" SAM3_JOB_COUNT="$JOB_COUNT" \
        TASKS_PER_JOB="$TASKS_PER_JOB" NUM_GPUS="$NUM_GPUS" \
        SHARD_TOTAL="$SHARD_TOTAL" SHARD_BASE="$SHARD_BASE" \
        nohup "$0" "$@" > "$DRIVER_LOG" 2>&1 &
    echo "detached: pid $!, driver log $DRIVER_LOG"
    echo "monitor : tail -f $SAM3_DIR/logs/${RUN_ID}_job${JOB_IDX}_*.log"
    exit 0
fi

launch "$@"
rc=$?
echo "progress: wc -l ${SAM3_OUT_ROOT}/manifest_shard*.jsonl"
exit $rc
