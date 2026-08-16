#!/bin/bash
# SAM3 GR-1 masking — job 4-2: the 2nd half of the original job 4.
# Owns tasks 21-23 of the 24 sorted gr1_unified.* dirs (3 tasks) and runs them on
# this pod's 4 GPUs (episode j of a task -> GPU j%4).
#
# Paired with the k8s Job `jungwook-data-sam3-gr1-4-2`, whose container args call:
#   bash /data/jungwook/action-tokenizer/analysis/sam3_masking/run_sam3_gr1_4_2.sh
#
# 8 half-jobs x 3 tasks = the same 24 dirs the 4 x 6 layout covered, so the two
# layouts are interchangeable: episodes already on disk are skipped either way.
# All jobs write into ONE shared output root (SAM3_OUT_ROOT in sam3_env.sh).
#
# Extra args are passed through to batch_sam3_robot_task.py, e.g.:
#   ./run_sam3_gr1_4_2.sh --limit 1     # smoke test: 1 episode per GPU shard
set -uo pipefail

export SAM3_JOB_IDX=7     # 0-based over the 8 half-jobs; NOT the yaml's JOB_ID
export SAM3_JOB_COUNT=8
export TASKS_PER_JOB=3

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_sam3_batch_k8s.sh" "$@"
