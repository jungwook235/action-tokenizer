#!/bin/bash
# SAM3 GR-1 masking — job 4 of 4. Owns tasks 18-23 of the 24 sorted
# gr1_unified.* dirs and runs them on this pod's 4 GPUs (episode j of a task -> GPU j%4).
#
# Paired with the k8s Job `jungwook-data-sam3-gr1-4`, whose container args call:
#   bash /data/jungwook/action-tokenizer/analysis/sam3_masking/run_sam3_gr1_4.sh
#
# All 4 jobs write into ONE shared output root (SAM3_OUT_ROOT in sam3_env.sh) —
# deliberately not the yaml's per-job MODEL_OUTPUT_DIR, which would split the
# dataset across 4 directories. Episodes already on disk are skipped, so a
# re-submitted job resumes where it left off.
#
# Extra args are passed through to batch_sam3_robot_task.py, e.g.:
#   ./run_sam3_gr1_4.sh --limit 1        # smoke test: 1 episode per GPU shard
set -uo pipefail

export SAM3_JOB_IDX=3     # 0-based; NOT the yaml's JOB_ID (that is a name string)
export SAM3_JOB_COUNT=4
export TASKS_PER_JOB=6

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_sam3_batch_k8s.sh" "$@"
