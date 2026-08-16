#!/bin/bash
# SAM3 GR-1 masking — job 1-1-2: run_sam3_gr1_1_1.sh split onto a 2-GPU pod.
# Same 3 tasks (dirs 0-2 of the 24 sorted gr1_unified.*) as job 1-1, but this pod
# only runs the SECOND half of each task's episodes: shards 2-3 of 4.
# Its sibling run_sam3_gr1_1_1_1.sh runs shards 0-1 — both must run to finish
# these 3 tasks.
#
# Paired with a k8s Job requesting 2 GPUs, whose container args call:
#   bash /data/jungwook/action-tokenizer/analysis/sam3_masking/run_sam3_gr1_1_1_2.sh
#
# The episode split is unchanged from the 4-GPU layout (episode j -> shard j%4),
# so outputs and manifest_shard{0..3}.jsonl are the same files job 1-1 wrote;
# episodes already on disk are skipped, so the layouts are interchangeable.
# All jobs write into ONE shared output root (SAM3_OUT_ROOT in sam3_env.sh).
#
# Extra args are passed through to batch_sam3_robot_task.py, e.g.:
#   ./run_sam3_gr1_1_1_2.sh --limit 1   # smoke test: 1 episode per shard
set -uo pipefail

export SAM3_JOB_IDX=0     # same task slice as 1-1-1; only the shards differ
export SAM3_JOB_COUNT=8
export TASKS_PER_JOB=3
export SHARD_TOTAL=4      # episode split is still 4-way …
export SHARD_BASE=2       # … this pod takes shards 2-3

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_sam3_batch_k8s_shard.sh" "$@"
