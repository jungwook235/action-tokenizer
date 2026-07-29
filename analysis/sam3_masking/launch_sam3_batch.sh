#!/bin/bash
# Launch N parallel SAM3 masking shards in the background (nohup).
#
# Usage:
#   ./launch_sam3_batch.sh <num_jobs> [gpu_ids_csv] [extra args passed to batch_sam3_robot_task.py]
# Examples:
#   ./launch_sam3_batch.sh 4                 # 4 shards, all on GPU 0
#   ./launch_sam3_batch.sh 8 0,1,2,3         # 8 shards round-robin over GPUs 0-3
#   ./launch_sam3_batch.sh 2 0 --limit 3     # smoke test: 2 shards, 3 episodes each
#
# Shards are resumable: re-running skips episodes whose outputs already exist.
set -euo pipefail
cd "$(dirname "$0")"

N=${1:?usage: launch_sam3_batch.sh <num_jobs> [gpu_ids_csv] [extra args]}
GPUS=${2:-0}
shift $(( $# >= 2 ? 2 : 1 ))
IFS=',' read -ra G <<< "$GPUS"

mkdir -p logs
for i in $(seq 0 $((N - 1))); do
    gpu=${G[$((i % ${#G[@]}))]}
    CUDA_VISIBLE_DEVICES=$gpu nohup ./venv_sam3/bin/python batch_sam3_robot_task.py \
        --num-shards "$N" --shard-id "$i" "$@" \
        > "logs/shard_${i}_of_${N}.log" 2>&1 &
    echo "shard $i/$N -> GPU $gpu, pid $!, log logs/shard_${i}_of_${N}.log"
done
echo "monitor: tail -f logs/shard_*_of_${N}.log ; progress: wc -l <out_root>/manifest_shard*.jsonl"
