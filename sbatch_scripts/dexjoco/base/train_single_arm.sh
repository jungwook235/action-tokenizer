#!/bin/bash
#SBATCH --job-name=full_train_dexjoco_single_arm_nactlat_gr00t_n15_base
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=sjw_alinlab
#SBATCH --output=out/%j-full_train_dexjoco_single_arm_gr00t_n15_base.out
#SBATCH --error=out/%j-full_train_dexjoco_single_arm_gr00t_n15_base.err

set -x
export PATH="$HOME/.local/bin:$PATH"
export WANDB_PROJECT=Action-Tokenizer-DexJoCo-SingleArm
export WANDB_API_KEY="66a73856614bc24a07523f3fbee42482fcbeffe3"
export HF_TOKEN="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"

BASE_DIR=/sjw_alinlab1/home/jungwook/action_tokenizer
cd $BASE_DIR

source /sjw_alinlab1/home/jungwook/miniconda3/bin/activate gr00t-actlat

CKPT_DIR="checkpoints/vla_nactlat_fm_dexjoco_single_arm/base"
# Glob expands to all 24 gr1_unified.* dataset dirs (each a LeRobot dataset root)
DATA_DIR=("/sjw_alinlab1/home/jungwook/dataset/dexjoco_lerobot/v20/click_mouse"
"/sjw_alinlab1/home/jungwook/dataset/dexjoco_lerobot/v20/hammer_nail"
"/sjw_alinlab1/home/jungwook/dataset/dexjoco_lerobot/v20/water_plant"
"/sjw_alinlab1/home/jungwook/dataset/dexjoco_lerobot/v20/fold_glasses"
"/sjw_alinlab1/home/jungwook/dataset/dexjoco_lerobot/v20/pick_bucket"
"/sjw_alinlab1/home/jungwook/dataset/dexjoco_lerobot/v20/pinch_tongs"
)

python scripts/gr00t_finetune_actlat_fm.py \
    --dataset-path "${DATA_DIR[@]}" \
    --output-dir $CKPT_DIR \
    --data-config dexjoco_single_arm \
    --embodiment-tag new_embodiment \
    --base-model-path "nvidia/GR00T-N1.5-3B" \
    --run-name "nactlat_baseline_dexjoco_single_arm_base" \
    --mode "vla" \
    --num-gpus 2 \
    --batch-size 32 \
    --max-steps 60000 \
    --save-steps 5000 \
    --save-total-limit 3 \
    --eval-steps 1000 \
    --val-ratio 0.003 \
    --use-fixed-val \
    --video-backend "decord"
