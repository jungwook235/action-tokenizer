#!/bin/bash
#SBATCH --job-name=gr00t_v2_headfix_state_pred_full_time_gr1
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=sjw_alinlab,h100
#SBATCH --output=out/%j-gr00t_v2_headfix_state_pred_full_time_gr1.out
#SBATCH --error=out/%j-gr00t_v2_headfix_state_pred_full_time_gr1.err
#SBATCH --comment "gr00t_v2_headfix_state_pred_full_time_gr1"

# Post-headfix VLA training. Reuses existing gr1_100demos_v2_state_pred_full_time
# tokenizer (correct weights; only the wrapper-side load was broken before).
# Mirrors target_tokens="time" of the original full_train script.

set -x
export PATH="$HOME/.local/bin:$PATH"
export WANDB_API_KEY="66a73856614bc24a07523f3fbee42482fcbeffe3"
export HF_TOKEN="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"

source /sjw_alinlab1/home/jungwook/miniconda3/bin/activate gr00t

DATA_DIR=/storage1/sjw_dataset/dataset/robocasa_gr1_tabletop/sim_100demos
VLA_CKPT_DIR=checkpoints/vla_actlat_fm_gr1_100demos/v2_headfix_state_pred_full_time
TOKENIZER_PATH=/sjw_alinlab1/home/jungwook/Isaac-GR00T/checkpoints_action_tokenizer/gr1_100demos_v2_state_pred_full_time/checkpoint-100000

python scripts/gr00t_finetune_actlat_fm.py \
    --dataset-path $DATA_DIR \
    --output-dir $VLA_CKPT_DIR \
    --data-config fourier_gr1_arms_waist_actlat_fm \
    --embodiment-tag new_embodiment \
    --base-model-path "nvidia/GR00T-N1.5-3B" \
    --run-name "actlat_fm_v2_headfix_state_pred_full_time_gr1_100demos" \
    --num-gpus 2 \
    --batch-size 32 \
    --max-steps 60000 \
    --save-steps 10000 \
    --eval-steps 1000 \
    --actlat-tokenizer-path $TOKENIZER_PATH \
    --actlat-target-tokens "time" \
    --val-ratio 0.003 \
    --video-backend "decord"
