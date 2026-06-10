#!/bin/bash
#SBATCH --job-name=rcasa_full_v2_state_pred_full_time_mask_statemask
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=sjw_alinlab
#SBATCH --output=out/%j-rcasa_full_v2_state_pred_full_time_mask_statemask.out
#SBATCH --error=out/%j-rcasa_full_v2_state_pred_full_time_mask_statemask.err
#SBATCH --comment "rcasa_full_v2_state_pred_full_time_mask_statemask"

set -x
export PATH="$HOME/.local/bin:$PATH"
export WANDB_API_KEY="66a73856614bc24a07523f3fbee42482fcbeffe3"
export HF_TOKEN="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"

source /sjw_alinlab1/home/jungwook/miniconda3/bin/activate gr00t

DATA_DIR=/storage1/sjw_dataset/dataset/huggingface/lerobot/shared/robocasa_preprocessed/robocasa_mg_gr00t_100
TOK_CKPT_DIR=checkpoints_action_tokenizer/robocasa_100demos_v2_state_pred_full_time_mask_statemask
VLA_CKPT_DIR=checkpoints/vla_actlat_fm_robocasa_100demos/v2_state_pred_full_time_mask_statemask
TOK_STEP=100000
ABS_TOK_CKPT="/sjw_alinlab1/home/jungwook/Isaac-GR00T/$TOK_CKPT_DIR"

# Stage2 VLA wandb project (robocasa 전용).
export WANDB_PROJECT="gr00t-actlat-fm-robocasa"

# === Stage 2: VLA Training ===
# target_tokens="time" → VLA denoises 16 tokens (time only, no hand tokens).
python scripts/gr00t_finetune_actlat_fm.py \
    --dataset-path $DATA_DIR \
    --output-dir $VLA_CKPT_DIR \
    --data-config single_panda_gripper_actlat_fm \
    --embodiment-tag new_embodiment \
    --base-model-path "nvidia/GR00T-N1.5-3B" \
    --run-name "actlat_fm_v2_state_pred_full_time_mask_statemask_robocasa_100demos" \
    --num-gpus 2 \
    --batch-size 32 \
    --max-steps 60000 \
    --save-steps 10000 \
    --eval-steps 1000 \
    --actlat-tokenizer-path "$ABS_TOK_CKPT/checkpoint-$TOK_STEP" \
    --actlat-target-tokens "time" \
    --val-ratio 0.003 \
    --video-backend "decord"
