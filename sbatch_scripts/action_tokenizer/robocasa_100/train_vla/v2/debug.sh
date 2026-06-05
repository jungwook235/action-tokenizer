#!/bin/bash

set -x
export PATH="$HOME/.local/bin:$PATH"
export WANDB_API_KEY="66a73856614bc24a07523f3fbee42482fcbeffe3"
export HF_TOKEN="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"

source /sjw_alinlab1/home/jungwook/miniconda3/bin/activate gr00t

DATA_DIR=/storage1/sjw_dataset/dataset/huggingface/lerobot/shared/robocasa_preprocessed/robocasa_mg_gr00t_100
TOK_CKPT_DIR=checkpoints_action_tokenizer/robocasa_100demos_v2_hand_pred_norecon_mask_fullstate
VLA_CKPT_DIR=checkpoints/vla_actlat_fm_robocasa_100demos/v2_hand_pred_norecon_mask_fullstate
TOK_STEP=100000

# Stage2 VLA wandb project (robocasa 전용). Stage1 토크나이저는 아래 --wandb-project 인자로 분리.
export WANDB_PROJECT="gr00t-actlat-fm-robocasa"


# === Stage 2: VLA Training ===
# target_tokens="all" → VLA denoises 18 tokens (16 time + 2 hand)
# decode시 hand tokens는 무시됨 (hand_in_recon=False)
python scripts/gr00t_finetune_actlat_fm.py \
    --dataset-path $DATA_DIR \
    --output-dir $VLA_CKPT_DIR \
    --data-config single_panda_gripper_actlat_fm \
    --embodiment-tag new_embodiment \
    --base-model-path "nvidia/GR00T-N1.5-3B" \
    --run-name "debug" \
    --num-gpus 1 \
    --batch-size 32 \
    --max-steps 60000 \
    --save-steps 10000 \
    --eval-steps 1 \
    --actlat-tokenizer-path $TOK_CKPT_DIR/checkpoint-$TOK_STEP \
    --actlat-target-tokens "all" \
    --val-ratio 0.003 \
    --video-backend "decord"
