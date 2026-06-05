#!/bin/bash

set -x
export PATH="$HOME/.local/bin:$PATH"

source /sjw_alinlab1/home/jungwook/miniconda3/bin/activate gr00t

# Match training: swx_100/full_train/v2/state_pred_full_time_mask_statemask.sh
DATA_DIR=/storage1/sjw_dataset/dataset/huggingface/lerobot/shared/bridge_orig_lerobot
DATA_CONFIG=bridge_flare_kty_actlat_fm
TOK_CKPT_DIR=checkpoints_action_tokenizer/swx_100demos_v2_state_pred_full_time_mask_statemask
TOK_STEP=100000
ABS_TOK_CKPT="/sjw_alinlab1/home/jungwook/Isaac-GR00T/$TOK_CKPT_DIR/checkpoint-$TOK_STEP"

FIXED_VAL_PATH=/sjw_alinlab1/home/jungwook/Isaac-GR00T/experiments/fixed_val_splits/swx_100demos.json
OUT_DIR=experiments/runs/recon_eval/swx_v2_state_pred_full_time_mask_statemask

mkdir -p "$OUT_DIR"

python scripts/eval_tokenizer_recon.py \
    --checkpoint-path "$ABS_TOK_CKPT" \
    --dataset-path "$DATA_DIR" \
    --data-config "$DATA_CONFIG" \
    --embodiment-tag new_embodiment \
    --split val \
    --val-ratio 0.003 \
    --val-seed 42 \
    --use-fixed-val \
    --fixed-val-path "$FIXED_VAL_PATH" \
    --target-tokens all \
    --batch-size 256 \
    --device cuda \
    --num-workers 8 \
    --output-dir "$OUT_DIR"
