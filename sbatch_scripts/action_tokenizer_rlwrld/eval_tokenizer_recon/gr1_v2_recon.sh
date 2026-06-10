#!/bin/bash

set -x
export PATH="$HOME/.local/bin:$PATH"

source /sjw_alinlab1/home/jungwook/miniconda3/bin/activate gr00t

# Match training: gr1 recon-only baseline (no gr1-specific train script exists yet — recipe
# follows robocasa/swx recon.sh but on gr1 dataset/data_config).
# data_config: same as gr1_100/full_train/v2/state_pred_full_time*.sh → fourier_gr1_arms_waist.
DATA_DIR=/storage1/sjw_dataset/dataset/robocasa_gr1_tabletop/sim_100demos
DATA_CONFIG=fourier_gr1_arms_waist
TOK_CKPT_DIR=checkpoints_action_tokenizer/gr1_100demos_v2_base
TOK_STEP=100000
ABS_TOK_CKPT="/sjw_alinlab1/home/jungwook/Isaac-GR00T/$TOK_CKPT_DIR/checkpoint-$TOK_STEP"

FIXED_VAL_PATH=/sjw_alinlab1/home/jungwook/Isaac-GR00T/experiments/fixed_val_splits/gr1_100demos.json
OUT_DIR=experiments/runs/recon_eval/gr1_v2_recon

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
