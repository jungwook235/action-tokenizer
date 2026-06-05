#!/bin/bash
#SBATCH --job-name=full_v3_state_pred_full_time_mask_statemask
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=sjw_alinlab
#SBATCH --output=out/%j-full_v3_state_pred_full_time_mask_statemask.out
#SBATCH --error=out/%j-full_v3_state_pred_full_time_mask_statemask.err
#SBATCH --comment "full_v3_state_pred_full_time_mask_statemask"

set -x
export PATH="$HOME/.local/bin:$PATH"
export WANDB_API_KEY="66a73856614bc24a07523f3fbee42482fcbeffe3"
export HF_TOKEN="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"

source /sjw_alinlab1/home/jungwook/miniconda3/bin/activate gr00t

DATA_DIR=/storage1/sjw_dataset/dataset/robocasa_gr1_tabletop/sim_100demos
TOK_CKPT_DIR=checkpoints_action_tokenizer/gr1_100demos_v3_state_pred_full_time_mask_statemask
VLA_CKPT_DIR=checkpoints/vla_actlat_fm_gr1_100demos/v3_state_pred_full_time_mask_statemask
TOK_STEP=100000
ABS_TOK_CKPT="/sjw_alinlab1/home/jungwook/Isaac-GR00T/$TOK_CKPT_DIR"

# Fixed validation split shared across all v3 experiments on this dataset.
FIXED_VAL_PATH=/sjw_alinlab1/home/jungwook/Isaac-GR00T/experiments/fixed_val_splits/gr1_100demos.json

# === Stage 1: Tokenizer Training ===
# v2/state_pred_full_time_mask_statemask 와 동일한 하이퍼파라미터에
# v3 신규 옵션을 추가:
#   --encoder-output-layernorm  (encoder 출력 LayerNorm)
#   --latent-noise-std 0.1      (encoded latent 에 Gaussian noise; training only)
#   --recon-loss-type l1        (mse → l1)
#   --use-fixed-val + --fixed-val-path  (val episode 고정 — 같은 path 가리키는 모든 실험이 동일한 val set)
#   --data-config fourier_gr1_arms_waist_q99  (action 정규화 q01/q99)
python scripts/train_action_latent_tokenizer_v3.py \
    --dataset-path $DATA_DIR \
    --output-dir $TOK_CKPT_DIR \
    --no-resume \
    --data-config fourier_gr1_arms_waist_q99 \
    --embodiment-tag new_embodiment \
    --run-name "actlat_v3_state_pred_full_time_mask_statemask" \
    --num-gpus 2 \
    --batch-size 1024 \
    --max-steps $TOK_STEP \
    --save-steps 5000 \
    --num-global-tokens 0 \
    --num-hand-tokens 0 \
    --lambda-recon 1.0 \
    --lambda-hand-pred 1.0 \
    --lambda-mask-recon 1.0 \
    --lambda-mask-hand-pred 1.0 \
    --mask-ratio-min 0.2 \
    --mask-ratio-max 0.4 \
    --mask-batch-ratio 0.5 \
    --state-pred-kv-source time \
    --hand-state-dims 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 \
    --hand-pred-future-steps 8 16 \
    --hand-pred-decoder-depth 2 \
    --recon-loss-type l1 \
    --decoder-mode self_attention \
    --encoder-output-layernorm \
    --latent-noise-std 0.1 \
    --use-fixed-val \
    --fixed-val-path "$FIXED_VAL_PATH" \
    --wandb-project "action-latent-tokenizer-v3" \
    --eval-steps 1000 \
    --report-to wandb

# === Stage 2: VLA Training ===
# target_tokens="time" → VLA denoises 16 tokens (time only, no hand tokens)
python scripts/gr00t_finetune_actlat_fm.py \
    --dataset-path $DATA_DIR \
    --output-dir $VLA_CKPT_DIR \
    --data-config fourier_gr1_arms_waist_actlat_fm_q99 \
    --embodiment-tag new_embodiment \
    --base-model-path "nvidia/GR00T-N1.5-3B" \
    --run-name "actlat_fm_v3_state_pred_full_time_mask_statemask_gr1_100demos" \
    --num-gpus 2 \
    --batch-size 32 \
    --max-steps 60000 \
    --save-steps 10000 \
    --eval-steps 1000 \
    --actlat-tokenizer-path "$ABS_TOK_CKPT/checkpoint-$TOK_STEP" \
    --actlat-target-tokens "time" \
    --val-ratio 0.003 \
    --video-backend "decord"
