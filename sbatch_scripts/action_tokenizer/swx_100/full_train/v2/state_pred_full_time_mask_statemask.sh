#!/bin/bash
#SBATCH --job-name=gr00t_swx_full_v2_state_pred_full_time_mask_statemask
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=sjw_alinlab
#SBATCH --output=out/%j-gr00t_swx_full_v2_state_pred_full_time_mask_statemask.out
#SBATCH --error=out/%j-gr00t_swx_full_v2_state_pred_full_time_mask_statemask.err
#SBATCH --comment "gr00t_swx_full_v2_state_pred_full_time_mask_statemask"

set -x
export PATH="$HOME/.local/bin:$PATH"
export WANDB_API_KEY="66a73856614bc24a07523f3fbee42482fcbeffe3"
export HF_TOKEN="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"

source /sjw_alinlab1/home/jungwook/miniconda3/bin/activate gr00t

DATA_DIR=/storage1/sjw_dataset/dataset/huggingface/lerobot/shared/bridge_orig_lerobot
TOK_CKPT_DIR=checkpoints_action_tokenizer/swx_100demos_v2_state_pred_full_time_mask_statemask
VLA_CKPT_DIR=checkpoints/vla_actlat_fm_swx_100demos/v2_state_pred_full_time_mask_statemask
TOK_STEP=100000
ABS_TOK_CKPT="/sjw_alinlab1/home/jungwook/Isaac-GR00T/$TOK_CKPT_DIR"

# Stage2 VLA wandb project (swx 전용).
export WANDB_PROJECT="gr00t-actlat-fm-swx"

# === Stage 1: Tokenizer Training ===
# Robocasa `v2_state_pred_full_time_mask_statemask` recipe (GR1 SOTA tier 동일) 를 swx 에 포팅.
# 구성: time-KV + full state prediction (10 dim) + masking (ratio 0.2-0.4, batch 50%) +
#       lambda_mask_hand_pred=1.0 (masked latent → future state, ID 회복 압력).
# action_dim=7 — bridge widowx. state_dim=10. hand_state_dims 0..9 = 전체 state.
# num_hand_tokens=0 (hand tok 사용 안함), state_pred_kv_source=time.
python scripts/train_action_latent_tokenizer_v2.py \
    --dataset-path $DATA_DIR \
    --output-dir $TOK_CKPT_DIR \
    --no-resume \
    --data-config bridge_flare_kty_actlat_fm \
    --embodiment-tag new_embodiment \
    --run-name "actlat_v2_state_pred_full_time_mask_statemask_swx_100demos" \
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
    --hand-state-dims 0 1 2 3 4 5 6 7 8 9 \
    --hand-pred-future-steps 8 16 \
    --hand-pred-decoder-depth 2 \
    --recon-loss-type mse \
    --decoder-mode self_attention \
    --wandb-project "action-latent-tokenizer-v2-swx" \
    --eval-steps 1000 \
    --report-to wandb

# === Stage 2: VLA Training ===
# target_tokens="time" → VLA denoises 16 tokens (time only, no hand tokens).
python scripts/gr00t_finetune_actlat_fm.py \
    --dataset-path $DATA_DIR \
    --output-dir $VLA_CKPT_DIR \
    --data-config bridge_flare_kty_actlat_fm \
    --embodiment-tag new_embodiment \
    --base-model-path "nvidia/GR00T-N1.5-3B" \
    --run-name "actlat_fm_v2_state_pred_full_time_mask_statemask_swx_100demos" \
    --num-gpus 2 \
    --batch-size 32 \
    --max-steps 60000 \
    --save-steps 10000 \
    --eval-steps 1000 \
    --actlat-tokenizer-path "$ABS_TOK_CKPT/checkpoint-$TOK_STEP" \
    --actlat-target-tokens "time" \
    --val-ratio 0.003 \
    --video-backend "torchvision_av"
