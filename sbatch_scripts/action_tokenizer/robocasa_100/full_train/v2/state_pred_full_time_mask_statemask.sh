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

# === Stage 1: Tokenizer Training ===
# Robocasa 포팅: GR1 `v2_state_pred_full_time_mask_statemask` recipe (AVG24=41.92%) 그대로 옮김.
# 구성: time-KV + full state prediction (20 dim) + masking (ratio 0.2-0.4, batch 50%) +
#       lambda_mask_hand_pred=1.0 (masked latent → future state, ID 회복 압력).
# action_dim=12 (EE pos3 + EE rot3 + gripper1 + base4 + ctrl_mode1) — robocasa.
# state_dim=20 (pretransform 자동 확장). hand_state_dims 0..19 = 전체 state.
# num_hand_tokens=0 (hand tok 사용 안함), state_pred_kv_source=time.
python scripts/train_action_latent_tokenizer_v2.py \
    --dataset-path $DATA_DIR \
    --output-dir $TOK_CKPT_DIR \
    --no-resume \
    --data-config single_panda_gripper \
    --embodiment-tag new_embodiment \
    --run-name "actlat_v2_state_pred_full_time_mask_statemask_robocasa_100demos" \
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
    --hand-state-dims 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
    --hand-pred-future-steps 8 16 \
    --hand-pred-decoder-depth 2 \
    --recon-loss-type mse \
    --decoder-mode self_attention \
    --wandb-project "action-latent-tokenizer-v2-robocasa" \
    --eval-steps 1000 \
    --report-to wandb

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
