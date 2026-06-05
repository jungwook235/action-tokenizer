#!/bin/bash
#SBATCH --job-name=rcasa_full_v2_hand_pred_norecon_fullstate
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=sjw_alinlab
#SBATCH --output=out/%j-rcasa_full_v2_hand_pred_norecon_fullstate.out
#SBATCH --error=out/%j-rcasa_full_v2_hand_pred_norecon_fullstate.err
#SBATCH --comment "rcasa_full_v2_hand_pred_norecon_fullstate"

set -x
export PATH="$HOME/.local/bin:$PATH"
export WANDB_API_KEY="66a73856614bc24a07523f3fbee42482fcbeffe3"
export HF_TOKEN="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"

source /sjw_alinlab1/home/jungwook/miniconda3/bin/activate gr00t

DATA_DIR=/storage1/sjw_dataset/dataset/huggingface/lerobot/shared/robocasa_preprocessed/robocasa_mg_gr00t_100
TOK_CKPT_DIR=checkpoints_action_tokenizer/robocasa_100demos_v2_hand_pred_norecon_fullstate
VLA_CKPT_DIR=checkpoints/vla_actlat_fm_robocasa_100demos/v2_hand_pred_norecon_fullstate
TOK_STEP=100000

# Stage2 VLA wandb project (robocasa 전용).
export WANDB_PROJECT="gr00t-actlat-fm-robocasa"

# === Stage 1: Tokenizer Training — Fullstate-prediction-only ablation ===
# Ablation: hand_pred_norecon_mask_fullstate에서 masking 축만 제거.
#   - num-hand-tokens 2 (state pred용 hand token 유지)
#   - lambda-hand-pred 1.0 (state pred 활성)
#   - no-hand-in-recon (hand token은 state pred에만 사용, recon 제외)
#   - state-pred-kv-source hand
#   - hand-state-dims 0..15 (robocasa state dim 전체 16개)
#   - hand-pred-future-steps 8 16
#   - lambda-mask-recon 0 (masking 비활성, default)
# full-state prediction 단독 기여도 측정용.
python scripts/train_action_latent_tokenizer_v2.py \
    --dataset-path $DATA_DIR \
    --output-dir $TOK_CKPT_DIR \
    --no-resume \
    --data-config single_panda_gripper \
    --embodiment-tag new_embodiment \
    --run-name "actlat_v2_hand_pred_norecon_fullstate_robocasa_100demos" \
    --num-gpus 2 \
    --batch-size 1024 \
    --max-steps $TOK_STEP \
    --save-steps 5000 \
    --num-global-tokens 0 \
    --num-hand-tokens 2 \
    --lambda-recon 1.0 \
    --lambda-hand-pred 1.0 \
    --no-hand-in-recon \
    --state-pred-kv-source hand \
    --hand-state-dims 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
    --hand-pred-future-steps 8 16 \
    --hand-pred-decoder-depth 2 \
    --recon-loss-type mse \
    --decoder-mode self_attention \
    --wandb-project "action-latent-tokenizer-v2-robocasa" \
    --eval-steps 1000 \
    --report-to wandb

# === Stage 2: VLA Training ===
# target_tokens="all" → VLA denoises 18 tokens (16 time + 2 hand)
# decode시 hand tokens는 무시됨 (hand_in_recon=False)
python scripts/gr00t_finetune_actlat_fm.py \
    --dataset-path $DATA_DIR \
    --output-dir $VLA_CKPT_DIR \
    --data-config single_panda_gripper_actlat_fm \
    --embodiment-tag new_embodiment \
    --base-model-path "nvidia/GR00T-N1.5-3B" \
    --run-name "actlat_fm_v2_hand_pred_norecon_fullstate_robocasa_100demos" \
    --num-gpus 2 \
    --batch-size 32 \
    --max-steps 60000 \
    --save-steps 10000 \
    --eval-steps 1000 \
    --actlat-tokenizer-path $TOK_CKPT_DIR/checkpoint-$TOK_STEP \
    --actlat-target-tokens "all" \
    --val-ratio 0.003 \
    --video-backend "decord"
