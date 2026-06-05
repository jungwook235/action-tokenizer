#!/bin/bash
#SBATCH --job-name=full_v2_hand_pred_norecon_mask_fullstate_deep8
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=sjw_alinlab
#SBATCH --output=out/%j-full_v2_hand_pred_norecon_mask_fullstate_deep8.out
#SBATCH --error=out/%j-full_v2_hand_pred_norecon_mask_fullstate_deep8.err
#SBATCH --comment "full_v2_hand_pred_norecon_mask_fullstate_deep8"

set -x
export PATH="$HOME/.local/bin:$PATH"
export WANDB_API_KEY="66a73856614bc24a07523f3fbee42482fcbeffe3"
export HF_TOKEN="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"

source /sjw_alinlab1/home/jungwook/miniconda3/bin/activate gr00t

DATA_DIR=/storage1/sjw_dataset/dataset/robocasa_gr1_tabletop/sim_100demos
TOK_CKPT_DIR=checkpoints_action_tokenizer/gr1_100demos_v2_hand_pred_norecon_mask_fullstate_deep8
VLA_CKPT_DIR=checkpoints/vla_actlat_fm_gr1_100demos/v2_hand_pred_norecon_mask_fullstate_deep8
TOK_STEP=100000

# === Stage 1: Tokenizer Training ===
# Base: v2_hand_pred_norecon_mask_fullstate (SOTA 42.00% AVG_24)
# 변경: encoder-depth 4→8, decoder-depth 2→8, hand-pred-decoder-depth 2→8.
# 의도: tokenizer capacity 확장이 41.x% plateau를 돌파하는지 검증.
python scripts/train_action_latent_tokenizer_v2.py \
    --dataset-path $DATA_DIR \
    --output-dir $TOK_CKPT_DIR \
    --no-resume \
    --data-config fourier_gr1_arms_waist \
    --embodiment-tag new_embodiment \
    --run-name "actlat_v2_hand_pred_norecon_mask_fullstate_deep8" \
    --num-gpus 2 \
    --batch-size 1024 \
    --max-steps $TOK_STEP \
    --save-steps 5000 \
    --encoder-depth 8 \
    --decoder-depth 8 \
    --num-global-tokens 0 \
    --num-hand-tokens 2 \
    --lambda-recon 1.0 \
    --lambda-hand-pred 1.0 \
    --lambda-mask-recon 1.0 \
    --mask-ratio-min 0.2 \
    --mask-ratio-max 0.4 \
    --mask-batch-ratio 0.5 \
    --no-hand-in-recon \
    --state-pred-kv-source hand \
    --hand-state-dims 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 \
    --hand-pred-future-steps 8 16 \
    --hand-pred-decoder-depth 8 \
    --recon-loss-type mse \
    --decoder-mode self_attention \
    --wandb-project "action-latent-tokenizer-v2" \
    --eval-steps 1000 \
    --report-to wandb

# === Stage 2: VLA Training ===
# target_tokens="all" → VLA denoises 18 tokens (16 time + 2 hand)
# decode시 hand tokens는 무시됨 (hand_in_recon=False)
python scripts/gr00t_finetune_actlat_fm.py \
    --dataset-path $DATA_DIR \
    --output-dir $VLA_CKPT_DIR \
    --data-config fourier_gr1_arms_waist_actlat_fm \
    --embodiment-tag new_embodiment \
    --base-model-path "nvidia/GR00T-N1.5-3B" \
    --run-name "actlat_fm_v2_hand_pred_norecon_mask_fullstate_deep8_gr1_100demos" \
    --num-gpus 2 \
    --batch-size 32 \
    --max-steps 60000 \
    --save-steps 10000 \
    --eval-steps 1000 \
    --actlat-tokenizer-path $TOK_CKPT_DIR/checkpoint-$TOK_STEP \
    --actlat-target-tokens "all" \
    --val-ratio 0.003 \
    --video-backend "decord"
