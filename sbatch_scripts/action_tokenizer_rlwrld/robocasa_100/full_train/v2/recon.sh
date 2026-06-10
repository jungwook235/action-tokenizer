#!/bin/bash
#SBATCH --job-name=robocasa_gr00t_action_tokenizer_full_v2_recon
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=sjw_alinlab,h100
#SBATCH --output=out/%j-robocasa_gr00t_action_tokenizer_full_v2_recon.out
#SBATCH --error=out/%j-robocasa_gr00t_action_tokenizer_full_v2_recon.err
#SBATCH --comment "robocasa_gr00t_action_tokenizer_full_v2_recon"

set -x
export PATH="$HOME/.local/bin:$PATH"
export WANDB_API_KEY="66a73856614bc24a07523f3fbee42482fcbeffe3"
export HF_TOKEN="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"

source /sjw_alinlab1/home/jungwook/miniconda3/bin/activate gr00t

DATA_DIR=/storage1/sjw_dataset/dataset/huggingface/lerobot/shared/robocasa_preprocessed/robocasa_mg_gr00t_100
TOK_CKPT_DIR=checkpoints_action_tokenizer/robocasa_100demos_v2_recon
VLA_CKPT_DIR=checkpoints/vla_actlat_fm_robocasa_100demos/v2_recon
TOK_STEP=100000
ABS_TOK_CKPT="/sjw_alinlab1/home/jungwook/Isaac-GR00T/$TOK_CKPT_DIR"

# Stage2 VLA wandb project (robocasa 전용).
export WANDB_PROJECT="gr00t-actlat-fm-robocasa"

# === Stage 1: Tokenizer Training — Recon-only baseline ===
# v2 grid에서 빠져있던 pure recon baseline.
#   - num-hand-tokens 0, num-global-tokens 0 (hand/global 비활성)
#   - lambda-recon 1.0 (default), 그 외 lambda 모두 0 (default)
#     → mask path / state-pred / global head 전부 비활성
# action reconstruction 단독 효과 측정용. mask_recon / state_pred 변종들의 lower-bound 레퍼런스.
python scripts/train_action_latent_tokenizer_v2.py \
    --dataset-path $DATA_DIR \
    --output-dir $TOK_CKPT_DIR \
    --no-resume \
    --data-config single_panda_gripper \
    --embodiment-tag new_embodiment \
    --run-name "actlat_v2_recon_robocasa_100demos" \
    --num-gpus 2 \
    --batch-size 1024 \
    --max-steps $TOK_STEP \
    --save-steps 5000 \
    --num-global-tokens 0 \
    --num-hand-tokens 0 \
    --lambda-recon 1.0 \
    --recon-loss-type mse \
    --decoder-mode self_attention \
    --wandb-project "action-latent-tokenizer-v2-robocasa" \
    --eval-steps 1000 \
    --report-to wandb

# === Stage 2: VLA Training ===
# num-hand-tokens=0 이므로 target_tokens="all"은 실질 time 16개만 denoise.
python scripts/gr00t_finetune_actlat_fm.py \
    --dataset-path $DATA_DIR \
    --output-dir $VLA_CKPT_DIR \
    --data-config single_panda_gripper_actlat_fm \
    --embodiment-tag new_embodiment \
    --base-model-path "nvidia/GR00T-N1.5-3B" \
    --run-name "actlat_fm_v2_recon_robocasa_100demos" \
    --num-gpus 2 \
    --batch-size 32 \
    --max-steps 60000 \
    --save-steps 10000 \
    --eval-steps 1000 \
    --actlat-tokenizer-path "$ABS_TOK_CKPT/checkpoint-$TOK_STEP" \
    --actlat-target-tokens "all" \
    --val-ratio 0.003 \
    --video-backend "decord"
