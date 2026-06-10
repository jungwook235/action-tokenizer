#!/bin/bash
#SBATCH --job-name=full_v3_mask_recon_q99_l1
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=sjw_alinlab
#SBATCH --output=out/%j-full_v3_mask_recon_q99_l1.out
#SBATCH --error=out/%j-full_v3_mask_recon_q99_l1.err
#SBATCH --comment "full_v3_mask_recon_q99_l1"

set -x
export PATH="$HOME/.local/bin:$PATH"
export WANDB_API_KEY="66a73856614bc24a07523f3fbee42482fcbeffe3"
export HF_TOKEN="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"

source /sjw_alinlab1/home/jungwook/miniconda3/bin/activate gr00t

DATA_DIR=/storage1/sjw_dataset/dataset/robocasa_gr1_tabletop/sim_100demos
TOK_CKPT_DIR=checkpoints_action_tokenizer/gr1_100demos_v3_mask_recon_q99_l1
VLA_CKPT_DIR=checkpoints/vla_actlat_fm_gr1_100demos/v3_mask_recon_q99_l1
TOK_STEP=100000
ABS_TOK_CKPT="/sjw_alinlab1/home/jungwook/Isaac-GR00T/$TOK_CKPT_DIR"

# Fixed validation split shared across all v3 experiments on this dataset.
FIXED_VAL_PATH=/sjw_alinlab1/home/jungwook/Isaac-GR00T/experiments/fixed_val_splits/gr1_100demos.json

# === Stage 1: Tokenizer Training ===
# v2/mask_recon 와 동일한 하이퍼파라미터에 v3 신규 옵션을 추가 (단, latent noise 미사용):
#   --encoder-output-layernorm, --recon-loss-type l1,
#   --use-fixed-val + --fixed-val-path, --data-config fourier_gr1_arms_waist_q99
# latent_noise_std 는 기본값 0.0 (= 노이즈 주입 안 함) 사용.
python scripts/train_action_latent_tokenizer_v3.py \
    --dataset-path $DATA_DIR \
    --output-dir $TOK_CKPT_DIR \
    --no-resume \
    --data-config fourier_gr1_arms_waist_q99 \
    --embodiment-tag new_embodiment \
    --run-name "actlat_v3_mask_recon_q99_l1" \
    --num-gpus 2 \
    --batch-size 1024 \
    --max-steps $TOK_STEP \
    --save-steps 5000 \
    --num-global-tokens 0 \
    --num-hand-tokens 0 \
    --lambda-recon 1.0 \
    --lambda-mask-recon 1.0 \
    --mask-ratio-min 0.2 \
    --mask-ratio-max 0.4 \
    --mask-batch-ratio 0.5 \
    --recon-loss-type l1 \
    --decoder-mode self_attention \
    --encoder-output-layernorm \
    --use-fixed-val \
    --fixed-val-path "$FIXED_VAL_PATH" \
    --wandb-project "action-latent-tokenizer-v3" \
    --eval-steps 1000 \
    --report-to wandb

# === Stage 2: VLA Training ===
python scripts/gr00t_finetune_actlat_fm.py \
    --dataset-path $DATA_DIR \
    --output-dir $VLA_CKPT_DIR \
    --data-config fourier_gr1_arms_waist_actlat_fm_q99 \
    --embodiment-tag new_embodiment \
    --base-model-path "nvidia/GR00T-N1.5-3B" \
    --run-name "actlat_fm_v3_mask_recon_q99_l1_gr1_100demos" \
    --num-gpus 2 \
    --batch-size 32 \
    --max-steps 60000 \
    --save-steps 10000 \
    --eval-steps 1000 \
    --actlat-tokenizer-path "$ABS_TOK_CKPT/checkpoint-$TOK_STEP" \
    --actlat-target-tokens "all" \
    --val-ratio 0.003 \
    --video-backend "decord"
