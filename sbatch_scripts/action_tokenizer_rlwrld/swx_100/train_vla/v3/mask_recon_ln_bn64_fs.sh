#!/bin/bash
#SBATCH --job-name=full_v3_swx_mask_recon_ln_bn64_fs
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=sjw_alinlab
#SBATCH --output=out/%j-full_v3_swx_mask_recon_ln_bn64_fs.out
#SBATCH --error=out/%j-full_v3_swx_mask_recon_ln_bn64_fs.err
#SBATCH --comment "full_v3_swx_mask_recon_ln_bn64_fs"

set -x
export PATH="$HOME/.local/bin:$PATH"
export WANDB_API_KEY="66a73856614bc24a07523f3fbee42482fcbeffe3"
export HF_TOKEN="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"

source /sjw_alinlab1/home/jungwook/miniconda3/bin/activate gr00t

DATA_DIR=/storage1/sjw_dataset/dataset/huggingface/lerobot/shared/bridge_orig_lerobot
TOK_CKPT_DIR=checkpoints_action_tokenizer/swx_100demos_v3_mask_recon_ln_bn64
VLA_CKPT_DIR=checkpoints/vla_actlat_fm_swx_100demos/v3_mask_recon_ln_bn64_fs
TOK_STEP=100000
ABS_TOK_CKPT="/sjw_alinlab1/home/jungwook/Isaac-GR00T/$TOK_CKPT_DIR"

# Fixed validation split (shared across all v3 experiments on this dataset).
FIXED_VAL_PATH=/sjw_alinlab1/home/jungwook/Isaac-GR00T/experiments/fixed_val_splits/swx_100demos.json

# Stage2 VLA wandb project (swx 전용).
export WANDB_PROJECT="gr00t-actlat-fm-swx"


# === Stage 2: VLA Training ===
# wrapper 가 _is_v3 + encoder.output_down_proj 를 자동 감지하여
# wrapper.emb_dim = 64 로 노출 → action head 가 64-dim latent 위에서 학습.
# decode_latent 호출 시 input_up_proj(64→256) 를 거쳐 v2 decoder 로 복원됨.
# num-hand-tokens=0 이므로 target_tokens="all"은 실질 time 16개만 denoise.
python scripts/gr00t_finetune_actlat_fm.py \
    --dataset-path $DATA_DIR \
    --output-dir $VLA_CKPT_DIR \
    --data-config bridge_flare_kty_actlat_fm \
    --embodiment-tag new_embodiment \
    --base-model-path "nvidia/GR00T-N1.5-3B" \
    --run-name "actlat_fm_v3_swx_mask_recon_ln_bn64_fs_100demos" \
    --num-gpus 2 \
    --batch-size 32 \
    --max-steps 60000 \
    --save-steps 10000 \
    --eval-steps 1000 \
    --actlat-tokenizer-path "$ABS_TOK_CKPT/checkpoint-$TOK_STEP" \
    --actlat-target-tokens "all" \
    --val-ratio 0.003 \
    --no-load-action-head \
    --video-backend "torchvision_av"
