#!/bin/bash
#SBATCH --job-name=full_train_v3_gr1_100demos_recon_ln_bottleneck64_ctok4_fs
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=sjw_alinlab
#SBATCH --output=out/%j-full_v3_gr1_recon_ln_bn64_ctok4_fs.out
#SBATCH --error=out/%j-full_v3_gr1_recon_ln_bn64_ctok4_fs.err
#SBATCH --comment "full_v3_gr1_recon_ln_bn64_ctok4_fs"

set -x
export PATH="$HOME/.local/bin:$PATH"
export WANDB_API_KEY="66a73856614bc24a07523f3fbee42482fcbeffe3"
export HF_TOKEN="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"

source /sjw_alinlab1/home/jungwook/miniconda3/bin/activate gr00t

DATA_DIR=/storage1/sjw_dataset/dataset/robocasa_gr1_tabletop/sim_100demos
TOK_CKPT_DIR=checkpoints_action_tokenizer/gr1_100demos_v3_recon_ln_bn64_ctok4
VLA_CKPT_DIR=checkpoints/vla_actlat_fm_gr1_100demos/v3_recon_ln_bn64_ctok4_fs
TOK_STEP=100000
ABS_TOK_CKPT="/sjw_alinlab1/home/jungwook/Isaac-GR00T/$TOK_CKPT_DIR"

# Fixed validation split shared across all v3 experiments on this dataset.
FIXED_VAL_PATH=/sjw_alinlab1/home/jungwook/Isaac-GR00T/experiments/fixed_val_splits/gr1_100demos.json

# === Stage 1: Tokenizer Training — v3 + LayerNorm + Bottleneck(token_dim=64) + Time-compress(x4) ===
# recon_ln_bn64_fs 와 동일한 하이퍼파라미터에 time 축 토큰 압축 옵션 하나만 추가:
#   - --compress-token 4 : encoder 입력단 Conv1d(kernel=stride=4)로 time 토큰을
#                          16 → 4 개로 압축. decoder 의 sub-pixel head 가 마지막에
#                          다시 16 step 으로 복원. bottleneck(token_dim=64)과 직교
#                          (압축은 토큰 개수, bottleneck 은 토큰 차원).
# latent_noise_std default 0.0, recon_loss_type mse, data-config 도 v2 그대로(q99 미사용).
# pure recon: no mask, no state-pred.
python scripts/train_action_latent_tokenizer_v3.py \
    --dataset-path $DATA_DIR \
    --output-dir $TOK_CKPT_DIR \
    --no-resume \
    --data-config fourier_gr1_arms_waist \
    --embodiment-tag new_embodiment \
    --run-name "actlat_v3_gr1_recon_ln_bn64_ctok4" \
    --num-gpus 2 \
    --batch-size 1024 \
    --max-steps $TOK_STEP \
    --save-steps 5000 \
    --num-global-tokens 0 \
    --num-hand-tokens 0 \
    --lambda-recon 1.0 \
    --recon-loss-type mse \
    --decoder-mode self_attention \
    --encoder-output-layernorm \
    --use-bottleneck \
    --token-dim 64 \
    --compress-token 4 \
    --use-fixed-val \
    --fixed-val-path "$FIXED_VAL_PATH" \
    --wandb-project "action-latent-tokenizer-v3-gr1" \
    --eval-steps 1000 \
    --report-to wandb

# === Stage 2: VLA Training ===
# wrapper 가 _is_v3 + encoder.output_down_proj + encoder.time_conv 를 자동 감지하여
# wrapper.emb_dim = 64 (token_dim), num_main_tokens = 16//4 = 4 로 노출.
# → action head 가 64-dim latent 위에서 토큰 4개만 예측하도록 자동 설정됨.
# decode_latent 호출 시 input_up_proj(64→256) → decoder → sub-pixel head 로 16 step 복원.
# num-hand-tokens=0 이므로 target_tokens="all"은 실질 압축된 time 4개만 denoise.
python scripts/gr00t_finetune_actlat_fm.py \
    --dataset-path $DATA_DIR \
    --output-dir $VLA_CKPT_DIR \
    --data-config fourier_gr1_arms_waist_actlat_fm \
    --embodiment-tag new_embodiment \
    --base-model-path "nvidia/GR00T-N1.5-3B" \
    --run-name "actlat_fm_v3_gr1_recon_ln_bn64_ctok4_fs_100demos" \
    --num-gpus 2 \
    --batch-size 32 \
    --max-steps 60000 \
    --save-steps 10000 \
    --eval-steps 1000 \
    --actlat-tokenizer-path "$ABS_TOK_CKPT/checkpoint-$TOK_STEP" \
    --actlat-target-tokens "all" \
    --val-ratio 0.003 \
    --no-load-action-head \
    --video-backend "decord"
