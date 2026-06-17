#!/bin/bash
#SBATCH --job-name=full_v3_gr1_mask_recon_ln_bn16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b200:4
#SBATCH --nodes=1
#SBATCH --qos=preempt
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120
#SBATCH --time=72:00:00
#SBATCH --output=/NHNHOME/data/wook/action-tokenizer/slurm/logs_1000/full_v3_gr1_mask_recon_ln_bn16_%j.out
#SBATCH --error=/NHNHOME/data/wook/action-tokenizer/slurm/logs_1000/full_v3_gr1_mask_recon_ln_bn16_%j.err


set -x
export PATH="$HOME/.local/bin:$PATH"
export WANDB_PROJECT=Action-Tokenizer-GR1-1000demos
export WANDB_API_KEY="66a73856614bc24a07523f3fbee42482fcbeffe3"
export HF_TOKEN="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"

BASE_DIR=/NHNHOME/data/wook/action-tokenizer
cd $BASE_DIR

source /NHNHOME/data/wook/miniconda3/bin/activate gr00t-actlat
export PATH="$CONDA_PREFIX/bin:$PATH"
hash -r
echo "[env-check] which python=$(which python)"
echo "[env-check] CONDA_PREFIX=$CONDA_PREFIX"
python -c "import sys, transformers; print('exe=', sys.executable, 'transformers=', transformers.__version__)"

DATA_DIR=(/NHNHOME/data/wook/dataset/gr00t_unified/gr1_unified.*)
TOK_CKPT_DIR=checkpoints_action_tokenizer/gr1_1000demos_v3_mask_recon_ln_bn16
VLA_CKPT_DIR=checkpoints/vla_actlat_fm_gr1_1000demos/v3_mask_recon_ln_bn16
TOK_STEP=100000
ABS_TOK_CKPT="/NHNHOME/data/wook/action-tokenizer/$TOK_CKPT_DIR"

# Fixed validation split shared across all v3 experiments on this dataset.
#FIXED_VAL_PATH=/sjw_alinlab1/home/jungwook/Isaac-GR00T/experiments/fixed_val_splits/gr1_100demos.json

# === Stage 1: Tokenizer Training — v3 + LayerNorm + Bottleneck(token_dim=64) ===
# v2_headfix/recon_fs (= v2_base, recon-only) 와 동일한 하이퍼파라미터에 v3 옵션
# 두 개만 추가:
#   - --encoder-output-layernorm        : encoder transformer 출력에 LayerNorm
#   - --use-bottleneck --token-dim 64   : VTP 스타일 bottleneck (latent dim emb_dim → 64)
# latent_noise_std 는 default 0.0 (= noise 비활성). recon_loss_type mse, data-config
# 도 v2 그대로 (q99 미사용). fixed val split 은 v3 컨벤션.
# pure recon: no mask, no state-pred.
python scripts/train_action_latent_tokenizer_v3.py \
    --dataset-path "${DATA_DIR[@]}" \
    --output-dir $TOK_CKPT_DIR \
    --no-resume \
    --data-config fourier_gr1_arms_waist \
    --embodiment-tag new_embodiment \
    --run-name "actlat_v3_gr1_mask_recon_ln_bn16_1000demos" \
    --num-gpus 4 \
    --batch-size 512 \
    --max-steps $TOK_STEP \
    --save-steps 10000 \
    --save-total-limit 3 \
    --no-cache-dataset \
    --dataloader-num-workers 16 \
    --num-global-tokens 0 \
    --num-hand-tokens 0 \
    --lambda-recon 1.0 \
    --lambda-mask-recon 1.0 \
    --mask-ratio-min 0.2 \
    --mask-ratio-max 0.4 \
    --mask-batch-ratio 0.5 \
    --recon-loss-type mse \
    --decoder-mode self_attention \
    --encoder-output-layernorm \
    --use-bottleneck \
    --token-dim 16 \
    --use-fixed-val \
    --wandb-project "Action-Tokenizer-GR1-1000demos-tokenizer" \
    --eval-steps 1000 \
    --report-to wandb

# === Stage 2: VLA Training ===
# wrapper 가 _is_v3 + encoder.output_down_proj 를 자동 감지하여
# wrapper.emb_dim = 64 로 노출 → action head 가 64-dim latent 위에서 학습.
# decode_latent 호출 시 input_up_proj(64→256) 를 거쳐 v2 decoder 로 복원됨.
# num-hand-tokens=0 이므로 target_tokens="all"은 실질 time 16개만 denoise.
python scripts/gr00t_finetune_actlat_fm.py \
    --dataset-path "${DATA_DIR[@]}" \
    --output-dir $VLA_CKPT_DIR \
    --data-config fourier_gr1_arms_waist_actlat_fm \
    --embodiment-tag new_embodiment \
    --base-model-path "nvidia/GR00T-N1.5-3B" \
    --run-name "actlat_fm_v3_gr1_mask_recon_ln_bn16_1000demos" \
    --num-gpus 4 \
    --batch-size 128 \
    --max-steps 60000 \
    --save-steps 5000 \
    --save-total-limit 3 \
    --eval-steps 1000 \
    --actlat-tokenizer-path "$ABS_TOK_CKPT/checkpoint-$TOK_STEP" \
    --actlat-target-tokens "all" \
    --val-ratio 0.003 \
    --use-fixed-val \
    --video-backend "decord"
