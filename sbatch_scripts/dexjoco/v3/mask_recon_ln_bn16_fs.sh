#!/bin/bash
#SBATCH --job-name=full_train_v3_dexjoco_dual_arm_mask_recon_ln_bottleneck16_fs
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=h100
#SBATCH --output=out/%j-full_train_v3_dexjoco_dual_arm_mask_recon_ln_bottleneck16_fs.out
#SBATCH --error=out/%j-full_train_v3_dexjoco_dual_arm_mask_recon_ln_bottleneck16_fs.err
#SBATCH --comment "full_train_v3_dexjoco_dual_arm_mask_recon_ln_bottleneck16_fs"

set -x
export PATH="$HOME/.local/bin:$PATH"
export WANDB_API_KEY="66a73856614bc24a07523f3fbee42482fcbeffe3"
export HF_TOKEN="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"
export WANDB_PROJECT=Action-Tokenizer-DexJoCo-DualArm

source /sjw_alinlab1/home/jungwook/miniconda3/bin/activate gr00t-actlat

DATA_DIR=("/sjw_alinlab1/home/jungwook/dataset/dexjoco_lerobot/v20/bimanual/bimanual_assembly"
"/sjw_alinlab1/home/jungwook/dataset/dexjoco_lerobot/v20/bimanual/bimanual_hanoi"
"/sjw_alinlab1/home/jungwook/dataset/dexjoco_lerobot/v20/bimanual/bimanual_microwave_cook"
"/sjw_alinlab1/home/jungwook/dataset/dexjoco_lerobot/v20/bimanual/bimanual_photograph"
"/sjw_alinlab1/home/jungwook/dataset/dexjoco_lerobot/v20/bimanual/bimanual_unlock_ipad"
)
TOK_CKPT_DIR=checkpoints_action_tokenizer/dexjoco_dual_arm_v3_mask_recon_ln_bn16
VLA_CKPT_DIR=checkpoints/vla_actlat_fm_dexjoco_dual_arm/v3_mask_recon_ln_bn16_fs
TOK_STEP=100000
ABS_TOK_CKPT="/sjw_alinlab1/home/jungwook/action_tokenizer/$TOK_CKPT_DIR"


# === Stage 1: Tokenizer Training — v3 + LayerNorm + Bottleneck(token_dim=16) ===
# v2/mask_recon (gr1_100demos_v2_maskloss) 와 동일한 하이퍼파라미터에 v3 옵션
# 두 개만 추가:
#   - --encoder-output-layernorm        : encoder transformer 출력에 LayerNorm
#   - --use-bottleneck --token-dim 16   : VTP 스타일 bottleneck (latent dim emb_dim → 16)
# latent_noise_std 는 default 0.0 (= noise 비활성). recon_loss_type 는 v2 mask_recon
# 그대로 mse 유지. data-config 도 v2 그대로 (q99 미사용). fixed val split 은 v3 컨벤션.
#python scripts/train_action_latent_tokenizer_v3.py \
#    --dataset-path "${DATA_DIR[@]}" \
#    --output-dir $TOK_CKPT_DIR \
#    --no-resume \
#    --data-config dexjoco_dual_arm \
#    --embodiment-tag new_embodiment \
#    --run-name "actlat_v3_dexjoco_dual_arm_mask_recon_ln_bn16" \
#    --num-gpus 2 \
#    --batch-size 1024 \
#    --max-steps $TOK_STEP \
#    --save-steps 5000 \
#    --num-global-tokens 0 \
#    --num-hand-tokens 0 \
#    --lambda-recon 1.0 \
#    --lambda-mask-recon 1.0 \
#    --mask-ratio-min 0.2 \
#    --mask-ratio-max 0.4 \
#    --mask-batch-ratio 0.5 \
#    --recon-loss-type mse \
#    --decoder-mode self_attention \
#    --encoder-output-layernorm \
#    --use-bottleneck \
#    --token-dim 16 \
#    --use-fixed-val \
#    --fixed-val-path "$FIXED_VAL_PATH" \
#    --wandb-project "action-latent-tokenizer-v3-dexjoco-dual-arm" \
#    --eval-steps 1000 \
#    --report-to wandb

# === Stage 2: VLA Training ===
# wrapper 가 _is_v3 + encoder.output_down_proj 를 자동 감지하여
# wrapper.emb_dim = 64 로 노출 → action head 가 64-dim latent 위에서 학습.
# decode_latent 호출 시 input_up_proj(64→256) 를 거쳐 v2 decoder 로 복원됨.
# num-hand-tokens=0 이므로 target_tokens="all"은 실질 time 16개만 denoise.
python scripts/gr00t_finetune_actlat_fm.py \
    --dataset-path "${DATA_DIR[@]}" \
    --output-dir $VLA_CKPT_DIR \
    --data-config dexjoco_dual_arm \
    --embodiment-tag new_embodiment \
    --base-model-path "nvidia/GR00T-N1.5-3B" \
    --run-name "actlat_fm_v3_dexjoco_dual_arm_mask_recon_ln_bn16_fs" \
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
