#!/bin/bash
#SBATCH --job-name=gr00t_robotwin_23_full_v2_maskloss
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=sjw_alinlab
#SBATCH --output=out/%j-gr00t_robotwin_23_full_v2_maskloss.out
#SBATCH --error=out/%j-gr00t_robotwin_23_full_v2_maskloss.err
#SBATCH --comment "gr00t_robotwin_23_full_v2_maskloss"

set -x
export PATH="$HOME/.local/bin:$PATH"
export WANDB_API_KEY="66a73856614bc24a07523f3fbee42482fcbeffe3"
export HF_TOKEN="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"

source /sjw_alinlab1/home/jungwook/miniconda3/bin/activate gr00t

# RoboTwin2.0 total — move/open/place 로 시작하고 _500 으로 끝나는 23개 task.
DATA_BASE=/storage1/sjw_dataset/dataset/huggingface/jungwoo/RoboTwin2.0/lerobot/RoboTwin2.0_total
DATASETS=(
    move_can_pot.franka_randomized_500
    move_pillbottle_pad.franka_randomized_500
    move_playingcard_away.franka_randomized_500
    move_stapler_pad.franka_randomized_500
    open_laptop.franka_randomized_500
    open_microwave.franka_randomized_500
    place_a2b_left.franka_randomized_500
    place_a2b_right.franka_randomized_500
    place_bread_basket.franka_randomized_500
    place_bread_skillet.franka_randomized_500
    place_burger_fries.franka_randomized_500
    place_can_basket.franka_randomized_500
    place_cans_plasticbox.franka_randomized_500
    place_container_plate.franka_randomized_500
    place_dual_shoes.franka_randomized_500
    place_empty_cup.franka_randomized_500
    place_fan.franka_randomized_500
    place_mouse_pad.franka_randomized_500
    place_object_basket.franka_randomized_500
    place_object_scale.franka_randomized_500
    place_object_stand.franka_randomized_500
    place_phone_stand.franka_randomized_500
    place_shoe.franka_randomized_500
)
DATA_DIRS=()
for d in "${DATASETS[@]}"; do DATA_DIRS+=("$DATA_BASE/$d"); done

TOK_CKPT_DIR=checkpoints_action_tokenizer/robotwin_23_v2_maskloss
VLA_CKPT_DIR=checkpoints/vla_actlat_fm_robotwin_23/v2_maskloss
TOK_STEP=100000
ABS_TOK_CKPT="/sjw_alinlab1/home/jungwook/Isaac-GR00T/$TOK_CKPT_DIR"

# Stage2 VLA wandb project (robotwin 전용).
export WANDB_PROJECT="gr00t-actlat-fm-robotwin"

# === Stage 1: Tokenizer Training — Masking-only ablation ===
# Ablation: state prediction 축 제거하고 masking 단독 효과만 측정.
#   - num-hand-tokens 0 (hand token 없이 pure time)
#   - lambda-hand-pred 0 (state pred 비활성, default)
#   - lambda-mask-recon 1.0 (masking 활성)
#   - mask-ratio 0.2-0.4, 50% 배치에 적용
# action_dim=16 — RoboTwin bimanual franka.
python scripts/train_action_latent_tokenizer_v2.py \
    --dataset-path "${DATA_DIRS[@]}" \
    --output-dir $TOK_CKPT_DIR \
    --no-resume \
    --data-config robotwin_actlat_fm \
    --embodiment-tag new_embodiment \
    --run-name "actlat_v2_maskloss_robotwin_23" \
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
    --recon-loss-type mse \
    --decoder-mode self_attention \
    --wandb-project "action-latent-tokenizer-v2-robotwin" \
    --eval-steps 1000 \
    --report-to wandb

# === Stage 2: VLA Training (from scratch) ===
# num-hand-tokens=0 이므로 target_tokens="all"은 실질 time 16개만 denoise.
# --no-load-action-head → from-scratch (pretrained head 무시).
python scripts/gr00t_finetune_actlat_fm.py \
    --dataset-path "${DATA_DIRS[@]}" \
    --output-dir $VLA_CKPT_DIR \
    --data-config robotwin_actlat_fm \
    --embodiment-tag new_embodiment \
    --base-model-path "nvidia/GR00T-N1.5-3B" \
    --run-name "actlat_fm_v2_maskloss_robotwin_23" \
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
