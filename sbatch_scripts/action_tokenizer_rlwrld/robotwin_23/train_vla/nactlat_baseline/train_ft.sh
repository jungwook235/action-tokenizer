#!/bin/bash
#SBATCH --job-name=nactlat_fm_robotwin_23_baseline_ft_sbatch
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=sjw_alinlab
#SBATCH --output=out/%j-nactlat_fm_robotwin_23_baseline_ft_sbatch.out
#SBATCH --error=out/%j-nactlat_fm_robotwin_23_baseline_ft_sbatch.err
#SBATCH --comment "nactlat_fm_robotwin_23_baseline_ft_sbatch"

set -x
export PATH="$HOME/.local/bin:$PATH"
export WANDB_API_KEY="66a73856614bc24a07523f3fbee42482fcbeffe3"
export HF_TOKEN="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"

source /sjw_alinlab1/home/jungwook/miniconda3/bin/activate gr00t

CKPT_DIR="checkpoints/vla_nactlat_fm_robotwin_23/baseline_ft"

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

# robotwin 전용 wandb project (actlat_fm 실험들과 동일 project 에 baseline 을 같이 기록).
export WANDB_PROJECT="gr00t-actlat-fm-robotwin"

# --mode "vla" → tokenizer 없이 GR00T 원본 FlowmatchingActionHead 로 학습.
# --no-load-action-head → from-scratch (pretrained head 무시).
# data-config robotwin_actlat_fm 는 q99 정규화 + max_action_dim=16, 비교 대상 실험들과
# 입력 도메인을 정확히 일치.
python scripts/gr00t_finetune_actlat_fm.py \
    --dataset-path "${DATA_DIRS[@]}" \
    --output-dir $CKPT_DIR \
    --data-config robotwin_actlat_fm \
    --embodiment-tag new_embodiment \
    --base-model-path "nvidia/GR00T-N1.5-3B" \
    --run-name "nactlat_baseline_ft_robotwin_23" \
    --mode "vla" \
    --num-gpus 2 \
    --batch-size 32 \
    --max-steps 60000 \
    --save-steps 10000 \
    --eval-steps 1000 \
    --val-ratio 0.003 \
    --video-backend "torchvision_av"
