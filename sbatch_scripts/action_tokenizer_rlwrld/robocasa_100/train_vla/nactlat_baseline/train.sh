#!/bin/bash
#SBATCH --job-name=rcasa_nactlat_fm_baseline
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=sjw_alinlab
#SBATCH --output=out/%j-rcasa_nactlat_fm_baseline.out
#SBATCH --error=out/%j-rcasa_nactlat_fm_baseline.err
#SBATCH --comment "rcasa_nactlat_fm_baseline"

set -x
export PATH="$HOME/.local/bin:$PATH"
export WANDB_API_KEY="66a73856614bc24a07523f3fbee42482fcbeffe3"
export HF_TOKEN="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"

source /sjw_alinlab1/home/jungwook/miniconda3/bin/activate gr00t

CKPT_DIR="checkpoints/vla_nactlat_fm_robocasa_100demos/baseline"
DATA_DIR=/storage1/sjw_dataset/dataset/huggingface/lerobot/shared/robocasa_preprocessed/robocasa_mg_gr00t_100

# robocasa 전용 wandb project (actlat_fm 실험들과 동일 project 에 baseline 을 같이 기록해 비교 편의).
export WANDB_PROJECT="gr00t-actlat-fm-robocasa"

# --mode "vla" → tokenizer 없이 GR00T 원본 FlowmatchingActionHead 로 학습.
# data-config 는 actlat_fm 과 동일하게 쓰되(액션 정규화 uniform min_max + max_action_dim=12),
# 이는 비교 대상 실험들과 입력 도메인을 정확히 일치시키기 위함이다.
python scripts/gr00t_finetune_actlat_fm.py \
    --dataset-path $DATA_DIR \
    --output-dir $CKPT_DIR \
    --data-config single_panda_gripper_actlat_fm \
    --embodiment-tag new_embodiment \
    --base-model-path "nvidia/GR00T-N1.5-3B" \
    --run-name "nactlat_baseline_robocasa_100demos" \
    --mode "vla" \
    --num-gpus 2 \
    --batch-size 32 \
    --max-steps 60000 \
    --save-steps 10000 \
    --eval-steps 1000 \
    --val-ratio 0.003 \
    --video-backend "decord"
