#!/bin/bash
#SBATCH --job-name=nactlat_fm_swx_100demos_baseline_fs_sbatch
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=sjw_alinlab
#SBATCH --output=out/%j-nactlat_fm_swx_100demos_baseline_fs_sbatch.out
#SBATCH --error=out/%j-nactlat_fm_swx_100demos_baseline_fs_sbatch.err
#SBATCH --comment "nactlat_fm_swx_100demos_baseline_fs_sbatch"

set -x
export PATH="$HOME/.local/bin:$PATH"
export WANDB_API_KEY="66a73856614bc24a07523f3fbee42482fcbeffe3"
export HF_TOKEN="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"

source /sjw_alinlab1/home/jungwook/miniconda3/bin/activate gr00t

CKPT_DIR="checkpoints/vla_nactlat_fm_swx_100demos/baseline_fs"
DATA_DIR=/storage1/sjw_dataset/dataset/huggingface/lerobot/shared/bridge_orig_lerobot

# swx 전용 wandb project (actlat_fm 실험들과 동일 project 에 baseline 을 같이 기록해 비교 편의).
export WANDB_PROJECT="gr00t-actlat-fm-swx"

# --mode "vla" → tokenizer 없이 GR00T 원본 FlowmatchingActionHead 로 학습.
# data-config 는 actlat_fm 과 동일하게 쓰되(액션 정규화 uniform min_max + max_action_dim=12),
# 이는 비교 대상 실험들과 입력 도메인을 정확히 일치시키기 위함이다.
python scripts/gr00t_finetune_actlat_fm.py \
    --dataset-path $DATA_DIR \
    --output-dir $CKPT_DIR \
    --data-config bridge_flare_kty_actlat_fm \
    --embodiment-tag new_embodiment \
    --base-model-path "nvidia/GR00T-N1.5-3B" \
    --run-name "nactlat_baseline_fs_swx_100demos" \
    --mode "vla" \
    --num-gpus 2 \
    --batch-size 32 \
    --max-steps 60000 \
    --save-steps 10000 \
    --eval-steps 1000 \
    --val-ratio 0.003 \
    --no-load-action-head \
    --video-backend "torchvision_av"
