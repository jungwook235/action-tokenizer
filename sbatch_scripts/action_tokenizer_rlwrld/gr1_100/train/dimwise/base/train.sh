#!/bin/bash
#SBATCH --job-name=action_tokenizer_gr1_100demos_dimwise_base_sbatch
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=sjw_alinlab
#SBATCH --output=out/%j-action_tokenizer_gr1_100demos_dimwise_base_sbatch.out
#SBATCH --error=out/%j-action_tokenizer_gr1_100demos_dimwise_base_sbatch.err
#SBATCH --comment "action_tokenizer_gr1_100demos_dimwise_base_sbatch"

set -x
export PATH="$HOME/.local/bin:$PATH"
export WANDB_API_KEY="66a73856614bc24a07523f3fbee42482fcbeffe3"
export HF_TOKEN="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"


source /sjw_alinlab1/home/jungwook/miniconda3/bin/activate gr00t

CKPT_DIR="checkpoints_action_tokenizer/gr1_100demos_dimwise_base"
DATA_DIR=/storage1/sjw_dataset/dataset/robocasa_gr1_tabletop/sim_100demos

# DimWise Base: 각 latent 토큰 = 하나의 action dimension의 전체 time 정보
# TimeWise Base와 달리 [B,T,D] → transpose → [B,D,T] → Linear(T→E) 인코딩
python scripts/train_action_latent_tokenizer.py \
    --dataset-path $DATA_DIR \
    --output-dir $CKPT_DIR \
    --data-config fourier_gr1_arms_waist \
    --embodiment-tag new_embodiment \
    --run-name "action_tokenizer_gr1_100demos_dimwise_base" \
    --tokenizer-type dimwise \
    --num-gpus 2 \
    --batch-size 1024 \
    --max-steps 100000 \
    --save-steps 5000 \
    --num-global-tokens 0 \
    --num-hand-tokens 0 \
    --lambda-recon 1.0 \
    --lambda-masked 0.0 \
    --recon-loss-type mse \
    --decoder_mode "self_attention" \
    --wandb-project "action-latent-tokenizer" \
    --eval-steps 1000 \
    --report-to wandb
