#!/bin/bash
#SBATCH --job-name=action_tokenizer_gr1_100demos_dimwise_globaltok2_sbatch
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=sjw_alinlab
#SBATCH --output=out/%j-action_tokenizer_gr1_100demos_dimwise_globaltok2_sbatch.out
#SBATCH --error=out/%j-action_tokenizer_gr1_100demos_dimwise_globaltok2_sbatch.err
#SBATCH --comment "action_tokenizer_gr1_100demos_dimwise_globaltok2_sbatch"

set -x
export PATH="$HOME/.local/bin:$PATH"
export WANDB_API_KEY="66a73856614bc24a07523f3fbee42482fcbeffe3"
export HF_TOKEN="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"


source /sjw_alinlab1/home/jungwook/miniconda3/bin/activate gr00t

CKPT_DIR="checkpoints_action_tokenizer/gr1_100demos_dimwise_globaltok2"
DATA_DIR=/storage1/sjw_dataset/dataset/robocasa_gr1_tabletop/sim_100demos

# DimWise GlobalTok: 2 global tokens + masked recon loss
# mask는 action dimension 축 기준 (DimensionMasking)으로 적용됨
python scripts/train_action_latent_tokenizer.py \
    --dataset-path $DATA_DIR \
    --output-dir $CKPT_DIR \
    --data-config fourier_gr1_arms_waist \
    --embodiment-tag new_embodiment \
    --run-name "action_tokenizer_gr1_100demos_dimwise_globaltok2" \
    --tokenizer-type dimwise \
    --num-gpus 2 \
    --batch-size 1024 \
    --max-steps 100000 \
    --save-steps 5000 \
    --num-global-tokens 2 \
    --num-hand-tokens 0 \
    --lambda-recon 1.0 \
    --lambda-masked 1.0 \
    --mask-ratio-min 0.4 \
    --mask-ratio-max 0.6 \
    --mask-mode random \
    --recon-loss-type mse \
    --masked_decoder_mode "self_attention" \
    --decoder_mode "self_attention" \
    --wandb-project "action-latent-tokenizer" \
    --eval-steps 1000 \
    --report-to wandb
