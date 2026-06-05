#!/bin/bash
#SBATCH --job-name=actlat_v2_global_contrastive_sbatch
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=sjw_alinlab
#SBATCH --output=out/%j-actlat_v2_global_contrastive_sbatch.out
#SBATCH --error=out/%j-actlat_v2_global_contrastive_sbatch.err
#SBATCH --comment "actlat_v2_global_contrastive_sbatch"

set -x
export PATH="$HOME/.local/bin:$PATH"
export WANDB_API_KEY="66a73856614bc24a07523f3fbee42482fcbeffe3"
export HF_TOKEN="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"

source /sjw_alinlab1/home/jungwook/miniconda3/bin/activate gr00t

CKPT_DIR="checkpoints_action_tokenizer/gr1_100demos_v2_global_contrastive"
DATA_DIR=/storage1/sjw_dataset/dataset/robocasa_gr1_tabletop/sim_100demos

# V2 + Global token contrastive learning
# FAST tokenizer로 action → discrete tokens → text encoder → contrastive with global tokens
python scripts/train_action_latent_tokenizer_v2.py \
    --dataset-path $DATA_DIR \
    --output-dir $CKPT_DIR \
    --data-config fourier_gr1_arms_waist \
    --embodiment-tag new_embodiment \
    --run-name "actlat_v2_global_contrastive" \
    --num-gpus 2 \
    --batch-size 1024 \
    --max-steps 100000 \
    --save-steps 5000 \
    --num-global-tokens 2 \
    --num-hand-tokens 0 \
    --lambda-recon 1.0 \
    --lambda-global 1 \
    --global-loss-mode contrastive \
    --text-encoder-width 256 \
    --text-encoder-layers 4 \
    --text-encoder-heads 4 \
    --fast-tokenizer-path "physical-intelligence/fast" \
    --fast-vocab-size 2048 \
    --recon-loss-type mse \
    --decoder-mode "self_attention" \
    --wandb-project "action-latent-tokenizer-v2" \
    --eval-steps 1000 \
    --report-to wandb \
    --resume
