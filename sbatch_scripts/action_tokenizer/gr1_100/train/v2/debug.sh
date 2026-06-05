#!/bin/bash

set -x
export PATH="$HOME/.local/bin:$PATH"
export WANDB_API_KEY="66a73856614bc24a07523f3fbee42482fcbeffe3"
export HF_TOKEN="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"

source /sjw_alinlab1/home/jungwook/miniconda3/bin/activate gr00t

CKPT_DIR="checkpoints_action_tokenizer/gr1_100demos_v2_full"
DATA_DIR=/storage1/sjw_dataset/dataset/robocasa_gr1_tabletop/sim_100demos

# V2 Full: 모든 loss 활성화
# recon + hand_pred + mask_recon + global contrastive
python scripts/train_action_latent_tokenizer_v2.py \
    --dataset-path $DATA_DIR \
    --output-dir $CKPT_DIR \
    --data-config fourier_gr1_arms_waist \
    --embodiment-tag new_embodiment \
    --run-name "actlat_v2_full" \
    --num-gpus 1 \
    --batch-size 128 \
    --max-steps 100000 \
    --save-steps 20 \
    --num-global-tokens 2 \
    --num-hand-tokens 2 \
    --lambda-recon 1.0 \
    --lambda-hand-pred 0.1 \
    --hand-state-dims 14 15 16 17 18 19 20 21 22 23 24 25 \
    --hand-pred-future-steps 8 16 \
    --hand-pred-decoder-depth 2 \
    --lambda-mask-recon 0.1 \
    --mask-ratio-min 0.2 \
    --mask-ratio-max 0.4 \
    --mask-batch-ratio 0.5 \
    --lambda-global 0.1 \
    --global-loss-mode contrastive \
    --text-encoder-width 256 \
    --text-encoder-layers 4 \
    --text-encoder-heads 4 \
    --fast-tokenizer-path "physical-intelligence/fast" \
    --fast-vocab-size 2048 \
    --recon-loss-type mse \
    --decoder-mode "self_attention" \
    --wandb-project "action-latent-tokenizer-v2" \
    --eval-steps 10 \
    --report-to wandb
