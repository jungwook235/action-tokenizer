#!/bin/bash

set -x
export PATH="$HOME/.local/bin:$PATH"
export WANDB_API_KEY="66a73856614bc24a07523f3fbee42482fcbeffe3"
export HF_TOKEN="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"


source /sjw_alinlab1/home/jungwook/miniconda3/bin/activate gr00t

CKPT_DIR="checkpoints_action_tokenizer/gr1_100demos_handtok2"
DATA_DIR=/storage1/sjw_dataset/dataset/robocasa_gr1_tabletop/sim_100demos

# HandTok: 2 hand tokens — recon path 2 (time + hand tokens) trains hand tokens
# recon2 is activated when num_hand_tokens > 0 and lambda_recon > 0
python scripts/train_action_latent_tokenizer.py \
    --dataset-path $DATA_DIR \
    --output-dir $CKPT_DIR \
    --data-config fourier_gr1_arms_waist \
    --embodiment-tag new_embodiment \
    --run-name "action_tokenizer_gr1_100demos_handtok2" \
    --num-gpus 2 \
    --batch-size 4096 \
    --max-steps 50000 \
    --save-steps 5000 \
    --num-global-tokens 0 \
    --num-hand-tokens 2 \
    --lambda-recon 1.0 \
    --lambda-masked 0.0 \
    --recon-loss-type mse \
    --hand-action-dims 14 15 16 17 18 19 20 21 22 23 24 25 \
    --hand-loss-weight 2.0 \
    --masked_decoder_mode "self_attention" \
    --decoder_mode "self_attention" \
    --wandb-project "action-latent-tokenizer" \
    --eval-steps 500 \
    --report-to wandb
