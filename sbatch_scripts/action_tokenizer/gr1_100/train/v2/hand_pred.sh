#!/bin/bash
#SBATCH --job-name=actlat_v2_hand_pred_sbatch
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=sjw_alinlab
#SBATCH --output=out/%j-actlat_v2_hand_pred_sbatch.out
#SBATCH --error=out/%j-actlat_v2_hand_pred_sbatch.err
#SBATCH --comment "actlat_v2_hand_pred_sbatch"

set -x
export PATH="$HOME/.local/bin:$PATH"
export WANDB_API_KEY="66a73856614bc24a07523f3fbee42482fcbeffe3"
export HF_TOKEN="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"

source /sjw_alinlab1/home/jungwook/miniconda3/bin/activate gr00t

CKPT_DIR="checkpoints_action_tokenizer/gr1_100demos_v2_hand_pred"
DATA_DIR=/storage1/sjw_dataset/dataset/robocasa_gr1_tabletop/sim_100demos

# V2 + Hand state prediction
# hand_state_dims: state에서 hand에 해당하는 dim (left_hand + right_hand)
# hand_pred_future_steps: 8step, 16step 후 hand state 예측
python scripts/train_action_latent_tokenizer_v2.py \
    --dataset-path $DATA_DIR \
    --output-dir $CKPT_DIR \
    --data-config fourier_gr1_arms_waist \
    --embodiment-tag new_embodiment \
    --run-name "actlat_v2_hand_pred" \
    --num-gpus 2 \
    --batch-size 1024 \
    --max-steps 100000 \
    --save-steps 5000 \
    --num-global-tokens 0 \
    --num-hand-tokens 2 \
    --lambda-recon 1.0 \
    --lambda-hand-pred 1 \
    --hand-state-dims 14 15 16 17 18 19 20 21 22 23 24 25 \
    --hand-pred-future-steps 8 16 \
    --hand-pred-decoder-depth 2 \
    --recon-loss-type mse \
    --decoder-mode "self_attention" \
    --wandb-project "action-latent-tokenizer-v2" \
    --eval-steps 1000 \
    --report-to wandb \
    --resume
