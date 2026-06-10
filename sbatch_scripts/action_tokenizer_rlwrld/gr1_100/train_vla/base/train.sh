#!/bin/bash
#SBATCH --job-name=actlat_fm_gr1_100demos_base_sbatch
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=sjw_alinlab
#SBATCH --output=out/%j-actlat_fm_gr1_100demos_base_sbatch.out
#SBATCH --error=out/%j-actlat_fm_gr1_100demos_base_sbatch.err
#SBATCH --comment "actlat_fm_gr1_100demos_base_sbatch"

set -x
export PATH="$HOME/.local/bin:$PATH"
export WANDB_API_KEY="66a73856614bc24a07523f3fbee42482fcbeffe3"
export HF_TOKEN="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"

source /sjw_alinlab1/home/jungwook/miniconda3/bin/activate gr00t

CKPT_DIR="checkpoints/vla_actlat_fm_gr1_100demos/base"
DATA_DIR=/storage1/sjw_dataset/dataset/robocasa_gr1_tabletop/sim_100demos
TOKENIZER_PATH=/sjw_alinlab1/home/jungwook/Isaac-GR00T/checkpoints_action_tokenizer/gr1_100demos_base/checkpoint-100000

python scripts/gr00t_finetune_actlat_fm.py \
    --dataset-path $DATA_DIR \
    --output-dir $CKPT_DIR \
    --data-config fourier_gr1_arms_waist_actlat_fm \
    --embodiment-tag new_embodiment \
    --base-model-path "nvidia/GR00T-N1.5-3B" \
    --run-name "actlat_fm_base_gr1_100demos" \
    --num-gpus 2 \
    --batch-size 32 \
    --max-steps 60000 \
    --save-steps 10000 \
    --eval-steps 1000 \
    --actlat-tokenizer-path $TOKENIZER_PATH \
    --actlat-target-tokens "all" \
    --val-ratio 0.003 \
    --video-backend "decord"