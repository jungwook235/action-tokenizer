#!/bin/bash
#SBATCH --job-name=full_train_dexjoco_recon_ln_bottleneck16_vae_fs
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=sjw_alinlab
#SBATCH --output=out/%j-full_train_v3_gr1_100demos_recon_ln_bottleneck16_vae_fs.out
#SBATCH --error=out/%j-full_train_v3_gr1_100demos_recon_ln_bottleneck16_vae_fs.err

set -x
export PATH="$HOME/.local/bin:$PATH"
export WANDB_PROJECT=Action-Tokenizer-GR1-1000demos
export WANDB_API_KEY="66a73856614bc24a07523f3fbee42482fcbeffe3"
export HF_TOKEN="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"

BASE_DIR=/NHNHOME/data/wook/action-tokenizer
cd $BASE_DIR

source /NHNHOME/data/wook/miniconda3/bin/activate gr00t-actlat
export PATH="$CONDA_PREFIX/bin:$PATH"
hash -r
echo "[env-check] which python=$(which python)"
echo "[env-check] CONDA_PREFIX=$CONDA_PREFIX"
python -c "import sys, transformers; print('exe=', sys.executable, 'transformers=', transformers.__version__)"

CKPT_DIR="checkpoints/vla_nactlat_fm_gr1_1000demos/base_fs"
# Glob expands to all 24 gr1_unified.* dataset dirs (each a LeRobot dataset root)
DATA_DIR=(/NHNHOME/data/wook/dataset/gr00t_unified/gr1_unified.*)

python scripts/gr00t_finetune_actlat_fm.py \
    --dataset-path "${DATA_DIR[@]}" \
    --output-dir $CKPT_DIR \
    --data-config fourier_gr1_arms_waist_actlat_fm \
    --embodiment-tag new_embodiment \
    --base-model-path "nvidia/GR00T-N1.5-3B" \
    --run-name "nactlat_baseline_gr1_1000demos_fs" \
    --mode "vla" \
    --num-gpus 8 \
    --batch-size 64 \
    --max-steps 60000 \
    --save-steps 5000 \
    --save-total-limit 3 \
    --eval-steps 1000 \
    --val-ratio 0.003 \
    --use-fixed-val \
    --no-load-action-head \
    --video-backend "decord"
