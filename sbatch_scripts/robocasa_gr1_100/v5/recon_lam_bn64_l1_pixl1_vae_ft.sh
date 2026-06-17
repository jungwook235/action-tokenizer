#!/bin/bash
#SBATCH --job-name=full_v5_gr1_recon_lam_bn64_l1_pixl1_vae_ft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b200:4
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --output=/NHNHOME/data/wook/action-tokenizer/slurm/logs/full_v5_gr1_recon_lam_bn64_l1_pixl1_vae_ft_%j.out
#SBATCH --error=/NHNHOME/data/wook/action-tokenizer/slurm/logs/full_v5_gr1_recon_lam_bn64_l1_pixl1_vae_ft_%j.err


set -ex  # -e: abort the whole job (skip Stage-2) if Stage-1 fails
export PATH="$HOME/.local/bin:$PATH"
export WANDB_PROJECT=Action-Tokenizer-GR1-100demos
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

# robocasa_gr1_tabletop is a GR1 embodiment (same action keys + single video.ego_view),
# so the same data-config (fourier_gr1_arms_waist) is reused; only the dataset path
# and per-dataset normalization stats differ.
DATA_DIR=/NHNHOME/data/wook/dataset/robocasa_gr1_tabletop/sim_100demos
TOK_CKPT_DIR=checkpoints_action_tokenizer/gr1_100demos_v5_recon_lam_bn64_l1_pixl1_vae
VLA_CKPT_DIR=checkpoints/vla_actlat_fm_gr1_100demos/v5_recon_lam_bn64_l1_pixl1_vae_fs
VLA_CKPT_DIR_FT=checkpoints/vla_actlat_fm_gr1_100demos/v5_recon_lam_bn64_l1_pixl1_vae

TOK_STEP=100000
ABS_TOK_CKPT="/NHNHOME/data/wook/action-tokenizer/$TOK_CKPT_DIR"

# Pretrained DreamDojo Latent Action Model checkpoint. Used (a) frozen, to extract
# the z_rep latent-action token that replaces DINO/VGGT as the fusion visual input,
# and (b) to initialize the trainable pixel decoder (patch_up + SpatioTransformer).
# If this local path is absent it is auto-downloaded from the nvidia/DreamDojo HF
# repo (LAM_400k.ckpt) and cached under HF_HOME — so DO NOT force HF offline here.
LAM_CKPT=DreamDojo/checkpoints/DreamDojo/LAM_400k.ckpt



python scripts/gr00t_finetune_actlat_fm.py \
    --dataset-path "$DATA_DIR" \
    --output-dir $VLA_CKPT_DIR_FT \
    --data-config fourier_gr1_arms_waist_actlat_fm \
    --embodiment-tag new_embodiment \
    --base-model-path "nvidia/GR00T-N1.5-3B" \
    --run-name "actlat_fm_v5_gr1_recon_lam_bn64_l1_pixl1_vae_100demos" \
    --num-gpus 4 \
    --batch-size 16 \
    --max-steps 60000 \
    --save-steps 5000 \
    --save-total-limit 3 \
    --eval-steps 1000 \
    --actlat-tokenizer-path "$ABS_TOK_CKPT/checkpoint-$TOK_STEP" \
    --actlat-target-tokens "all" \
    --actlat-frames \
    --val-ratio 0.003 \
    --use-fixed-val \
    --video-backend "decord"
