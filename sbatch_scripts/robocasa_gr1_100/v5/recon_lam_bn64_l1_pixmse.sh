#!/bin/bash
#SBATCH --job-name=full_v5_gr1_recon_lam_bn64_l1_pixl1_fs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b200:4
#SBATCH --nodes=1
#SBATCH --qos=preempt
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120
#SBATCH --time=72:00:00
#SBATCH --output=/NHNHOME/data/wook/action-tokenizer/slurm/logs/full_v5_gr1_recon_lam_bn64_l1_pixl1_fs_%j.out
#SBATCH --error=/NHNHOME/data/wook/action-tokenizer/slurm/logs/full_v5_gr1_recon_lam_bn64_l1_pixl1_fs_%j.err


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
TOK_CKPT_DIR=checkpoints_action_tokenizer/gr1_100demos_v5_recon_lam_bn64_l1_pixl1
VLA_CKPT_DIR=checkpoints/vla_actlat_fm_gr1_100demos/v5_recon_lam_bn64_l1_pixl1_fs
TOK_STEP=100000
ABS_TOK_CKPT="/NHNHOME/data/wook/action-tokenizer/$TOK_CKPT_DIR"

# Pretrained DreamDojo Latent Action Model checkpoint. Used (a) frozen, to extract
# the z_rep latent-action token that replaces DINO/VGGT as the fusion visual input,
# and (b) to initialize the trainable pixel decoder (patch_up + SpatioTransformer).
# If this local path is absent it is auto-downloaded from the nvidia/DreamDojo HF
# repo (LAM_400k.ckpt) and cached under HF_HOME — so DO NOT force HF offline here.
LAM_CKPT=DreamDojo/checkpoints/DreamDojo/LAM_400k.ckpt

# === Stage 1: V5 Tokenizer Training — RLA-LAM hybrid, pixel reconstruction ===
# Same RLA fusion + action recon as V4, but the visual element is swapped to LAM:
#   * fusion visual context = frozen-LAM z_rep [B,1,32]  (not DINO patch diff)
#   * second decoder = LAM SpatioTransformer reconstructing FUTURE-FRAME PIXELS
#     (not future DINO features); the T per-timestep latents are merged into one by
#     a learnable softmax weighted sum (LAM's z_rep slot, via action_up).
# token-dim 64 (bn64), action recon L1, pixel recon MSE (l1_pixmse).
#
# NOTE: unlike the V4 DINO script we do NOT set HF_HUB_OFFLINE — Stage-1 may need to
# pull LAM_400k.ckpt from Hugging Face on first run (HF_TOKEN above gates access).
python scripts/train_action_latent_tokenizer_v5.py \
    --dataset-path "$DATA_DIR" \
    --output-dir $TOK_CKPT_DIR \
    --no-resume \
    --data-config fourier_gr1_arms_waist \
    --embodiment-tag new_embodiment \
    --run-name "actlat_v5_gr1_recon_lam_bn64_l1_pixl1_100demos" \
    --num-gpus 4 \
    --batch-size 64 \
    --max-steps $TOK_STEP \
    --save-steps 10000 \
    --save-total-limit 3 \
    --dataloader-num-workers 16 \
    --token-dim 64 \
    --encoder-depth 4 \
    --decoder-depth 4 \
    --fusion-width 1024 \
    --fusion-depth 6 \
    --lam-ckpt "$LAM_CKPT" \
    --lam-latent-dim 32 \
    --lambda-recon 1.0 \
    --lambda-pixel 1.0 \
    --recon-loss-type l1 \
    --pixel-loss-type l1 \
    --decoder-mode self_attention \
    --video-backend decord \
    --use-fixed-val \
    --wandb-project "Action-Tokenizer-GR1-100demos-tokenizer" \
    --eval-steps 100 \
    --report-to wandb

# === Stage 2: VLA Training ===
# Unchanged vs V4: the wrapper auto-detects the V5 tokenizer (_is_v5 marker) from the
# checkpoint and, given --actlat-frames, runs the frozen LAM extractor on the frames
# to produce z_rep for encode (LAM ckpt auto-resolved from the _lam_ckpt marker / HF).
python scripts/gr00t_finetune_actlat_fm.py \
    --dataset-path "$DATA_DIR" \
    --output-dir $VLA_CKPT_DIR \
    --data-config fourier_gr1_arms_waist_actlat_fm \
    --embodiment-tag new_embodiment \
    --base-model-path "nvidia/GR00T-N1.5-3B" \
    --run-name "actlat_fm_v5_gr1_recon_lam_bn64_l1_pixl1_fs_100demos" \
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
    --no-load-action-head \
    --video-backend "decord"
