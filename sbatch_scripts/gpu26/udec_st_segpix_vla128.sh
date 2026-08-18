#!/bin/bash
#SBATCH --job-name=gr1_vla128_udec_st_segpix
#SBATCH --partition=h200
#SBATCH --qos=background
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120
#SBATCH --output=/home/wook/logs_gpu26/%x_%j.out
#SBATCH --error=/home/wook/logs_gpu26/%x_%j.err

# gpu26 port of the mlxp base recipe (~/ref_mlxp_base_vla128.sh =
# recon_dino_bn64_l1_mse_naiveln_vae_4gpus_bs256_vla128.sh). ONLY the decoder-
# architecture flags differ from the ref; every other Stage-1/Stage-2
# hyperparameter is kept verbatim. gpu26 adaptations: storage-contract paths,
# flock-guarded tar->/scratch extraction, idempotent stages (a requeue restarts
# this script from the top; each stage skips itself when its final checkpoint
# exists and otherwise --resume's from the latest one under /ckpt).
# h200 run (2026-08-18): normal QoS (assigned partition, no preemption);
# checkpoints on /ckpt per the 2026-08-18 policy. save-steps: ref values
# (user decision 2026-08-17).

export WANDB_API_KEY="66a73856614bc24a07523f3fbee42482fcbeffe3"
export PYTHONUNBUFFERED=1
export WANDB_DISABLE_STATS=true
set -euxo pipefail

ARCH=shared_trunk
DECODER_FLAGS=(--decoder-arch shared_trunk --decoder-trunk-depth 4 --decoder-branch-depth 2)
# segpix third branch (flags mirror mlxp udec-st-segpix); mask root is set
# after the masks_pf extraction below
SEGPIX_STATIC_FLAGS=(--use-seg-pixel-decoder --lambda-seg-pixel 0.1 --seg-pixel-patch 14 --mask-subdir masks_pf)

# --- storage contract (submit filter checks these literals) ---
# 2026-08-18 policy: checkpoints go to /ckpt/$USER (real POSIX FS, auto-archived
# to S3 in ~20s) — the GR00T_S3_COMPAT staging workaround is no longer needed.
S3_DATA_ROOT=/s3data/gr1-unified-lerobot/v1
TOK_CKPT_DIR=/ckpt/$USER/tok_gr1_v4_udec_st_segpix_bs256
VLA_CKPT_DIR=/ckpt/$USER/vla_gr1_v4_udec_st_segpix_vla128
SCRATCH_DATA=/scratch/$USER/gr1_unified
TOK_STEP=100000
VLA_STEP=60000

BASE_DIR=/home/wook/action-tokenizer
cd "$BASE_DIR"
source /home/wook/miniconda3/bin/activate gr00t-actlat
echo "[env-check] $(which python)"
python -c "import torch; print('cuda=', torch.cuda.is_available(), 'ngpu=', torch.cuda.device_count())"

# --- Stage 0: idempotent tar extraction (flock: both udec jobs may share the node) ---
mkdir -p "$SCRATCH_DATA"
(
  flock -x 200
  for tarpath in "$S3_DATA_ROOT"/*.tar; do
    base=$(basename "$tarpath" .tar)
    marker="$SCRATCH_DATA/.done.$base"
    [ -f "$marker" ] && continue
    [ -f "$tarpath.ok" ] || { echo "[extract] missing .ok for $base"; exit 1; }
    tar -xf "$tarpath" -C "$SCRATCH_DATA"
    touch "$marker"
    echo "[extract] done: $base"
  done
) 200>"$SCRATCH_DATA/.extract.lock"

DATA_DIR=("$SCRATCH_DATA"/gr1_unified.*/)
echo "[data-check] ${#DATA_DIR[@]} dataset roots (expect 24)"
[ "${#DATA_DIR[@]}" -eq 24 ]

# --- Stage 0b: masks_pf tar extraction (segpix targets; cutout not needed) ---
S3_MASK_ROOT=/s3data/gr1-sam3-norobot/v1
SCRATCH_MASKS=/scratch/$USER/gr1_sam3
mkdir -p "$SCRATCH_MASKS"
(
  flock -x 201
  for tarpath in "$S3_MASK_ROOT"/*.masks_pf.tar; do
    base=$(basename "$tarpath" .tar)
    marker="$SCRATCH_MASKS/.done.$base"
    [ -f "$marker" ] && continue
    [ -f "$tarpath.ok" ] || { echo "[extract-mask] missing .ok for $base"; exit 1; }
    tar -xf "$tarpath" -C "$SCRATCH_MASKS"
    touch "$marker"
    echo "[extract-mask] done: $base"
  done
) 201>"$SCRATCH_MASKS/.extract.lock"

MASK_DIRS=("$SCRATCH_MASKS"/gr1_unified.*/)
echo "[data-check] ${#MASK_DIRS[@]} mask roots (expect 24)"
[ "${#MASK_DIRS[@]}" -eq 24 ]
SEGPIX_FLAGS=("${SEGPIX_STATIC_FLAGS[@]}" --mask-dataset-root "$SCRATCH_MASKS")

export WANDB_PROJECT="Action-Tokenizer-GR1-1000demos"

# --- Stage 0c: seed the Stage-1 resume checkpoint into /ckpt (h200 AZ) ---
# The 50k checkpoint was produced elsewhere (RLDX relay) and /ckpt is per-AZ: the
# head node mounts a different filesystem (ale7jb4v) than the h200 nodes
# (ape7jb4v), so the relay lands in home and THIS job — which runs on h200 —
# copies it into the real Stage-1 output dir once. Skipped when the output dir
# already holds checkpoints (requeue after further progress). Fails loudly on a
# partial copy so training never silently restarts from scratch.
RESUME_SEED=/home/wook/incoming_rldx/st_segpix/checkpoint-50000
mkdir -p "$TOK_CKPT_DIR"
if compgen -G "$TOK_CKPT_DIR/checkpoint-*" > /dev/null; then
  echo "[seed] $TOK_CKPT_DIR already has checkpoints — skipping seed"
elif [ -d "$RESUME_SEED" ]; then
  n_src=$(find "$RESUME_SEED" -type f | wc -l)
  b_src=$(du -sb "$RESUME_SEED" | cut -f1)
  cp -a "$RESUME_SEED" "$TOK_CKPT_DIR/"
  n_dst=$(find "$TOK_CKPT_DIR/checkpoint-50000" -type f | wc -l)
  b_dst=$(du -sb "$TOK_CKPT_DIR/checkpoint-50000" | cut -f1)
  echo "[seed] $RESUME_SEED -> $TOK_CKPT_DIR/checkpoint-50000 files $n_src/$n_dst bytes $b_src/$b_dst"
  [ "$n_src" -eq "$n_dst" ] && [ "$b_src" -eq "$b_dst" ]
else
  echo "[seed] ERROR: no seed checkpoint at $RESUME_SEED (this run must resume from 50k)"
  exit 1
fi

# === Stage 1: V4 tokenizer — ref recipe + unified decoder flags ===
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
if compgen -G "$TOK_CKPT_DIR/checkpoint-$TOK_STEP/model*" > /dev/null; then
  echo "[stage1] checkpoint-$TOK_STEP exists — skipping Stage 1"
else
  python scripts/train_action_latent_tokenizer_v4.py \
    --dataset-path "${DATA_DIR[@]}" \
    --output-dir "$TOK_CKPT_DIR" \
    --data-config fourier_gr1_arms_waist \
    --embodiment-tag new_embodiment \
    --run-name "actlat_v4_gr1_udec_st_segpix_1000demos_bs256_vla128" \
    --num-gpus 4 \
    --batch-size 64 \
    --max-steps $TOK_STEP \
    --save-steps 10000 \
    --save-total-limit 3 \
    --dataloader-num-workers 6 \
    --token-dim 64 \
    --encoder-depth 4 \
    --decoder-depth 4 \
    --dino-model "facebook/dinov2-large" \
    --dino-channels 1024 \
    --dino-final-norm naive \
    --fusion-width 1024 \
    --fusion-depth 6 \
    --dino-decoder-depth 6 \
    --lambda-recon 1.0 \
    --lambda-dino 0.1 \
    --use-vae \
    --lambda-kl 1e-6 \
    --recon-loss-type l1 \
    --dino-loss-type mse \
    --dino-w-l1 0.0 \
    --dino-w-mse 1.0 \
    --decoder-mode self_attention \
    --video-backend decord \
    --use-fixed-val \
    --wandb-project "Action-Tokenizer-GR1-1000demos-tokenizer" \
    --eval-steps 1000 \
    --report-to wandb \
    --resume \
    "${DECODER_FLAGS[@]}" \
    "${SEGPIX_FLAGS[@]}"
fi

# Re-enable HF network for Stage-2 (GR00T base is pre-cached in ~/.cache, but
# eagle/aux downloads may still be needed).
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

# === Stage 2: VLA — ref recipe unchanged (wrapper auto-rebuilds the unified decoder) ===
if compgen -G "$VLA_CKPT_DIR/checkpoint-$VLA_STEP/model*" > /dev/null; then
  echo "[stage2] checkpoint-$VLA_STEP exists — nothing to do"
else
  python scripts/gr00t_finetune_actlat_fm.py \
    --dataset-path "${DATA_DIR[@]}" \
    --output-dir "$VLA_CKPT_DIR" \
    --data-config fourier_gr1_arms_waist_actlat_fm \
    --embodiment-tag new_embodiment \
    --base-model-path "nvidia/GR00T-N1.5-3B" \
    --run-name "actlat_fm_v4_gr1_udec_st_segpix_1000demos_bs256_vla128" \
    --num-gpus 4 \
    --batch-size 32 \
    --max-steps $VLA_STEP \
    --save-steps 5000 \
    --save-total-limit 20 \
    --eval-steps 1000 \
    --actlat-tokenizer-path "$TOK_CKPT_DIR/checkpoint-$TOK_STEP" \
    --actlat-target-tokens "all" \
    --actlat-frames \
    --dataloader_num_workers 6 \
    --val-ratio 0.003 \
    --use-fixed-val \
    --actlat-vae-no-sample \
    --resume \
    --video-backend "decord"
fi

echo "[done] udec_st_segpix pipeline complete"
