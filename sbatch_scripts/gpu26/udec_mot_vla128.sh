#!/bin/bash
#SBATCH --job-name=gr1_vla128_udec_mot
#SBATCH --partition=h100
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
# flock-guarded tar->/scratch extraction, GR00T_S3_COMPAT checkpointing,
# idempotent stages (background QoS restarts this script from the top; each
# stage skips itself when its final checkpoint exists and otherwise --resume's
# from the latest one on /s3ckpt).
# save-steps: ref values kept (user decision 2026-08-17 — accepts >30min save
# interval to limit undeletable checkpoint accumulation on /s3ckpt).

export WANDB_API_KEY="66a73856614bc24a07523f3fbee42482fcbeffe3"
export PYTHONUNBUFFERED=1
set -euxo pipefail

ARCH=mot
DECODER_FLAGS=(--decoder-arch mot --mot-depth 6)

# --- storage contract (submit filter checks these literals) ---
S3_DATA_ROOT=/s3data/gr1-unified-lerobot/v1
TOK_CKPT_DIR=/s3ckpt/$USER/tok_gr1_v4_udec_${ARCH}_bs256
VLA_CKPT_DIR=/s3ckpt/$USER/vla_gr1_v4_udec_${ARCH}_vla128
SCRATCH_DATA=/scratch/$USER/gr1_unified
TOK_STEP=100000
VLA_STEP=60000

# gpu26 S3-compat switch: scratch-staged checkpoints (data-only copy to
# /s3ckpt) + WANDB_DIR respected. Other servers never set this -> stock code.
export GR00T_S3_COMPAT=1
export GR00T_CKPT_STAGE_DIR=/scratch/$USER/ckpt_stage_udec_${ARCH}
export WANDB_DIR=$HOME/wandb_runs
mkdir -p "$WANDB_DIR"

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

export WANDB_PROJECT="Action-Tokenizer-GR1-1000demos"

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
    --run-name "actlat_v4_gr1_udec_${ARCH}_1000demos_bs256_vla128" \
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
    "${DECODER_FLAGS[@]}"
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
    --run-name "actlat_fm_v4_gr1_udec_${ARCH}_1000demos_bs256_vla128" \
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

echo "[done] udec_${ARCH} pipeline complete"
