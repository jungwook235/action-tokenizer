#!/bin/bash
# Download the pretrained DreamDojo Latent Action Model checkpoint (LAM_400k.ckpt,
# ~8.5 GB) to the local path the V5 tokenizer expects, so training never hits HF at
# run time (the V5 resolver uses the local file when present).
#
# IMPORTANT: we explicitly UNSET HF_TOKEN. The sbatch script exports an invalid
# HF_TOKEN ("hf_yKdv..."), which takes precedence over the (valid) cached login and
# causes a 401. Unsetting it falls back to the working `hf auth login` token.
#
# Usage:
#   bash scripts/download_lam_ckpt.sh
# Re-running is safe: hf_hub_download resumes / skips an already-complete file.

set -euo pipefail

BASE_DIR=/NHNHOME/data/wook/action-tokenizer
cd "$BASE_DIR"

# Activate the project env (so huggingface_hub is available); harmless if already active.
source /NHNHOME/data/wook/miniconda3/bin/activate gr00t-actlat 2>/dev/null || true

# Use the cached `hf auth login` token, NOT the invalid env var.
unset HF_TOKEN

REPO_ID="nvidia/DreamDojo"
FILENAME="LAM_400k.ckpt"
DEST_DIR="DreamDojo/checkpoints/DreamDojo"
DEST_PATH="$DEST_DIR/$FILENAME"

mkdir -p "$DEST_DIR"

echo "[download_lam] whoami: $(python -c 'from huggingface_hub import HfApi; print(HfApi().whoami().get("name"))')"
echo "[download_lam] downloading $REPO_ID/$FILENAME -> $DEST_PATH (~8.5 GB) ..."

python - <<'PY'
from huggingface_hub import hf_hub_download

dest_dir = "DreamDojo/checkpoints/DreamDojo"
path = hf_hub_download(
    repo_id="nvidia/DreamDojo",
    filename="LAM_400k.ckpt",
    local_dir=dest_dir,
)
print(f"[download_lam] DONE -> {path}")
PY

echo "[download_lam] final file:"
ls -lh "$DEST_PATH"
echo "[download_lam] OK. The V5 trainer will now use this local ckpt (no HF at run time)."
