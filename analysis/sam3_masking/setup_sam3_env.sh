#!/bin/bash
# Bootstrap the SAM3 masking environment on a bare k8s pod (idempotent).
#
#   ./setup_sam3_env.sh                  # venv + deps + ffmpeg + official repo + weights
#   ./setup_sam3_env.sh --venv-only      # just the venv + pip deps (what the launcher calls)
#   ./setup_sam3_env.sh --skip-sam31     # skip the official repo / SAM 3.1 ckpt (~3.5 GB)
#   ./setup_sam3_env.sh --convert-sam31  # additionally convert SAM 3.1 -> transformers format
#   ./setup_sam3_env.sh --force          # rebuild the venv from scratch
#
# What lands where (everything on the persistent /data mount — a pod's root fs is
# ephemeral, so nothing may live under ~ or /usr):
#   venv_sam3/                     uv venv, python 3.11, torch cu128 + transformers 5.x
#   $HF_HOME/hub/                  jetjodh/sam3 + jetjodh/sam3.1 weights (hub cache;
#                                  the scripts pass repo ids, so they resolve from here)
#   sam3_official/                 facebookresearch/sam3 source tree (editable install)
#
# Re-running is cheap: deps are re-synced only when requirements_sam3.txt changes,
# downloads no-op when cached, the clone is skipped when present.
set -uo pipefail

SAM3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SAM3_DIR/sam3_env.sh"

VENV_ONLY=0
SKIP_SAM31=0
CONVERT_SAM31=0
FORCE=0
for a in "$@"; do
    case "$a" in
        --venv-only)     VENV_ONLY=1 ;;
        --skip-sam31)    SKIP_SAM31=1 ;;
        --convert-sam31) CONVERT_SAM31=1 ;;
        --force)         FORCE=1 ;;
        -h|--help)       sed -n 2,20p "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "[x] unknown arg: $a" >&2; exit 2 ;;
    esac
done

step() { echo; echo "=== $* ==="; }
die()  { echo "[x] $*" >&2; exit 1; }

step "target layout"
echo "SAM3_DIR      = $SAM3_DIR"
echo "venv          = $SAM3_VENV (python $SAM3_PYTHON_VERSION)"
echo "HF_HOME       = $HF_HOME"
echo "sam3 weights  = $SAM3_HF_REPO   (ungated mirror of the gated facebook/sam3)"
echo "sam3.1 ckpt   = $SAM31_HF_REPO"
echo "official repo = $SAM3_OFFICIAL_DIR"

# --- 1. uv ------------------------------------------------------------------
step "uv"
if ! command -v uv >/dev/null 2>&1; then
    echo "[i] uv not found -> pip install uv"
    pip install -q uv || die "pip install uv failed"
fi
uv --version || die "uv unusable"

# --- 2. venv ----------------------------------------------------------------
step "venv"
REQ="$SAM3_DIR/requirements_sam3.txt"
[ -f "$REQ" ] || die "missing $REQ"
STAMP="$SAM3_VENV/.sam3_requirements.sha256"
WANT="$(sha256sum "$REQ" | cut -d' ' -f1)"

if [ "$FORCE" = "1" ] && [ -d "$SAM3_VENV" ]; then
    echo "[i] --force: removing $SAM3_VENV"
    rm -rf "$SAM3_VENV"
fi
# The venv that came over from the old cluster is a stub: pyvenv.cfg points at a
# conda prefix that does not exist here and bin/python is a dangling symlink.
# Detect that (rather than trusting the directory's existence) and rebuild.
if [ -d "$SAM3_VENV" ] && ! "$SAM3_PY" -c "import sys" >/dev/null 2>&1; then
    echo "[!] $SAM3_VENV exists but its interpreter is broken (stale/foreign venv) -> recreating"
    rm -rf "$SAM3_VENV"
fi
if [ ! -x "$SAM3_PY" ]; then
    uv venv --python "$SAM3_PYTHON_VERSION" "$SAM3_VENV" || die "uv venv failed"
fi

if [ "$(cat "$STAMP" 2>/dev/null)" = "$WANT" ]; then
    echo "[i] deps already in sync with requirements_sam3.txt -> skip install"
else
    # unsafe-best-match: the cu128 index shadows a few PyPI packages with older
    # versions (iopath 0.1.9), which blocks resolution under the default strategy.
    uv pip install --python "$SAM3_PY" --index-strategy unsafe-best-match -r "$REQ" \
        || die "dependency install failed"
    echo "$WANT" > "$STAMP"
fi
"$SAM3_PY" - <<'PY' || die "core import check failed"
import sys, numpy, torch, transformers, cv2
print(f"python       {sys.version.split()[0]}")
print(f"numpy        {numpy.__version__}")
print(f"torch        {torch.__version__}  (cuda {torch.version.cuda})")
print(f"transformers {transformers.__version__}")
print(f"cv2          {cv2.__version__}")
from transformers import Sam3Model, Sam3Processor, Sam3VideoModel, Sam3VideoProcessor  # noqa
print("transformers SAM3 classes: OK")
print(f"cuda visible : {torch.cuda.is_available()} ({torch.cuda.device_count()} device(s))")
PY

# --- 3. ffmpeg --------------------------------------------------------------
# run_sam3.write_video() shells out to ffmpeg to re-encode mp4v -> h264. Without
# it the videos are still written, just as mp4v — i.e. a pod missing ffmpeg would
# silently produce differently-encoded data, so this runs in --venv-only mode too
# (the per-job launchers only call --venv-only). Same apt bootstrap the training
# scripts do; non-fatal so a locked-down pod does not abort the whole setup.
step "ffmpeg"
if command -v ffmpeg >/dev/null 2>&1; then
    echo "[i] present: $(ffmpeg -version 2>/dev/null | head -1)"
else
    echo "[i] installing ffmpeg via apt-get"
    (apt-get update -q && apt-get install -y -q ffmpeg) \
        || echo "[!] apt-get ffmpeg failed — videos will stay mp4v-encoded (not fatal)"
fi

if [ "$VENV_ONLY" = "1" ]; then
    step "done (--venv-only)"
    exit 0
fi

# --- 4. SAM3 weights into the HF hub cache ----------------------------------
# The scripts pass repo ids (--model jetjodh/sam3), so pre-populating $HF_HOME is
# what makes the GPU job start without hitting the network.
step "SAM3 weights -> $HF_HOME/hub"
# sam3.pt (the official-repo-format copy of the same 3.0 weights, 3.4 GB) is
# excluded: the transformers route reads model.safetensors, and the official route
# below uses the 3.1 multiplex checkpoint instead.
"$SAM3_VENV/bin/hf" download "$SAM3_HF_REPO" --exclude "sam3.pt" \
    || die "download $SAM3_HF_REPO failed"

# --- 5. official SAM3 repo + SAM 3.1 ----------------------------------------
if [ "$SKIP_SAM31" = "1" ]; then
    echo; echo "[i] --skip-sam31: skipping official repo and the 3.1 checkpoint"
else
    step "facebookresearch/sam3 source tree"
    if [ -d "$SAM3_OFFICIAL_DIR/.git" ]; then
        echo "[i] present: $SAM3_OFFICIAL_DIR ($(git -C "$SAM3_OFFICIAL_DIR" rev-parse --short HEAD))"
    else
        # The dir that came over from the old cluster is empty, and git clone
        # refuses a non-empty target -> remove it if (and only if) it is empty.
        rmdir "$SAM3_OFFICIAL_DIR" 2>/dev/null
        [ -e "$SAM3_OFFICIAL_DIR" ] && die "$SAM3_OFFICIAL_DIR exists, is not empty, and is not a clone"
        git clone --depth 1 https://github.com/facebookresearch/sam3 "$SAM3_OFFICIAL_DIR" \
            || die "git clone facebookresearch/sam3 failed"
    fi
    # --no-deps: its pyproject would otherwise renegotiate numpy/timm and can pull
    # a non-cu128 torch. The runtime deps are already in requirements_sam3.txt.
    if ! "$SAM3_PY" -c "import sam3" >/dev/null 2>&1; then
        uv pip install --python "$SAM3_PY" --no-deps -e "$SAM3_OFFICIAL_DIR" \
            || die "editable install of sam3_official failed"
    fi
    "$SAM3_PY" -c "
from sam3.model_builder import build_sam3_multiplex_video_predictor
print('official sam3 import: OK')" || die "official sam3 import failed"

    step "SAM 3.1 multiplex checkpoint -> $HF_HOME/hub"
    "$SAM3_VENV/bin/hf" download "$SAM31_HF_REPO" || die "download $SAM31_HF_REPO failed"
    # `hf download <repo> <file>` echoes "path=<abs path>" -> strip the prefix.
    SAM31_CKPT="$("$SAM3_VENV/bin/hf" download "$SAM31_HF_REPO" sam3.1_multiplex.pt)" \
        || die "resolving sam3.1_multiplex.pt failed"
    SAM31_CKPT="${SAM31_CKPT#path=}"
    [ -f "$SAM31_CKPT" ] || die "sam3.1_multiplex.pt not where hf reported: $SAM31_CKPT"
    echo "[i] sam3.1_multiplex.pt = $SAM31_CKPT"
    echo "[i] bench_sam31.py has this path hardcoded in its CKPT constant (old-cluster"
    echo "    path) — point it here before running the official-route benchmark."

    if [ "$CONVERT_SAM31" = "1" ]; then
        step "SAM 3.1 -> transformers format ($SAM3_DIR/sam3.1-hf)"
        "$SAM3_PY" "$SAM3_DIR/convert_sam3_video_to_hf.py" \
            --checkpoint_path "$SAM31_CKPT" --output_path "$SAM3_DIR/sam3.1-hf" \
            || die "conversion failed"
    fi
fi

step "setup complete"
cat <<EOF
venv python : $SAM3_PY
smoke test  : ./run_sam3_batch_k8s.sh --limit 1      # 1 episode per GPU shard
full run    : ./run_sam3_batch_k8s.sh                # all shards, all GPUs
EOF
