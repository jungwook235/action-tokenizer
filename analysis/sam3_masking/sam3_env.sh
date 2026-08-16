# Shared env for the SAM3 masking pipeline (source this, do not execute).
#
#   source /data/jungwook/action-tokenizer/analysis/sam3_masking/sam3_env.sh
#
# Every value is overridable from the outside (VAR=... before sourcing).

SAM3_DIR="${SAM3_DIR:-/data/jungwook/action-tokenizer/analysis/sam3_masking}"

# --- venv -------------------------------------------------------------------
# uv-managed venv living on the shared /data mount, so it survives pod restarts
# (k8s pods have an ephemeral root fs; only /data persists).
export SAM3_VENV="${SAM3_VENV:-$SAM3_DIR/venv_sam3}"
export SAM3_PY="${SAM3_PY:-$SAM3_VENV/bin/python}"
export SAM3_PYTHON_VERSION="${SAM3_PYTHON_VERSION:-3.11}"

# --- HF cache ---------------------------------------------------------------
# HOME differs between pods, so pin the cache to an absolute path on /data.
# The scripts pass hub repo ids (jetjodh/sam3), so the weights must be in this
# cache — setup_sam3_env.sh pre-downloads them here.
export HF_HOME="${HF_HOME:-/data/jungwook/.cache/huggingface}"

# --- checkpoints ------------------------------------------------------------
# transformers-format SAM3, used by run_sam3.py / batch_sam3_robot_task.py
# (--model default). facebook/sam3 is gated (manual approval); jetjodh/sam3 is an
# ungated mirror of the same weights, which is why the code defaults to it.
export SAM3_HF_REPO="${SAM3_HF_REPO:-jetjodh/sam3}"

# Official-repo SAM 3.1 multiplex checkpoint, used by bench_sam31.py.
export SAM31_HF_REPO="${SAM31_HF_REPO:-jetjodh/sam3.1}"

# Official facebookresearch/sam3 source tree (bench_sam31.py imports `sam3`).
export SAM3_OFFICIAL_DIR="${SAM3_OFFICIAL_DIR:-$SAM3_DIR/sam3_official}"

# --- data -------------------------------------------------------------------
export SAM3_DATA_GLOB="${SAM3_DATA_GLOB:-/data/shared_dataset/GR00T-X-Embodiment-Sim/gr1_unified.*}"
# NOT derived from MODEL_OUTPUT_DIR on purpose: the k8s yamls give each of the 4
# jobs its own MODEL_OUTPUT_DIR (…/jungwook-data-sam3-gr1-N), which would scatter
# one dataset across 4 roots. All 4 jobs must write into this single shared root.
export SAM3_OUT_ROOT="${SAM3_OUT_ROOT:-/data/rlwrld-unified-checkpoints/jungwook/action_tokenizer/data/GR00T-X-Embodiment-Sim_sam3_robot_task}"

# --- runtime knobs ----------------------------------------------------------
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
