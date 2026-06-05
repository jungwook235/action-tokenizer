#!/bin/bash

export NO_ALBUMENTATIONS_UPDATE=1
export TF_CPP_MIN_LOG_LEVEL=3

# =========================================
PORT=$(( 27800 + (RANDOM % 100) ))

CKPT_NAME="vla_nactlat_fm_robocasa_100demos/baseline"
CKPT_STEP="${CKPT_STEP:-60000}"
# =========================================
BASE_DIR="/sjw_alinlab1/home/jungwook/Isaac-GR00T"
CONDA_PATH="/sjw_alinlab1/home/jungwook/miniconda3"
CKPT_PATH="$BASE_DIR/checkpoints/$CKPT_NAME/checkpoint-$CKPT_STEP"
echo "[i] Evaluating 'CKPT_NAME=$CKPT_NAME, CKPT_STEP=$CKPT_STEP'..."

# Determine embodiment tag from metadata.json (first key)
METADATA_PATH="$CKPT_PATH/experiment_cfg/metadata.json"
if [ -f "$METADATA_PATH" ]; then
    EMBODIMENT_TAG=$(python3 -c "import json, sys; data=json.load(open(sys.argv[1])); print(list(data.keys())[0])" "$METADATA_PATH")
    echo "[i] Using embodiment_tag from metadata.json: $EMBODIMENT_TAG"
else
    echo "[!] Warning: metadata.json not found at $METADATA_PATH, using default"
    EMBODIMENT_TAG="new_embodiment"
fi

# Baseline (mode=vla) has no tokenizer — pass empty string to trigger no-tokenizer branch
ACTLAT_TOKENIZER_PATH=""
ACTLAT_TARGET_TOKENS="all"

# === Policy server (conda env: gr00t) ===
"$CONDA_PATH"/envs/gr00t/bin/python $BASE_DIR/scripts/inference_service_actlat_fm.py --server \
    --port $PORT \
    --model_path $CKPT_PATH \
    --data_config single_panda_gripper_actlat_fm \
    --embodiment_tag $EMBODIMENT_TAG \
    --actlat_tokenizer_path "$ACTLAT_TOKENIZER_PATH" \
    --actlat_target_tokens "$ACTLAT_TARGET_TOKENS" &
SERVE_PID=$!

sleep 20  # Wait for policy server to start

TASK_NAMES=(
  "TurnSinkSpout"
  "TurnOnStove"
  "TurnOnSinkFaucet"
  "TurnOnMicrowave"
  "TurnOffStove"
  "TurnOffSinkFaucet"
  "TurnOffMicrowave"
  "PnPStoveToCounter"
  
) # 24 tasks total

echo "[i] Running tasks sequentially: ${TASK_NAMES[@]}"

for TASK_NAME in "${TASK_NAMES[@]}"; do
    OUTPUT_DIR="$BASE_DIR/output/robocasa_kitchen/$CKPT_NAME/$CKPT_STEP/$TASK_NAME"
    mkdir -p "$OUTPUT_DIR"
    echo "[i] >>> Starting task: $TASK_NAME"
    "$CONDA_PATH"/envs/robocasa2/bin/python $BASE_DIR/scripts_FINEACT/robocasa_service.py --client \
        --port $PORT \
        --env_name $TASK_NAME \
        --video_dir $OUTPUT_DIR \
        --max_episode_steps 750 \
        --n_episodes 50 \
        --generative_textures \
        >& "$OUTPUT_DIR/eval.log"
    echo "[i] <<< Finished task: $TASK_NAME"
done

kill "$SERVE_PID"
echo "[i] Finished evaluating 'CKPT_NAME=$CKPT_NAME, CKPT_STEP=$CKPT_STEP'"
