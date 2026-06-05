#!/bin/bash
#SBATCH --job-name="Eval_actlat_fm_robocasa_100demos_v2_state_pred_full_time_mask"
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --comment="Eval_actlat_fm_robocasa_100demos_v2_state_pred_full_time_mask"
#SBATCH --partition=background
#SBATCH --array=0-7
#SBATCH --output=out/%j_%x_%A_%a.out
#SBATCH --error=out/%j_%x_%A_%a.err

export NO_ALBUMENTATIONS_UPDATE=1
export TF_CPP_MIN_LOG_LEVEL=3

# =========================================
SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
PORT=$(( 27000 + (SLURM_ARRAY_TASK_ID * 100) + (RANDOM % 100) ))

CKPT_NAME="vla_actlat_fm_robocasa_100demos/v2_state_pred_full_time_mask"
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

# Read actlat tokenizer config from backed-up metadata (written by training)
CONFIG_PATH="$CKPT_PATH/experiment_cfg/metadata_config.json"
ACTLAT_TOKENIZER_PATH=""
ACTLAT_TARGET_TOKENS="all"
if [ -f "$CONFIG_PATH" ]; then
    ACTLAT_TOKENIZER_PATH=$(python3 -c "import json, sys; data=json.load(open(sys.argv[1])); print(data.get('actlat_tokenizer_path',''))" "$CONFIG_PATH")
    ACTLAT_TARGET_TOKENS=$(python3 -c "import json, sys; data=json.load(open(sys.argv[1])); print(data.get('actlat_target_tokens','all'))" "$CONFIG_PATH")
    echo "[i] Tokenizer: $ACTLAT_TOKENIZER_PATH, target: $ACTLAT_TARGET_TOKENS"
fi

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
  "PnPSinkToCounter"
  "PnPMicrowaveToCounter"
  "PnPCounterToStove"
  "PnPCounterToSink"
  "PnPCounterToMicrowave"
  "PnPCounterToCab"
  "PnPCabToCounter"
  "OpenSingleDoor"
  "OpenDrawer"
  "OpenDoubleDoor"
  "CoffeeSetupMug"
  "CoffeeServeMug"
  "CoffeePressButton"
  "CloseSingleDoor"
  "CloseDrawer"
  "CloseDoubleDoor"
) # 24 tasks total

# Each array index runs 3 tasks sequentially (i, i+8, i+16)
SELECTED_TASKS=()
if [ $SLURM_ARRAY_TASK_ID -lt 8 ]; then
    SELECTED_TASKS+=("${TASK_NAMES[$SLURM_ARRAY_TASK_ID]}")
fi
if [ $((SLURM_ARRAY_TASK_ID + 8)) -lt ${#TASK_NAMES[@]} ]; then
    SELECTED_TASKS+=("${TASK_NAMES[$((SLURM_ARRAY_TASK_ID + 8))]}")
fi
if [ $((SLURM_ARRAY_TASK_ID + 16)) -lt ${#TASK_NAMES[@]} ]; then
    SELECTED_TASKS+=("${TASK_NAMES[$((SLURM_ARRAY_TASK_ID + 16))]}")
fi
TASK_NAMES=("${SELECTED_TASKS[@]}")

echo "[i] Running tasks: ${TASK_NAMES[@]}"

MAIN_PIDS=()
for TASK_NAME in "${TASK_NAMES[@]}"; do
    OUTPUT_DIR="$BASE_DIR/output/robocasa_kitchen/$CKPT_NAME/$CKPT_STEP/$TASK_NAME"
    mkdir -p "$OUTPUT_DIR"
    "$CONDA_PATH"/envs/robocasa2/bin/python $BASE_DIR/scripts_FINEACT/robocasa_service.py --client \
        --port $PORT \
        --env_name $TASK_NAME \
        --video_dir $OUTPUT_DIR \
        --max_episode_steps 750 \
        --n_episodes 50 \
        --generative_textures \
        >& "$OUTPUT_DIR/eval-$SLURM_ARRAY_TASK_ID.log" &
    MAIN_PIDS+=($!)
done

for pid in "${MAIN_PIDS[@]}"; do
    wait "$pid"
done

kill "$SERVE_PID"
echo "[i] Finished evaluating 'CKPT_NAME=$CKPT_NAME, CKPT_STEP=$CKPT_STEP' on array $SLURM_ARRAY_TASK_ID"
