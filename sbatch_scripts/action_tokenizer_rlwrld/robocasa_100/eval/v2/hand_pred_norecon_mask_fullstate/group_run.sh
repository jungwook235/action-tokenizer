#!/usr/bin/env bash
set -euo pipefail

SBATCH_FILE="/sjw_alinlab1/home/jungwook/Isaac-GR00T/sbatch_scripts/action_tokenizer/robocasa_100/eval/v2/hand_pred_norecon_mask_fullstate/eval_robocasa_local.sh"

# Pass checkpoint steps as arguments, or use defaults
if [[ $# -ge 1 ]]; then
  STEPS=("$@")
else
  STEPS=( 60000 )
fi

echo "[i] Submitting ${#STEPS[@]} jobs in sequence using dependency=afterok"

PREV_JOBID=""

for STEP in "${STEPS[@]}"; do
  if [[ -z "${PREV_JOBID}" ]]; then
    JOBID=$(sbatch -p background --parsable --export=ALL,CKPT_STEP="${STEP}" "${SBATCH_FILE}")
  else
    JOBID=$(sbatch -p background --parsable --dependency=afterok:${PREV_JOBID} \
                   --export=ALL,CKPT_STEP="${STEP}" "${SBATCH_FILE}")
  fi

  echo "[+] Submitted CKPT_STEP=${STEP} as JobID=${JOBID}"
  PREV_JOBID="${JOBID}"
done

echo "[i] Chain submitted. Track with: squeue -j ${PREV_JOBID}"
