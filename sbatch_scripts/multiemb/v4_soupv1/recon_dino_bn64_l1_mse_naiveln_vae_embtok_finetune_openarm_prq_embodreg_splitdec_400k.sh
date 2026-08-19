#!/bin/bash
#SBATCH --job-name=ft_v4_soupv1_openarm_prq_embodreg_vicreg_splitdec_recon_dino_bn64_l1_mse_naiveln_vae_embtok_400k
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --partition=sjw_alinlab,h100
#SBATCH --wckey=project-short-name:sub_human
#SBATCH --output=out-multiemb/%j-ft_v4_soupv1_openarm_prq_embodreg_vicreg_splitdec_400k.out
#SBATCH --error=out-multiemb/%j-ft_v4_soupv1_openarm_prq_embodreg_vicreg_splitdec_400k.err

set -ex  # -e: abort if Stage-1 fails
export PATH="$HOME/.local/bin:$PATH"
export WANDB_PROJECT=Action-Tokenizer-Joint-SoupV1
export WANDB_API_KEY="66a73856614bc24a07523f3fbee42482fcbeffe3"
export HF_TOKEN="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"
MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/jungwook/action_tokenizer/checkpoints_action_tokenizer
#MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/jungwook/action_tokenizer/checkpoints_action_tokenizer sbatch '/sjw_alinlab1/home/jungwook/action_tokenizer/sbatch_scripts/multiemb/v4_soupv1/recon_dino_bn64_l1_mse_naiveln_vae_embtok_finetune_openarm_prq_embodreg_splitdec_400k.sh'
BASE_DIR=/sjw_alinlab1/home/jungwook/action_tokenizer
cd $BASE_DIR
source /sjw_alinlab1/home/jungwook/miniconda3/bin/activate gr00t-actlat
export PATH="$CONDA_PREFIX/bin:$PATH"

# The cached-DINO reader memoizes one mmap handle per episode without bound; with the
# default soft limit of 1024 a multi-source run exhausts fds mid-training and surfaces as
# an opaque NCCL timeout (EXP-0007 106015/106625). Raise it to the hard limit up front.
ulimit -Sn "$(ulimit -Hn)"

# === EXP-0010: embod_reg on the action latent + per-domain recon decoder split ===
# Identical to recon_dino_bn64_l1_mse_naiveln_vae_embtok_finetune_openarm_prq_400k.sh
# (same soupv1-400k parent, same 5 sources, same hyper-params) EXCEPT the two additive
# features below, so the comparison against that run is single-variable-per-flag.
#
# A) --embod-reg-mode vicreg: aligns the human vs robot ACTION LATENT distributions. The
#    label is per-SAMPLE, not per-group: robot (openarm_teleop_v3, FK→{p,r,q}) and human
#    (pnp_clean_260506, eef→{p,r,q}) share ONE embodiment group here, so the loader tags
#    each sample with is_human and the regularizer contrasts within the batch.
#    vicreg (not meanshift) because the reference study measured meanshift COLLAPSING --
#    its apparent gain was a small-batch variance-shrinkage artifact, and only vicreg's
#    variance hinge opposes a joint collapse. --embod-reg-gather is on by default and is
#    effectively mandatory: with a per-rank micro-batch the centroid estimator's variance
#    term dominates the true mean gap.
#    Watch in wandb: loss_embod_reg, embod_reg_gap (should shrink), embod_reg_std_min
#    (must NOT drift toward 0 -- that is the collapse alarm), embod_reg_n_human/_n_robot
#    (must both be ~half the global batch; a zero means the labels stopped arriving).
# B) --split-recon-decoder: ONE shared action encoder, TWO recon decoders. The base
#    decoder stays the robot one and a __human twin is added, copy-initialized from it
#    (--split-recon-decoder-init copy) so step 0 is numerically unchanged. Motivation is
#    inference flexibility: decode the same latent "the robot way" or "the human way".
#    Stage-2 selects with --embodiment-id openarm_prq (robot) or openarm_prq__human.
# MEASURED BASIS for the embod-reg values below (scripts/exp0010_zstats.py, 4096 samples
# off .../ft_openarm_prq_400k/checkpoint-200000, 2026-08-19). Nothing here is a guess:
#   * z per-dim std = 1.79 med (pooled) / 2.18 med (per-token). The reference's hardcoded
#     hinge floor of 1.0 is therefore INERT for us (var term 1e-5) and would permit a ~45%
#     shrink before resisting -- so --embod-reg-vic-std 1.75, just under the p10 of the
#     measured per-token std, which blocks collapse without pulling at the current point.
#   * cov at the reference vic_cov=0.04 was 5.41, i.e. 84% of the whole regularizer: the
#     term would have been a decorrelation objective wearing an alignment objective's
#     name. --embod-reg-vic-cov 0.004 puts it at 31% (inv 69%).
#   * H/R centroid gap = 0.887 unstratified, but 1.223 when contrasted WITHIN time-token
#     bins, and the per-bin gap grows down the chunk (t0 0.89 -> t15 1.90, max 2.86). The
#     misalignment is concentrated LATE and the marginal contrast hides a third of it,
#     hence --embod-reg-pool tokens.
#   * centroid estimator floor: 0.324 on a per-rank micro-batch vs 0.040 with the 8-GPU
#     all-gather, against a true gap of 0.89 -- 36% bias without gather, 4% with it.
#     --embod-reg-gather is not optional.
#   * loss_embod_reg at these settings evaluates to 1.766 on the trained tokenizer, and
#     that run's total loss was 0.0189 (recon 0.00208, dino 0.1635 x 0.1, kl 407 x 1e-6).
#     --embod-reg-weight 0.003 -> a 0.0053 contribution: ~28% of the total and ~2.5x the
#     recon term, which has room to give (it converged at 0.002). If loss_recon more than
#     doubles by 5k steps, or embod_reg_gap has not started falling, stop and re-weight.
EMB_CONFIG=/sjw_alinlab1/home/jungwook/action_tokenizer/sbatch_scripts/multiemb/v4_soupv1/embodiments_openarm_prq_finetune.json
PRETRAIN_CKPT_DIR=checkpoints_action_tokenizer/joint_soupv1_v4_recon_dino_bn64_l1_mse_naiveln_vae_embtok
PRETRAIN_STEP=400000
ABS_PRETRAIN="$BASE_DIR/$PRETRAIN_CKPT_DIR/checkpoint-$PRETRAIN_STEP"
TOK_CKPT_DIR=$MODEL_OUTPUT_DIR/checkpoints_action_tokenizer/joint_soupv1_v4_recon_dino_bn64_l1_mse_naiveln_vae_embtok_ft_openarm_prq_embodreg_vicreg_splitdec_400k_ft100k
FT_STEP=100000

python scripts/train_action_latent_tokenizer_v4_multiemb.py \
    --resume \
    --embodiments-config "$EMB_CONFIG" \
    --output-dir $TOK_CKPT_DIR \
    --run-name "actlat_v4_ft_soupv1_openarm_prq_embodreg_vicreg_splitdec_recon_dino_bn64_l1_mse_naiveln_vae_embtok_400k" \
    --num-gpus 8 \
    --batch-size 64 \
    --max-steps $FT_STEP \
    --save-steps 5000 \
    --save-total-limit 10 \
    --dataloader-num-workers 16 \
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
    --use-embodiment-class-token \
    --tokenizer-finetuning-mode \
    --finetuning-freeze-mode \
    --new-class-token 1 \
    --finetuning-pretrained-path "$ABS_PRETRAIN" \
    --embod-reg-mode vicreg \
    --embod-reg-weight 0.003 \
    --embod-reg-gather \
    --embod-reg-pool tokens \
    --embod-reg-vic-var 1.0 \
    --embod-reg-vic-std 1.75 \
    --embod-reg-vic-cov 0.004 \
    --split-recon-decoder \
    --split-recon-decoder-init copy \
    --video-backend decord \
    --use-fixed-val \
    --wandb-project "Action-Tokenizer-Joint-SoupV1-tokenizer-ft" \
    --eval-steps 1000 \
    --report-to wandb

# Ablation variants (same file, one flag each):
#   A only : drop --split-recon-decoder / --split-recon-decoder-init
#   B only : drop the five --embod-reg-* lines
# Change the run-name / output-dir to match when you do.
