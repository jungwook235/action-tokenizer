#!/bin/bash
#SBATCH --job-name=scratch_v4_openarm_prq_embodreg_vicreg_w0p03_splitdec_recon_dino_bn64_l1_mse_naiveln_vae_embtok_100k
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --partition=sjw_alinlab,h100
#SBATCH --wckey=project-short-name:sub_human
#SBATCH --output=out-multiemb/%j-scratch_v4_openarm_prq_embodreg_w0p03_splitdec_100k.out
#SBATCH --error=out-multiemb/%j-scratch_v4_openarm_prq_embodreg_w0p03_splitdec_100k.err

set -ex  # -e: abort if Stage-1 fails
export PATH="$HOME/.local/bin:$PATH"
export WANDB_PROJECT=Action-Tokenizer-Joint-SoupV1
export WANDB_API_KEY="66a73856614bc24a07523f3fbee42482fcbeffe3"
export HF_TOKEN="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"
MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/jungwook/action_tokenizer/checkpoints_action_tokenizer
#MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/jungwook/action_tokenizer/checkpoints_action_tokenizer sbatch '/sjw_alinlab1/home/jungwook/action_tokenizer/sbatch_scripts/multiemb/v4_soupv1/recon_dino_bn64_l1_mse_naiveln_vae_embtok_scratch_openarm_prq_embodreg_w0p03_splitdec_100k.sh'
BASE_DIR=/sjw_alinlab1/home/jungwook/action_tokenizer
cd $BASE_DIR
source /sjw_alinlab1/home/jungwook/miniconda3/bin/activate gr00t-actlat
export PATH="$CONDA_PREFIX/bin:$PATH"

# The cached-DINO reader memoizes one mmap handle per episode without bound; with the
# default soft limit of 1024 a multi-source run exhausts fds mid-training and surfaces as
# an opaque NCCL timeout (EXP-0007 106015/106625). Raise it to the hard limit up front.
ulimit -Sn "$(ulimit -Hn)"

# === EXP-0012: EXP-0010's two features, but trained FROM SCRATCH (random init) ===
# Identical to recon_dino_bn64_l1_mse_naiveln_vae_embtok_finetune_openarm_prq_embodreg_splitdec_400k.sh
# (EXP-0010, job 109760) EXCEPT that nothing is initialized from the soupv1-400k parent:
# --tokenizer-finetuning-mode, --finetuning-freeze-mode, --finetuning-pretrained-path and
# --new-class-token are dropped, so every parameter is randomly initialized and trains.
# EVERY OTHER VALUE IS DELIBERATELY UNCHANGED -- including --max-steps 100000 ($FT_STEP),
# --save-steps 5000, --save-total-limit 10, 8 GPUs x batch 64, and the wandb project. This
# is a single-axis change (pretrained-init on/off), NOT a re-tuned pretrain recipe, so the
# comparison against 109760 stays interpretable. Do not "upgrade" the schedule here.
# ($FT_STEP keeps its name so the diff against the EXP-0010 script stays minimal; it is
# just the step budget, and it is still 100000.)
#
# --new-class-token was REMOVED because the code makes it finetuning-only, not as a
# judgement call: gr00t/model/action_latent_tokenizer_v4_multiemb.py asserts
#   if new_class_token > 0: assert tokenizer_finetuning_mode,
#       "new_class_token > 0 is only valid in tokenizer_finetuning_mode."
# so keeping it here would abort at model construction. Off the finetuning path the class
# token table is instead sized max(class_token_id)+1 = 6 rows and this group uses row 5
# (rows 0-4 stay as 5 x 1024 = 5,120 unused params -- harmless, and it keeps the row index
# identical to the EXP-0010 checkpoint). _class_token() then resolves to
# embodiment_class_token[cid] directly, and remap_to_single_embodiment falls back to the
# same key for Stage-2, so --use-embodiment-class-token stays ON and works.
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
#    From random init the copy still applies and is still what we want: the twin is copied
#    from the robot decoder's RANDOM weights, so both start at the same point and diverge
#    only under their own domain's gradients. Verified in
#    scripts/train_action_latent_tokenizer_v4_multiemb.py: the copy block is gated on
#    (split_recon_decoder and split_recon_decoder_init == "copy" and not is_resuming)
#    ONLY -- it does not require tokenizer_finetuning_mode, so it runs on this path.
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
#
# ⚠️ THAT MEASURED BASIS DOES NOT TRANSFER TO RANDOM INIT -- the constants above are kept
# unchanged on purpose (single-axis change), but they were derived from an ALREADY-TRAINED
# tokenizer's z distribution, so here they are provisional rather than measured:
#   * --embod-reg-vic-std 1.75 was placed just under the p10 of a CONVERGED per-token std.
#     A freshly initialized encoder's z scale is not that distribution, so the hinge may
#     sit fully active (pushing the scale up from step 0) or fully inert. Neither is the
#     measured intent.
#   * --embod-reg-weight 0.003 was sized to be ~28% of a converged total loss of 0.0189.
#     Early recon loss is orders of magnitude above its converged 0.002, so the same
#     weight buys the regularizer a far smaller RELATIVE share exactly while the
#     representation is being formed. THIS SCRIPT IS THAT 10x BRACKET: it is the only
#     value changed from EXP-0010 (0.003 -> 0.03), and 0.03 is not a measured number
#     either -- it is one decade up from a value whose basis no longer applies. If
#     loss_recon stays badly elevated against the w0p003 sibling by 5k steps, the weight
#     is too high; that is a weight verdict, not a hinge verdict.
# Re-measure with scripts/exp0010_zstats.py on an early checkpoint of THIS run (e.g.
# checkpoint-5000) before concluding anything about the hinge or the weight. Do not
# retro-fit the numbers above; they document EXP-0010.
EMB_CONFIG=/sjw_alinlab1/home/jungwook/action_tokenizer/sbatch_scripts/multiemb/v4_soupv1/embodiments_openarm_prq_finetune.json
TOK_CKPT_DIR=$MODEL_OUTPUT_DIR/checkpoints_action_tokenizer/joint_openarm_prq_v4_scratch_recon_dino_bn64_l1_mse_naiveln_vae_embtok_embodreg_vicreg_w0p03_splitdec_100k
FT_STEP=100000

python scripts/train_action_latent_tokenizer_v4_multiemb.py \
    --resume \
    --embodiments-config "$EMB_CONFIG" \
    --output-dir $TOK_CKPT_DIR \
    --run-name "actlat_v4_scratch_openarm_prq_embodreg_vicreg_w0p03_splitdec_recon_dino_bn64_l1_mse_naiveln_vae_embtok_100k" \
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
    --embod-reg-mode vicreg \
    --embod-reg-weight 0.03 \
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

# Sibling: ..._scratch_openarm_prq_embodreg_w0p003_splitdec_100k.sh is this file with
# --embod-reg-weight 0.003 (the EXP-0010 value) and its own job-name / run-name /
# output-dir / out-err names. Nothing else differs, so the pair is a single-variable
# weight sweep.
