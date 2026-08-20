#!/bin/bash
#SBATCH --job-name=scratch_v4_openarm_prq_embodreg_vicreg_w0p003_splitdec_recon_dino_bn64_l1_mse_naiveln_vae_embtok_400k
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --partition=sjw_alinlab,h100
#SBATCH --wckey=project-short-name:sub_human
#SBATCH --output=out-multiemb/%j-scratch_v4_openarm_prq_embodreg_w0p003_splitdec_400k.out
#SBATCH --error=out-multiemb/%j-scratch_v4_openarm_prq_embodreg_w0p003_splitdec_400k.err

set -ex  # -e: abort if training fails
export PATH="$HOME/.local/bin:$PATH"
export WANDB_PROJECT=Action-Tokenizer-Joint-SoupV1
export WANDB_API_KEY="66a73856614bc24a07523f3fbee42482fcbeffe3"
export HF_TOKEN="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"
MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/jungwook/action_tokenizer/checkpoints_action_tokenizer
#MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/jungwook/action_tokenizer/checkpoints_action_tokenizer sbatch '/sjw_alinlab1/home/jungwook/action_tokenizer/sbatch_scripts/multiemb/v4_soupv1/recon_dino_bn64_l1_mse_naiveln_vae_embtok_scratch_openarm_prq_embodreg_w0p003_splitdec_400k.sh'
BASE_DIR=/sjw_alinlab1/home/jungwook/action_tokenizer
cd $BASE_DIR
source /sjw_alinlab1/home/jungwook/miniconda3/bin/activate gr00t-actlat
export PATH="$CONDA_PREFIX/bin:$PATH"

# The cached-DINO reader memoizes one mmap handle per episode without bound; with the
# default soft limit of 1024 a multi-source run exhausts fds mid-training and surfaces as
# an opaque NCCL timeout (EXP-0007 106015/106625). Raise it to the hard limit up front.
ulimit -Sn "$(ulimit -Hn)"

# === EXP-0012: embod_reg + split recon decoder trained FROM SCRATCH (random init) ===
# This is the from-scratch counterpart of
#   recon_dino_bn64_l1_mse_naiveln_vae_embtok_finetune_openarm_prq_embodreg_splitdec_400k.sh
# (EXP-0010, job 109760), which adapted the soupv1-400k tokenizer with a FROZEN backbone.
# Here there is no parent checkpoint: every parameter is randomly initialized and trains.
#
# DERIVED FROM THE PRETRAIN RECIPE, NOT THE FINETUNE ONE. The finetune script's schedule
# (100k steps, save every 5k, limit 10) is a freeze-mode adaptation budget and is wrong for
# random init. The values below come from the script that produced the soupv1 400k parent,
# sbatch_scripts/multiemb/v4_soupv1/recon_dino_bn64_l1_mse_naiveln_vae_embtok.sh:
#   max-steps 400000 (vs ft 100000), save-steps 10000 (vs 5000), save-total-limit 100
#   (vs 10), num-gpus 8 x batch-size 64 (vs the ft run's 2 x 256; global batch 512 either
#   way), wandb project suffix -scratch so it does not land in the ft run's group.
# Only the SBATCH header, MODEL_OUTPUT_DIR, ulimit and conda path are taken from the ft
# script -- the pretrain script targets a different cluster (BASE_DIR=/NHNHOME/...,
# --gres=gpu:b200:8) and its scaffolding must not be copied here.
# lr/warmup are NOT set by either script, so both inherit the code defaults and are
# therefore already identical: learning_rate 5e-5, weight_decay 1e-5, warmup_ratio 0.05,
# lr_scheduler_type "constant" (note: HF's constant schedule ignores warmup_ratio, so the
# 0.05 is inert -- true of the 400k parent as well, so this run matches it).
#
# MEMORY (freeze removal raises trainable params 17x: 9.6M -> ~164.3M of 164.3M total).
# Precedent on THIS cluster, not an extrapolation: jobs 36606 and 38847
# (joint_v4_soupv2_..._400k) trained 186.6M/180.3M params with everything trainable at
# exactly --num-gpus 8 --batch-size 64 --max-steps 400000 --save-steps 10000. Those models
# are LARGER than this one, so 8x64 has headroom. The DINOv2-large backbone (~300M) is not
# in the parameter count because the DINO features are read from cache, so the optimizer
# state is the dominant term: ~164.3M x 8B ~= 1.3GB, trivial on 80GB H100s.
# Disk: full-trainable optimizer state makes each checkpoint ~2.0GB (vs 734MB in freeze
# mode, where optimizer.pt covered only the 9.6M unfrozen params). 400k/10k = 40 saves
# ~= 80GB for this run. save-total-limit 100 is above that on purpose: nothing is deleted.
#
# CLASS TOKEN, from scratch: the JSON declares class_token_id 5 (it was the 6th slot added
# to the soupv1 0..4 set by the ft run's --new-class-token 1). Off the finetuning path the
# model sizes the table as max(class_token_id)+1 = 6 rows and uses row 5, leaving rows 0-4
# as 5 x 1024 = 5,120 unused parameters -- harmless, and keeping id 5 means the class-token
# row index matches the EXP-0010 checkpoint, which remap_to_single_embodiment resolves via
# the _class_token_id__openarm_prq buffer either way. Do NOT re-number it to 0 just to save
# 5,120 params; that would fork the embodiments JSON for no benefit.
#
# ⚠️ THE embod_reg CONSTANTS BELOW LOSE THEIR MEASURED BASIS IN THIS RUN. In EXP-0010 the
# values were not guesses: scripts/exp0010_zstats.py measured the ALREADY-TRAINED
# tokenizer's z distribution (4096 samples off ft_openarm_prq_400k/checkpoint-200000) and
# every constant was set from it -- vic_std 1.75 just under the p10 of the measured
# per-token std (1.79 pooled / 2.18 per-token median), vic_cov 0.004 to hold the covariance
# term at 31% of the regularizer instead of the reference's 84%, and weight 0.003 to put
# loss_embod_reg (1.766 there) at ~28% of a converged total loss of 0.0189.
# A randomly-initialized tokenizer has none of that:
#   * the std hinge at 1.75 is set against a converged z scale. Early z statistics differ,
#     so the hinge may sit fully active (if initial per-dim std < 1.75, it pushes the scale
#     UP from step 0) or fully inert -- neither was the measured intent.
#   * recon loss starts orders of magnitude above its converged 0.002, so weight 0.003
#     buys the regularizer a far smaller RELATIVE share than the ~28% it had in EXP-0010,
#     i.e. embod_reg is effectively weaker at exactly the point where the representation
#     is being formed. This is the reason a 10x variant (w0p03) exists as a sibling script.
# TREAT THESE AS PROVISIONAL: they are pretrained-z numbers and from-scratch needs its own
# measurement. Re-run scripts/exp0010_zstats.py against an early checkpoint of THIS run
# (e.g. checkpoint-10000) before drawing any conclusion about the hinge or the weight.
# Watch in wandb: loss_embod_reg, embod_reg_gap, embod_reg_std_min (must NOT drift toward
# 0 -- collapse alarm), embod_reg_n_human/_n_robot (both ~half the global batch; a zero
# means the is_human labels stopped arriving).
#
# --split-recon-decoder-init copy is KEPT and is meaningful without a parent: the human
# twin is copied from the robot decoder's RANDOM init, so both decoders start from the same
# point and diverge only under their own domain's gradients. The copy block in
# scripts/train_action_latent_tokenizer_v4_multiemb.py is gated on
# (split_recon_decoder and init == "copy" and not is_resuming) only -- it does not require
# tokenizer_finetuning_mode, so it runs on this path.
#
# The embodiments JSON is deliberately the same file the ft run used: embod_reg needs the
# robot source (openarm_teleop_v3, FK->{p,r,q}) and the human source (pnp_clean_260506,
# eef->{p,r,q}) inside ONE embodiment group so the loader can tag each sample with is_human
# and the regularizer can contrast within the batch. The "_finetune" in the filename
# describes where the file was first used, not a constraint on this run.
EMB_CONFIG=/sjw_alinlab1/home/jungwook/action_tokenizer/sbatch_scripts/multiemb/v4_soupv1/embodiments_openarm_prq_finetune.json
TOK_CKPT_DIR=$MODEL_OUTPUT_DIR/checkpoints_action_tokenizer/joint_openarm_prq_v4_scratch_recon_dino_bn64_l1_mse_naiveln_vae_embtok_embodreg_vicreg_w0p003_splitdec_400k
TOK_STEP=400000

python scripts/train_action_latent_tokenizer_v4_multiemb.py \
    --resume \
    --embodiments-config "$EMB_CONFIG" \
    --output-dir $TOK_CKPT_DIR \
    --run-name "actlat_v4_scratch_openarm_prq_embodreg_vicreg_w0p003_splitdec_recon_dino_bn64_l1_mse_naiveln_vae_embtok_400k" \
    --num-gpus 8 \
    --batch-size 64 \
    --max-steps $TOK_STEP \
    --save-steps 10000 \
    --save-total-limit 100 \
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
    --wandb-project "Action-Tokenizer-Joint-SoupV1-tokenizer-scratch" \
    --eval-steps 1000 \
    --report-to wandb

# Sibling: ..._scratch_openarm_prq_embodreg_w0p03_splitdec_400k.sh is this file with
# --embod-reg-weight 0.03 (10x) and its own run-name / output-dir / out-err names.
# Nothing else differs between the two, so the pair is a single-variable weight sweep.
