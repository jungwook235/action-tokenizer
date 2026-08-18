"""Embodiment-agnostic regularizer on the tokenizer's ACTION LATENT ``z``.

Additive, default-OFF (built only when ``embod_reg_mode != ""``). Given a pooled
latent ``h (N, d)`` and per-sample domain labels ``is_human (N,)`` (float 0=robot,
1=human), it pulls the human vs robot latent distributions together so that the
action latent the Stage-2 policy consumes is embodiment-agnostic.

Adapted for EXP-0010 from the reference implementation in
``/sjw_alinlab/home/junhyeong/RLDX-1-egopi/rldx/model/modules/embod_reg.py``
(read-only reference; this is our own re-implementation). Two deliberate
differences from the reference:

  * The reference taps an early-DiT hidden and can stratify by the flow-matching
    time ``t``. Stage-1 has no flow time, so ``t``/``t_bins`` are dropped; the
    stratification hook survives as a generic ``bin_ids`` (we feed the action
    latent's TIME-TOKEN index, which is the analogous "matched phase" axis).
  * The caller pools ACROSS embodiment groups before calling us, so ``N`` equals
    the full per-rank micro-batch on every rank. That is what makes ``gather``
    safe here (see ``_sizes_agree``).

Modes:
  * ``vicreg``    : invariance(centroid) + variance hinge(per-dim std >= 1) +
                    covariance(off-diag). RECOMMENDED DEFAULT.
  * ``coral``     : ``||cov(H) - cov(R)||_F^2 / (4 d^2)`` (second-order match).
  * ``meanshift`` : ``||mean(H) - mean(R)||_2^2 / d`` (first-order match).
  * ``dann``      : gradient-reversed ``d -> 256 -> 1`` domain classifier (BCE).

Why vicreg is the default and not meanshift — the reference author's measurements
(quoted in EXP-0010's EXPERIMENT.md): meanshift alone COLLAPSED. Its good-looking
local gap (0.079) was a 4-sample variance-shrinkage artifact that vanished once
all-gather fixed the statistics (0.169 == baseline), and every meanshift variant
made the late-t gap WORSE (1.10-1.44 vs baseline 0.875). "Pull the two centroids
together" is trivially satisfied by shrinking both clouds; only the variance hinge
opposes that.

DDP-safety: when a batch lacks >=2 samples on either side we return a zero loss
that still *touches* every module parameter, so no rank drops a param from the
autograd graph. All internal math runs in float32.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

VALID_EMBOD_REG_MODES = ("coral", "meanshift", "dann", "vicreg")


def _dist_on() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _all_gather_keep_grad(t: torch.Tensor) -> torch.Tensor:
    """all_gather where the local slice keeps its gradient. No-op outside DDP."""
    if not _dist_on():
        return t
    out = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(out, t.contiguous())
    out[dist.get_rank()] = t  # keep grad on our own slice
    return torch.cat(out, 0)


def _sizes_agree(n: int, device) -> bool:
    """True iff every rank brings the same N (precondition for a shaped all_gather).

    Every rank runs this collective and sees the same answer, so the gather/no-gather
    decision below can never diverge across ranks (which would deadlock).
    """
    if not _dist_on():
        return True
    ns = [torch.zeros((), dtype=torch.long, device=device) for _ in range(dist.get_world_size())]
    dist.all_gather(ns, torch.tensor(n, dtype=torch.long, device=device))
    return len({int(x.item()) for x in ns}) == 1


class _GradientReversal(torch.autograd.Function):
    """Identity forward; negated (scaled) gradient backward (DANN)."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return _GradientReversal.apply(x, lambda_)


class EmbodAgnosticReg(nn.Module):
    """Align human vs robot action-latent distributions.

    Args:
        d: latent width (``token_dim``, 64 here).
        mode: one of :data:`VALID_EMBOD_REG_MODES`.
        lambda_: GRL strength (``dann`` only). The loss WEIGHT lives in the model.
        gather: all-gather (h, is_human, bin_ids) across ranks before contrasting.
            Effectively mandatory: with a small per-rank micro-batch the plug-in
            centroid estimator's variance term (trSigma/n) dominates the true mean
            gap. The loss value is identical on all ranks and the local slice keeps
            grad, so DDP grad-averaging reconstructs the global-batch gradient.
        vic_var / vic_cov: vicreg weights for the variance hinge and the off-diagonal
            covariance penalty, relative to the invariance term. The variance hinge is
            the whole point — it is what every first-moment-only design lacked.
    """

    def __init__(self, d: int, mode: str = "vicreg", lambda_: float = 1.0,
                 gather: bool = True, vic_var: float = 1.0, vic_cov: float = 0.04):
        super().__init__()
        if mode not in VALID_EMBOD_REG_MODES:
            raise ValueError(f"unknown embod_reg mode: {mode!r} (valid: {VALID_EMBOD_REG_MODES})")
        self.d = int(d)
        self.mode = mode
        self.lambda_ = float(lambda_)
        self.gather = bool(gather)
        self.vic_var = float(vic_var)
        self.vic_cov = float(vic_cov)
        # Diagnostics, refreshed every forward (the trainer logs them as scalars).
        self.last_nh = 0.0        # human sample count actually contrasted
        self.last_nr = 0.0        # robot sample count actually contrasted
        self.last_bins = 0.0      # stratification bins that had >=2 on both sides
        self.last_gap = 0.0       # ||mean(H)-mean(R)||^2 / d, measured (not the loss)
        self.last_std_min = 0.0   # min per-dim std over both streams (collapse alarm)
        if mode == "dann":
            self.clf = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 1))

    def _zero_touch(self, h: torch.Tensor) -> torch.Tensor:
        """Zero loss that keeps every module param in the autograd graph (DDP)."""
        z = h.float().sum() * 0.0
        for p in self.parameters():
            z = z + p.sum() * 0.0
        return z

    @staticmethod
    def _cov(x: torch.Tensor) -> torch.Tensor:
        xc = x - x.mean(0, keepdim=True)
        return (xc.t() @ xc) / (x.shape[0] - 1)

    def _stratified_inv(self, h, human, robot, bin_id, n_bins):
        """Centroid contrast computed WITHIN bins and pooled, weighted by min(n_h, n_r).

        The unconditional contrast asks only that the two marginals overlap, which says
        nothing about correspondence: the clouds can coincide with every human sample
        matched to the wrong robot sample. Conditioning on a bin that means the same
        thing on both streams (here: the same action-chunk time step) turns the same
        machinery into "at the same phase, the two streams must look alike".

        Returns None when no bin has >=2 samples on both sides.
        """
        num, den = None, 0.0
        for b in range(n_bins):
            hb = human & (bin_id == b)
            rb = robot & (bin_id == b)
            nh, nr = int(hb.sum().item()), int(rb.sum().item())
            if nh < 2 or nr < 2:
                continue
            diff = h[hb].mean(0) - h[rb].mean(0)
            w = float(min(nh, nr))
            term = w * (diff * diff).sum() / self.d
            num = term if num is None else num + term
            den += w
            self.last_bins += 1.0
        return None if num is None else num / den

    def forward(self, h: torch.Tensor, is_human: torch.Tensor,
                bin_ids: torch.Tensor | None = None) -> torch.Tensor:
        """``h`` [N,d] pooled latent, ``is_human`` [N] in {0,1}, optional ``bin_ids`` [N]."""
        h = h.float()
        is_human = is_human.reshape(-1).float()
        if self.gather and _sizes_agree(h.shape[0], h.device):
            h = _all_gather_keep_grad(h)
            is_human = _all_gather_keep_grad(is_human)
            if bin_ids is not None:
                bin_ids = _all_gather_keep_grad(bin_ids.reshape(-1).float())
        if bin_ids is not None:
            bin_ids = bin_ids.reshape(-1).long()

        human = is_human > 0.5
        robot = ~human
        n_h, n_r = int(human.sum().item()), int(robot.sum().item())
        self.last_nh, self.last_nr, self.last_bins = float(n_h), float(n_r), 0.0
        self.last_gap, self.last_std_min = 0.0, 0.0

        # Need both domains present (>=2 each) to form a contrast. Otherwise emit a
        # graph-touching zero so every rank agrees on the param set.
        if n_h < 2 or n_r < 2:
            return self._zero_touch(h)

        if self.mode == "dann":
            # dtype guard: under bf16 the classifier params may be CAST (not autocast)
            # while h arrives fp32. Cast the feature to the classifier's own dtype; the
            # GRL sits before the cast so its backward is unaffected.
            wdt = self.clf[0].weight.dtype
            logits = self.clf(grad_reverse(h, self.lambda_).to(wdt)).reshape(-1)
            return F.binary_cross_entropy_with_logits(logits.float(), is_human.float())

        H, R = h[human], h[robot]
        with torch.no_grad():  # measurement only — never a gradient path
            gap = H.mean(0) - R.mean(0)
            self.last_gap = float((gap * gap).sum().item() / self.d)
            self.last_std_min = float(
                min(torch.sqrt(H.var(0, unbiased=False) + 1e-4).min().item(),
                    torch.sqrt(R.var(0, unbiased=False) + 1e-4).min().item())
            )

        strat_inv = None
        if bin_ids is not None and bin_ids.numel() == h.shape[0]:
            strat_inv = self._stratified_inv(h, human, robot, bin_ids,
                                             int(bin_ids.max().item()) + 1)

        if self.mode == "meanshift":
            if bin_ids is not None:
                return strat_inv if strat_inv is not None else self._zero_touch(h)
            diff = H.mean(0) - R.mean(0)
            return (diff * diff).sum() / self.d

        if self.mode == "vicreg":
            # Variance-PRESERVING alignment:
            #   invariance  the centroid pull (what we want)
            #   variance    hinge keeping EACH stream's per-dim std >= 1 (anti-collapse)
            #   covariance  off-diagonal penalty per stream, so the preserved variance is
            #               spread over directions instead of piling into one
            # Standard VICReg weights (25/25/1) are rescaled so the invariance term keeps
            # the same magnitude as meanshift and ``embod_reg_weight`` keeps its meaning.
            # Conditional form: with bins the invariance term is the WITHIN-BIN contrast;
            # variance/covariance stay global (they are per-stream anti-collapse terms
            # and carry no cross-stream statement to condition on).
            if strat_inv is not None:
                inv = strat_inv
            elif bin_ids is not None:
                return self._zero_touch(h)
            else:
                diff = H.mean(0) - R.mean(0)
                inv = (diff * diff).sum() / self.d

            var = h.new_zeros(())
            cov = h.new_zeros(())
            for X in (H, R):
                std = torch.sqrt(X.var(0, unbiased=False) + 1e-4)
                var = var + F.relu(1.0 - std).pow(2).mean()
                C = self._cov(X)
                off = C - torch.diag_embed(torch.diagonal(C))
                cov = cov + (off * off).sum() / self.d
            return inv + self.vic_var * 0.5 * var + self.vic_cov * 0.5 * cov

        # coral: Frobenius^2 of the covariance difference, normalized by 4 d^2.
        delta = self._cov(H) - self._cov(R)
        return (delta * delta).sum() / (4.0 * self.d * self.d)
