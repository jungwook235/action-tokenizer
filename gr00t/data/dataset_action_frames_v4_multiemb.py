"""Multi-embodiment data plumbing for the joint V4 tokenizer.

Mixes several embodiments (each with its own ``data_config`` / dataset paths /
normalization stats) into one training run. Because ``action_dim`` differs per
embodiment, samples cannot be stacked into a single tensor — so a batch carries
samples from multiple embodiments and the collator splits them into per-embodiment
groups (each group stacks independently). The trainer/model then run one forward
per group through the per-embodiment encoder/decoder + the shared fusion/DINO
decoder, summing the losses.

Components:
  * ``EmbodimentTaggedDataset``  — wraps an ``ActionFramesDatasetV4`` so each item
    carries its ``embodiment`` name.
  * ``MultiEmbActionFramesCollator`` — groups a mixed feature list by embodiment.
  * ``WeightedEmbodimentSampler`` — optional per-embodiment sampling weights;
    only used when weights are given (default: plain size-proportional shuffle).
"""

import numpy as np
import torch

from gr00t.data.dataset_action_frames_v4 import ActionFramesCollatorV4
from gr00t.data.dataset_dino_cache_v4 import CachedActionFramesCollatorV4
from gr00t.experiment.trainer import BaseSampler


class EmbodimentTaggedDataset(torch.utils.data.Dataset):
    """Wrap an ``ActionFramesDatasetV4`` so items carry their embodiment name."""

    def __init__(self, base: torch.utils.data.Dataset, embodiment: str):
        self.base = base
        self.embodiment = embodiment

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        item = self.base[index]
        item["embodiment"] = self.embodiment
        return item

    def set_epoch(self, epoch):
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(epoch)


class MultiEmbActionFramesCollator:
    """Group a mixed feature list by ``embodiment`` and stack each group.

    Returns::

        {"embodiment_order": [name, ...],
         "groups": {name: {"action": [b,T,D], "frame_x0": [b,3,H,W], "frame_x1": ...}}}

    Within a group, ``action`` tensors share ``action_dim`` so they stack cleanly;
    across groups ``action_dim`` may differ. Frame stacking reuses
    ``ActionFramesCollatorV4`` so frame handling stays identical to single-emb.

    Mixed cache/live: an embodiment using a precomputed DINO cache yields items
    with ``x0_feat``/``x1_feat`` (no frames). Each bucket is homogeneous (one
    embodiment), so we pick the collator per bucket: ``CachedActionFramesCollatorV4``
    for cached groups (emits ``x0_feat``/``x1_feat``), ``ActionFramesCollatorV4``
    for live groups (emits ``frame_x0``/``frame_x1``). The trainer then either uses
    the cached feats directly or runs DINO on the frames, per group.
    """

    def __init__(self, pass_is_human: bool = False):
        # pass_is_human ([EXP-0010]): also stack the per-sample ``is_human`` label into
        # each group, for the embodiment regularizer and/or the per-domain decoder split.
        # Off by default -> the emitted batch is byte-identical to before.
        self.pass_is_human = bool(pass_is_human)
        self._frame_collator = ActionFramesCollatorV4()
        self._cached_collator = CachedActionFramesCollatorV4()

    def __call__(self, features: list[dict]) -> dict:
        buckets: dict[str, list[dict]] = {}
        order: list[str] = []
        for f in features:
            name = f["embodiment"]
            if name not in buckets:
                buckets[name] = []
                order.append(name)
            buckets[name].append(f)

        groups = {}
        for name, feats in buckets.items():
            # Cached items carry x0_feat (no frames); live items carry frame_x0.
            # Both collators ignore the extra "embodiment" key.
            if "x0_feat" in feats[0]:
                groups[name] = self._cached_collator(feats)
            else:
                groups[name] = self._frame_collator(feats)
            # [EXP-0010] Stacked here rather than inside the two sub-collators so both
            # the cached and the live path get it from one place.
            if self.pass_is_human and "is_human" in feats[0]:
                groups[name]["is_human"] = torch.tensor(
                    [float(f["is_human"]) for f in feats], dtype=torch.float32
                )
        return {"embodiment_order": order, "groups": groups}


class WeightedEmbodimentSampler(BaseSampler):
    """Per-embodiment weighted sampler (mirrors ``BaseSampler`` DDP behavior).

    Like ``BaseSampler`` it returns a full-length index list identical across
    ranks (accelerate handles per-rank sharding — do NOT add rank here). Each
    index ``i`` is drawn with probability ∝ ``weights[i]``, where
    ``weights[i] = group_weight[emb(i)] / group_size[emb(i)]`` so each embodiment's
    total probability mass equals its configured weight.

    Only needed when explicit weights are given; otherwise use plain
    ``BaseSampler(shuffle=True)`` (size-proportional).
    """

    def __init__(self, data_source, per_index_weights, seed: int = 0):
        super().__init__(data_source, shuffle=True, seed=seed)
        self.weights = torch.as_tensor(per_index_weights, dtype=torch.double)
        assert len(self.weights) == len(data_source), (
            f"weights len {len(self.weights)} != dataset len {len(data_source)}"
        )

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        idx = torch.multinomial(
            self.weights, len(self.data_source), replacement=True, generator=g
        )
        return iter(idx.tolist())


def build_per_index_weights(group_sizes: list[int], group_weights: list[float]) -> np.ndarray:
    """Per-sample weights for ``WeightedEmbodimentSampler`` over a ConcatDataset
    whose member datasets appear in ``group_sizes`` order.

    weight[i] = group_weight[g] / group_size[g] for the group g that index i
    belongs to → each group's summed mass == group_weight[g].
    """
    parts = []
    for size, w in zip(group_sizes, group_weights):
        if size == 0:
            continue
        parts.append(np.full(size, float(w) / float(size), dtype=np.float64))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)
