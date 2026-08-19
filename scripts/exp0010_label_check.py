#!/usr/bin/env python3
"""EXP-0010 data-level proof that the is_human label is CORRECT, not just present.

Instantiates the real EgoPiPrqCachedDatasetV4 sources from the embodiments JSON (cached
DINO feats -> no GPU, no video decode), checks every source's per-sample label against
its declared ``mode``, then runs the real MultiEmbActionFramesCollator over a mixed
feature list and reports the stacked robot/human counts.

Usage: python label_check.py <embodiments.json> [n_samples_per_source]
"""
import json
import sys

import torch

import gr00t.experiment.data_config_v3  # noqa: F401  (register extra configs)
from gr00t.data.dataset_action_frames_v4_multiemb import (
    EmbodimentTaggedDataset,
    MultiEmbActionFramesCollator,
)
from gr00t.data.dataset_egopi_prq_v4 import EgoPiPrqCachedDatasetV4

cfg_path = sys.argv[1]
n_take = int(sys.argv[2]) if len(sys.argv) > 2 else 3
cfg = json.load(open(cfg_path))
g = cfg["embodiments"][0]
name = g["name"]
print(f"[config] {cfg_path}\n[group] {name} sources={len(g['sources'])}")

feats, ok = [], True
for src in g["sources"]:
    ds = EgoPiPrqCachedDatasetV4(
        prq_mode=src["mode"],
        prq_stats_path=g["prq_stats"],
        fk_cache_h5=src.get("fk_cache"),
        filter_json=g.get("filter"),
        filter_tag=src.get("filter_tag"),
        dataset_path=src["dataset_path"],
        data_config_name=src["data_config"],
        embodiment_tag=g.get("embodiment_tag", "new_embodiment"),
        split="train",
        val_ratio=0.003,
        val_seed=42,
        normalization_mode="min_max",
        image_size=224,
        feature_source="dino",
        dino_model="facebook/dinov2-large",
        dino_final_norm="naive",
        use_fixed_val=True,
        fixed_val_path=None,
        video_backend="decord",
    )
    tagged = EmbodimentTaggedDataset(ds, name)
    expect = 1.0 if src["mode"] == "human" else 0.0
    got = set()
    for i in range(min(n_take, len(tagged))):
        it = tagged[i * max(1, len(tagged) // max(n_take, 1))]
        assert "is_human" in it, "dataset item is missing the is_human key"
        got.add(float(it["is_human"]))
        feats.append(it)
    good = got == {expect}
    ok &= good
    tail = src["dataset_path"].rsplit("/", 1)[-1]
    print(f"  [{'PASS' if good else 'FAIL'}] mode={src['mode']:<5} {tail:<30} "
          f"len={len(tagged):>7,}  is_human={sorted(got)} (expected {expect})")

print("\n[collator] MultiEmbActionFramesCollator(pass_is_human=True) over the mixed list")
out = MultiEmbActionFramesCollator(pass_is_human=True)([dict(f) for f in feats])
gr = out["groups"][name]
lbl = gr["is_human"]
n_h = int((lbl > 0.5).sum())
exp_h = sum(1 for s in g["sources"] for _ in range(n_take) if s["mode"] == "human")
c_ok = lbl.shape[0] == len(feats) and n_h == exp_h
ok &= c_ok
print(f"  action={tuple(gr['action'].shape)} x0_feat={tuple(gr['x0_feat'].shape)} "
      f"is_human={tuple(lbl.shape)} dtype={lbl.dtype}")
print(f"  [{'PASS' if c_ok else 'FAIL'}] stacked counts: human={n_h} (expected {exp_h}), "
      f"robot={len(feats) - n_h} (expected {len(feats) - exp_h})")

print("\n[collator] default MultiEmbActionFramesCollator() must NOT emit is_human")
gr0 = MultiEmbActionFramesCollator()([dict(f) for f in feats])["groups"][name]
d_ok = "is_human" not in gr0 and set(gr0) == {"action", "x0_feat", "x1_feat"}
ok &= d_ok
print(f"  [{'PASS' if d_ok else 'FAIL'}] keys={sorted(gr0)}")

print(f"\n{'LABEL CHECK PASSED' if ok else 'LABEL CHECK FAILED'}")
sys.exit(0 if ok else 1)
