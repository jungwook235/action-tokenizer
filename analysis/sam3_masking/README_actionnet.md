# ActionNet SAM3 masking — how to run it on another server

Produces, for every episode of `gr1_actionnet_lerobot_15fps`, a cutout video, an
overlay video, and a mask npz in which the **robot masks and the task-object masks
are stored separately**.

## 1. Files to copy

Everything lives in `analysis/sam3_masking/`. The run needs exactly these:

| file | why |
|---|---|
| `batch_sam3_actionnet.py` | entry point |
| `sam3_batch_core.py` | SAM3 session, cutout/overlay, npz writing |
| `run_sam3.py` | `load_video` / `write_video` / `PALETTE` |
| `actionnet_task_nouns.json` | task string -> object prompts (1530 instructions) |

Optional:

| file | why |
|---|---|
| `build_actionnet_task_nouns.py` | regenerate the noun table (edit VOCAB/ALIASES, re-run) |
| `probe_actionnet_nouns.py` | screen new candidate prompts on a few frames (~1s each) |
| `compare_prompt_combos_actionnet.py` | compare whole prompt combos on sample episodes |
| `requirements_sam3.txt`, `setup_sam3_env.sh`, `sam3_env.sh` | environment bootstrap |

## 2. Environment

`transformers>=5.x` is mandatory — `Sam3VideoModel` does not exist in 4.x. Also
needs torch (CUDA), `opencv-python-headless`, and `ffmpeg` on PATH (the mp4s get
re-encoded to h264 after writing).

Bare machine:

```bash
./setup_sam3_env.sh --venv-only --skip-sam31    # venv + deps, no SAM 3.1 extras
```

Weights: the scripts pass the hub id **`jetjodh/sam3`** — an ungated mirror, because
`facebook/sam3` is gated and our HF account has no access. Set `HF_HOME` to a
persistent path so the ~3 GB download is cached, or pre-download it there.

Sanity check before launching:

```bash
python -c "import torch, transformers; from transformers import Sam3VideoModel; \
print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
```

## 3. Run

```bash
python batch_sam3_actionnet.py \
    --root    /path/to/ActionNet/gr1_actionnet_lerobot_15fps \
    --out-root /path/to/output \
    --num-shards 24 --shard-id $SHARD          # one process per GPU
```

`--root` defaults to this cluster's path, so pass it explicitly elsewhere.

Smoke test first (3 episodes, ~4 min, covers a normal episode, one where the idle
arm is easy to lose, and a cabinet-door episode with no manipulable object):

```bash
python batch_sam3_actionnet.py --root ... --out-root /tmp/smoke \
    --episodes 8084 15316 19262
```

Useful flags: `--limit N` (first N of the shard), `--max-nouns` (default 3),
`--cutout-from {union,robot,object}`, `--no-prompt-masks` (drops the per-prompt
planes to save disk), `--bg {green,black,white}`, `--fps` (override; default is
`meta/info.json`, i.e. 15.0), `--nouns` (a different noun table).

**Sharding / resume.** Episode j goes to shard `j % num_shards`; a shard skips any
episode whose three outputs already exist. So a killed job is restarted by
re-submitting the same command, and the shard count may be changed between runs
without redoing finished work. Per-episode failures are logged to the manifest and
the shard keeps going.

## 4. Prompts

Per episode, in ONE propagation:

```
["robot hand", "robot arm"] + actionnet_task_nouns.json[task][:3]
```

Chosen from a 6-episode × 6-combo probe:

* `["robot hand","robot arm"]` covers both arms from frame 0 in 6/6 episodes.
  `"Robot"` alone silently drops the *idle* arm in 2/6 and starts 15–24 frames late.
* The task instruction as one prompt is dead here (0 detections in 5/6 episodes),
  but its **nouns** detect in ~100% of frames — e.g. ep008084 sentence 0/241 vs
  `blue cup` 241/241 + `bowl` 241/241.
* Colour adjectives are kept (`white basket`, `orange cup`): SAM3 grounds
  colour+noun far better than the bare noun on this real footage.

Known limits: bare `container` is weak (the annotator's word often does not match
what is on screen — `white container` scores far better); the dark roll-top cabinet
in ~17 episodes grounds to nothing; small objects drop out while occluded mid-grasp
(irrelevant after union with the robot mask).

## 5. Output

```
<out_root>/
    cutout/chunk-XXX/observation.images.ego_view/episode_XXXXXX.mp4
    overlay/chunk-XXX/observation.images.ego_view/episode_XXXXXX.mp4
    masks/chunk-XXX/observation.images.ego_view/episode_XXXXXX.npz
    manifest_shard{K}.jsonl
```

npz — uint8 0/1, **frame t == step t of the episode parquet** (verified: mp4 frame
count == `episodes.jsonl` length):

| key | shape | |
|---|---|---|
| `mask` | (T,H,W) | union of all prompts |
| `robot_mask` | (T,H,W) | `robot hand` ∪ `robot arm` |
| `object_mask` | (T,H,W) | task nouns ∪ |
| `prompt_masks` | (P,T,H,W) | one plane per prompt |
| `prompts` | (P,) | prompt strings |
| `prompt_roles` | (P,) | `"robot"` / `"object"` |
| `n_robot_prompts` | () | `prompt_masks[:n]` robot, `[n:]` objects |
| `episode_index`, `fps` | () | fps is 15.0, from `meta/info.json` |

```python
d = np.load(path)
robot, obj = d["robot_mask"], d["object_mask"]      # (T,H,W) each
assert (robot | obj == d["mask"]).all()
```

## 6. Cost

0.27 s/frame on an A100 80GB → 7.71M frames ≈ **578 GPU-hours**.

| 1-GPU jobs | wall clock |
|---|---|
| 16 | ~36 h |
| 24 | ~24 h |
| 48 | ~12 h |

Disk ≈ 1.0 MB/episode for the npz plus the two mp4s → roughly 30–50 GB total.
Give each job ≥4 CPUs: the mp4 encoding is on the CPU and is part of that 0.27 s.
