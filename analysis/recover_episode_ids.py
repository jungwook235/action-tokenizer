"""Recover a per-sample EPISODE id for the shared vsep cache, so downstream
probes can use a leak-free by-episode (grouped) train/val split.

The cache (vsep_collect.py) stores, per sample i:
    samp_task[i]   = task index (which dataset)
    samp_local[i]  = flat index into that task's val ActionFramesDatasetV4
but NOT the episode. The dataset's flat index maps deterministically to
(episode, base_index) via `all_steps`:

    trajectory_ids, trajectory_lengths = _get_trajectories()   # val episodes, sorted asc
    all_steps = [(eid, b) for eid,len in zip(ids,lengths) for b in range(len)]
    episode_of(local) = all_steps[local][0]

`get_fixed_split_for_split` returns the val episodes sorted ascending by their
position in episodes.jsonl (== episode_index order), and `_get_all_steps`
iterates the RAW episode length. So we can rebuild `all_steps` from metadata
only (episodes.jsonl + meta/fixed_val_split.json) — no torch, no GPU, no
dataset instantiation. We assert max(samp_local within task) < sum(val_lengths)
as a correctness check.

Output: <cache_dir>/episode_ids.npz with
    episode_id       [N] int64  global unique episode id (task*1e6 + local_episode_idx)
    episode_index    [N] int64  the dataset's own episode_index (per task)
    fold_loepto      [N] int64  leave-one-episode-per-task-out fold (0..n_ep_per_task-1)
    meta (json)

Usage:
    python recover_episode_ids.py --cache output/visual_sep_gr1/cache.npz
"""
import argparse, json
from pathlib import Path
import numpy as np

EP_FILE = "meta/episodes.jsonl"
FV_FILE = "meta/fixed_val_split.json"


def val_all_steps(dataset_path: Path):
    """Reconstruct (episode_ids_per_step, base_index_per_step) for the val split
    of one task, matching ActionFramesDatasetV4._get_all_steps exactly."""
    dp = Path(dataset_path)
    meta = [json.loads(l) for l in open(dp / EP_FILE)]
    all_ids = np.array([e["episode_index"] for e in meta], dtype=np.int64)
    all_len = np.array([e["length"] for e in meta], dtype=np.int64)
    fv = json.load(open(dp / FV_FILE))
    val_ids = [int(e) for e in fv["val_episode_ids"]]
    id_to_idx = {int(eid): i for i, eid in enumerate(all_ids.tolist())}
    sel = np.array([id_to_idx[e] for e in val_ids], dtype=np.int64)
    sel.sort()  # matches get_fixed_split_for_split ("v2 deterministic ordering")
    ids, lens = all_ids[sel], all_len[sel]
    step_eid, step_rank = [], []
    for rank, (eid, L) in enumerate(zip(ids.tolist(), lens.tolist())):
        step_eid += [eid] * L
        step_rank += [rank] * L
    return np.array(step_eid, np.int64), np.array(step_rank, np.int64), ids, lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cache = Path(args.cache)
    out = Path(args.out) if args.out else cache.parent / "episode_ids.npz"

    d = np.load(cache, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    samp_task = d["samp_task"].astype(np.int64)
    samp_local = d["samp_local"].astype(np.int64)
    dpaths = meta["dataset_path"]
    N = len(samp_task)

    episode_index = np.full(N, -1, np.int64)
    episode_rank = np.full(N, -1, np.int64)  # 0..n_ep_per_task-1 within the task
    per_task_nep = {}
    for ti, dp in enumerate(dpaths):
        step_eid, step_rank, ids, lens = val_all_steps(dp)
        per_task_nep[ti] = int(len(ids))
        m = samp_task == ti
        loc = samp_local[m]
        assert loc.max() < len(step_eid), (
            f"task {ti}: samp_local max {loc.max()} >= n_steps {len(step_eid)} "
            f"(val episodes {ids.tolist()} lengths {lens.tolist()})")
        episode_index[m] = step_eid[loc]
        episode_rank[m] = step_rank[loc]

    assert (episode_index >= 0).all() and (episode_rank >= 0).all()
    # global unique episode id + LOEPTO fold (rank within task = fold)
    global_ep = samp_task.astype(np.int64) * 1_000_000 + episode_index
    fold_loepto = episode_rank.copy()

    n_ep_per_task = np.array([per_task_nep[t] for t in range(len(dpaths))])
    info = {
        "cache": str(cache), "N": int(N),
        "n_tasks": len(dpaths),
        "n_unique_episodes": int(len(np.unique(global_ep))),
        "n_ep_per_task_min": int(n_ep_per_task.min()),
        "n_ep_per_task_max": int(n_ep_per_task.max()),
        "n_folds_loepto": int(n_ep_per_task.min()),
    }
    np.savez_compressed(out, episode_id=global_ep, episode_index=episode_index,
                        episode_rank=episode_rank, fold_loepto=fold_loepto,
                        meta=json.dumps(info))
    print(json.dumps(info, indent=2))
    print("per-task episode counts:", n_ep_per_task.tolist())
    print("WROTE", out)
    print("#### EPISODE IDS DONE ####")


if __name__ == "__main__":
    main()
