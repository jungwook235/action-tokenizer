"""Build a balanced val-chunk cache for a SINGLE lerobot dataset whose task label
lives per-episode (episodes.jsonl 'tasks' -> tasks.jsonl 'task_index'), e.g. robocasa.
Mirrors output/visual_sep_gr1/cache.npz (A, Z3, Z4mu, task) but adds a leak-free
episode fold, since we sample many episodes from ONE dataset.

Sampling: use split="all"; keep tasks with >= --min-eps episodes; take up to --max-eps
episodes per task; sample --chunks-per-ep evenly-spaced chunks per episode. Assign each
task's episodes to --n-folds LOEPTO folds (episode-rank % n_folds) for a leak-free
by-episode intent split.

Encodes v3 (action-only) and v4 (DINO-fused mu) with the given tokenizer ckpts.
Requires GPU for the v4 DINO+encoder forward.

Usage (inside an srun --gpus=1 session):
  python collect_cache_taskep.py \
     --dataset-path <dir> --data-config single_panda_gripper_front \
     --v3-ckpt <...> --v4-ckpt <...> \
     --min-eps 3 --max-eps 4 --chunks-per-ep 12 --out output/actlat_robocasa/cache.npz
"""
import sys, argparse, json, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))
import gr00t.experiment.data_config_v3  # noqa: F401,E402
from gr00t.data.dataset_action_frames_v4 import ActionFramesCollatorV4, ActionFramesDatasetV4  # noqa: E402
from gr00t.data.merge_norm_stats import apply_merged_normalization_metadata  # noqa: E402
from vsep_collect import encode_v3, encode_v4  # noqa: E402

EP_FILE = "meta/episodes.jsonl"
TASK_FILE = "meta/tasks.jsonl"


def episode_task_map(dataset_path):
    dp = Path(dataset_path)
    t2i = {}
    for l in open(dp / TASK_FILE):
        d = json.loads(l); t2i[d["task"]] = d["task_index"]
    ep2task, ep2name = {}, {}
    for l in open(dp / EP_FILE):
        e = json.loads(l)
        name = e["tasks"][0] if isinstance(e["tasks"], list) else e["tasks"]
        ep2task[int(e["episode_index"])] = t2i[name]
        ep2name[int(e["episode_index"])] = name
    return ep2task, ep2name, t2i


def select_indices(ds, ep2task, args):
    """Return arrays (flat_idx, task, episode_index, fold) for the balanced sample."""
    tids = np.asarray(ds._trajectory_ids, dtype=np.int64)     # episode ids, in dataset order
    lens = np.asarray(ds._trajectory_lengths, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lens)])          # flat range per episode
    # group episodes (by dataset position) under their task
    by_task = {}
    for pos, eid in enumerate(tids.tolist()):
        t = ep2task.get(int(eid))
        if t is None:
            continue
        by_task.setdefault(t, []).append(pos)
    rng = np.random.default_rng(args.sample_seed)
    flat, task, epidx, fold = [], [], [], []
    kept_tasks = 0
    for t, positions in sorted(by_task.items()):
        if len(positions) < args.min_eps:
            continue
        kept_tasks += 1
        chosen = positions[:args.max_eps]                      # first max_eps episodes (deterministic)
        for rank, pos in enumerate(chosen):
            lo, hi = int(offsets[pos]), int(offsets[pos + 1])
            n = hi - lo
            k = min(args.chunks_per_ep, n)
            if k <= 0:
                continue
            sel = np.linspace(lo, hi - 1, k).round().astype(int)
            sel = np.unique(sel)
            flat += sel.tolist()
            task += [t] * len(sel)
            epidx += [int(tids[pos])] * len(sel)
            fold += [rank % args.n_folds] * len(sel)
    print(f"[select] kept {kept_tasks} tasks (>= {args.min_eps} eps); {len(flat)} chunks", flush=True)
    return (np.array(flat, np.int64), np.array(task, np.int64),
            np.array(epidx, np.int64), np.array(fold, np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--v3-ckpt", required=True)
    ap.add_argument("--v4-ckpt", required=True)
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--normalization-mode", default="min_max")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--video-backend", default="decord")
    ap.add_argument("--min-eps", type=int, default=3)
    ap.add_argument("--max-eps", type=int, default=4)
    ap.add_argument("--chunks-per-ep", type=int, default=12)
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true", help="build dataset + selection only, no encode")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    ep2task, ep2name, t2i = episode_task_map(args.dataset_path)
    print(f"[collect] building split=all dataset: {Path(args.dataset_path).name}", flush=True)
    ds = ActionFramesDatasetV4(
        dataset_path=args.dataset_path, data_config_name=args.data_config,
        embodiment_tag=args.embodiment_tag, split="all",
        normalization_mode=args.normalization_mode, image_size=args.image_size,
        video_backend=args.video_backend, use_fixed_val=False)
    apply_merged_normalization_metadata([ds], [ds])

    flat, task, epidx, fold = select_indices(ds, ep2task, args)
    uniq_tasks = np.unique(task)
    print(f"[collect] {len(flat)} chunks, {len(uniq_tasks)} tasks, "
          f"{len(np.unique(epidx))} episodes, folds={np.bincount(fold).tolist()}", flush=True)
    if args.dry_run:
        print("#### DRY RUN OK ####"); return

    subset = torch.utils.data.Subset(ds, flat.tolist())
    loader = torch.utils.data.DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers, collate_fn=ActionFramesCollatorV4())
    print("[collect] encoding v4 (+actions+DINO) ...", flush=True)
    t0 = time.time()
    A, Z4, Vc, Vd = encode_v4(args.v4_ckpt, loader, device)
    print(f"[collect]   A{A.shape} Z4{Z4.shape} in {time.time()-t0:.0f}s", flush=True)
    loader3 = torch.utils.data.DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.num_workers, collate_fn=ActionFramesCollatorV4())
    print("[collect] encoding v3 ...", flush=True)
    Z3 = encode_v3(args.v3_ckpt, loader3, device)
    assert Z3.shape[0] == A.shape[0] == len(flat)

    # remap task ids to a compact 0..C-1 range for the probe
    id_map = {int(t): i for i, t in enumerate(uniq_tasks.tolist())}
    task_compact = np.array([id_map[int(t)] for t in task], np.int64)
    global_ep = task_compact.astype(np.int64) * 1_000_000 + epidx  # unique per (task,episode)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(dataset_path=args.dataset_path, data_config=args.data_config,
                v3_ckpt=args.v3_ckpt, v4_ckpt=args.v4_ckpt, N=int(len(flat)),
                T=int(A.shape[1]), D=int(A.shape[2]), K3=int(Z3.shape[-1]), K4=int(Z4.shape[-1]),
                n_tasks=int(len(uniq_tasks)), embodiment_tag=args.embodiment_tag,
                normalization_mode=args.normalization_mode, min_eps=args.min_eps,
                max_eps=args.max_eps, chunks_per_ep=args.chunks_per_ep, n_folds=args.n_folds,
                sample_seed=args.sample_seed,
                task_names=[str(k) for k, v in sorted(t2i.items(), key=lambda kv: kv[1])])
    np.savez_compressed(out, A=A, Z3=Z3, Z4=Z4, Vcontext=Vc, Vdyn=Vd,
                        task=task_compact, task_orig=task, episode_index=epidx,
                        episode_id=global_ep, fold_loepto=fold, samp_flat=flat,
                        meta=json.dumps(meta))
    print(f"[collect] wrote {out} ({out.stat().st_size/1e6:.1f} MB)", flush=True)
    print("#### COLLECT DONE ####")


if __name__ == "__main__":
    main()
