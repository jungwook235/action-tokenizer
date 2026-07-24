"""Build a balanced val-chunk cache for EgoDex (egodex_lerobot_part1_gr1, lerobot)
encoded with the JOINT multi-embodiment v4 tokenizer's egodex_gr1 class token.

EgoDex is the HIGH-DoF endpoint (44-dim gr1-retargeted joints). There is no v3 for
egodex (all egodex tokenizers are joint-v4), so the cache stores A + Z4(mu) + task +
fold only (Z3 omitted; the probe compares raw / PCA-k / v4).

Registers a minimal runtime data-config (no file edits): action.joints[0:44] + video.egoview,
min_max norm. Samples the TOP --max-tasks tasks by #episodes (EgoDex has ~150 eps/task so
no starvation), max_eps episodes/task, chunks_per_ep evenly-spaced chunks; assigns LOEPTO
folds (episode-rank % n_folds) for a leak-free by-episode intent split.

DECODE-L1 GATE: after encoding, decode Z4(mu) -> a_hat and report mean L1 vs the raw
(min_max) action. Reference: gr1 v4 decode-L1 ~= 0.0014. If egodex L1 is within a few x
of that the egodex_gr1 encoder fits the lerobot joints; if orders higher, the action repr
mismatches training and the v4 latents are NOT trustworthy (abort/flag).

Usage (srun --partition=debug --gpus=1):
  python collect_cache_egodex.py --dataset-path <egodex_lerobot> \
     --v4-ckpt <joint_egodex_dexdual/checkpoint-250000> --embodiment-id egodex_gr1 \
     --max-tasks 80 --min-eps 3 --max-eps 6 --chunks-per-ep 16 --out output/actlat_egodex/cache.npz
"""
import sys, argparse, json, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))
import gr00t.experiment.data_config_v3  # noqa: F401,E402
from gr00t.data.dataset import ModalityConfig  # noqa: E402
from gr00t.experiment.data_config import DATA_CONFIG_MAP  # noqa: E402
from gr00t.data.dataset_action_frames_v4 import ActionFramesCollatorV4, ActionFramesDatasetV4  # noqa: E402
from gr00t.data.merge_norm_stats import apply_merged_normalization_metadata  # noqa: E402
from gr00t.model.action_latent_tokenizer_wrapper import ActionLatentTokenizerWrapper  # noqa: E402
from analyze_latents import _encode_mu_and_sample  # noqa: E402
from collect_cache_taskep import episode_task_map  # noqa: E402


class _EgodexJointsConfig:
    action_normalization_modes = {"action.joints": "min_max"}
    def modality_config(self):
        return {
            "action": ModalityConfig(delta_indices=list(range(16)), modality_keys=["action.joints"]),
            "video": ModalityConfig(delta_indices=[0], modality_keys=["video.egoview"]),
        }


def select_top_tasks(ds, ep2task, max_tasks, min_eps, max_eps, chunks_per_ep, n_folds, seed):
    tids = np.asarray(ds._trajectory_ids, dtype=np.int64)
    lens = np.asarray(ds._trajectory_lengths, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lens)])
    by_task = {}
    for pos, eid in enumerate(tids.tolist()):
        t = ep2task.get(int(eid))
        if t is None:
            continue
        by_task.setdefault(t, []).append(pos)
    # rank tasks by #episodes desc, keep top max_tasks with >= min_eps
    ranked = sorted([(t, ps) for t, ps in by_task.items() if len(ps) >= min_eps],
                    key=lambda kv: -len(kv[1]))[:max_tasks]
    flat, task, epidx, fold = [], [], [], []
    for t, positions in ranked:
        chosen = positions[:max_eps]
        for rank, pos in enumerate(chosen):
            lo, hi = int(offsets[pos]), int(offsets[pos + 1])
            k = min(chunks_per_ep, hi - lo)
            if k <= 0:
                continue
            sel = np.unique(np.linspace(lo, hi - 1, k).round().astype(int))
            flat += sel.tolist(); task += [t] * len(sel)
            epidx += [int(tids[pos])] * len(sel); fold += [rank % n_folds] * len(sel)
    print(f"[select] kept {len(ranked)} tasks (top by #eps, >= {min_eps}); {len(flat)} chunks", flush=True)
    return (np.array(flat, np.int64), np.array(task, np.int64),
            np.array(epidx, np.int64), np.array(fold, np.int64))


@torch.no_grad()
def encode_v4_egodex(ckpt, embodiment_id, loader, device):
    wrap = ActionLatentTokenizerWrapper.from_checkpoint(ckpt, device=device, embodiment_id=embodiment_id)
    wrap.eval()
    assert wrap._is_v4(), "expected a v4 tokenizer"
    enc = wrap.tokenizer.encoder
    dtype = enc.action_proj.weight.dtype
    A, Z, Vc, Vd = [], [], [], []
    l1_sum, n_seen = 0.0, 0
    for b in loader:
        a = b["action"].to(device)
        f0, f1 = wrap._resolve_dino_feats(b["frame_x0"], b["frame_x1"], None, None, device)
        mu, _s, _lv, _z = _encode_mu_and_sample(enc, a, f0, f1)
        zero_g = mu[:, :0]
        a_hat = wrap.tokenizer.decode(zero_g, mu.to(dtype), zero_g)
        l1_sum += torch.nn.functional.l1_loss(a_hat.float(), a.float()).item() * a.shape[0]
        n_seen += a.shape[0]
        A.append(a.float().cpu()); Z.append(mu.float().cpu())
        f0m, f1m = f0.float().mean(1).cpu(), f1.float().mean(1).cpu()
        Vc.append(torch.cat([f0m, f1m], 1)); Vd.append((f1.float() - f0.float()).mean(1).cpu())
    decode_l1 = l1_sum / max(1, n_seen)
    out = (torch.cat(A).numpy(), torch.cat(Z).numpy(), torch.cat(Vc).numpy(),
           torch.cat(Vd).numpy(), float(decode_l1))
    del wrap
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--v4-ckpt", required=True)
    ap.add_argument("--embodiment-id", default="egodex_gr1")
    ap.add_argument("--data-config", default="egodex_joints_front")
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--normalization-mode", default="min_max")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--video-backend", default="decord")
    ap.add_argument("--max-tasks", type=int, default=80)
    ap.add_argument("--min-eps", type=int, default=3)
    ap.add_argument("--max-eps", type=int, default=6)
    ap.add_argument("--chunks-per-ep", type=int, default=16)
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--gate-l1", type=float, default=0.0072, help="abort if decode-L1 exceeds (5x gr1 0.0014)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    DATA_CONFIG_MAP[args.data_config] = _EgodexJointsConfig()
    ep2task, ep2name, t2i = episode_task_map(args.dataset_path)
    print(f"[collect] building split=all egodex dataset ...", flush=True)
    ds = ActionFramesDatasetV4(
        dataset_path=args.dataset_path, data_config_name=args.data_config,
        embodiment_tag=args.embodiment_tag, split="all",
        normalization_mode=args.normalization_mode, image_size=args.image_size,
        video_backend=args.video_backend, use_fixed_val=False)
    apply_merged_normalization_metadata([ds], [ds])

    flat, task, epidx, fold = select_top_tasks(
        ds, ep2task, args.max_tasks, args.min_eps, args.max_eps,
        args.chunks_per_ep, args.n_folds, args.sample_seed)
    uniq = np.unique(task)
    print(f"[collect] {len(flat)} chunks, {len(uniq)} tasks, {len(np.unique(epidx))} eps, "
          f"folds={np.bincount(fold).tolist()}", flush=True)
    if args.dry_run:
        print("#### DRY RUN OK ####"); return

    subset = torch.utils.data.Subset(ds, flat.tolist())
    loader = torch.utils.data.DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers, collate_fn=ActionFramesCollatorV4())
    print(f"[collect] encoding v4 (embodiment_id={args.embodiment_id}) ...", flush=True)
    t0 = time.time()
    A, Z4, Vc, Vd, decode_l1 = encode_v4_egodex(args.v4_ckpt, args.embodiment_id, loader, device)
    print(f"[collect]   A{A.shape} Z4{Z4.shape} in {time.time()-t0:.0f}s", flush=True)
    print(f"#### DECODE-L1 GATE: egodex={decode_l1:.5f}  gr1_ref=0.00140  ratio={decode_l1/0.0014:.1f}x  "
          f"threshold={args.gate_l1:.4f} -> {'PASS' if decode_l1 <= args.gate_l1 else 'FAIL(repr-mismatch)'} ####", flush=True)

    id_map = {int(t): i for i, t in enumerate(uniq.tolist())}
    task_compact = np.array([id_map[int(t)] for t in task], np.int64)
    global_ep = task_compact.astype(np.int64) * 1_000_000 + epidx
    # per-task episode counts (for top-k task subsetting in the probe)
    ep_per_task = {int(id_map[int(t)]): len(set(epidx[task == t].tolist())) for t in uniq}

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(dataset_path=args.dataset_path, data_config=args.data_config,
                v4_ckpt=args.v4_ckpt, embodiment_id=args.embodiment_id, N=int(len(flat)),
                T=int(A.shape[1]), D=int(A.shape[2]), K4=int(Z4.shape[-1]),
                n_tasks=int(len(uniq)), decode_l1=decode_l1, gate_l1=args.gate_l1,
                gate_pass=bool(decode_l1 <= args.gate_l1), gr1_ref_l1=0.0014,
                max_tasks=args.max_tasks, min_eps=args.min_eps, max_eps=args.max_eps,
                chunks_per_ep=args.chunks_per_ep, n_folds=args.n_folds,
                ep_per_task=ep_per_task,
                task_names=[str(k) for k, v in sorted(t2i.items(), key=lambda kv: kv[1])])
    # Z3 stored as zeros placeholder so intent_probe (which reads Z3) still loads; probe skips v3.
    np.savez_compressed(out, A=A, Z3=np.zeros((len(flat), A.shape[1], 1), np.float32), Z4=Z4,
                        Vcontext=Vc, Vdyn=Vd, task=task_compact, task_orig=task,
                        episode_index=epidx, episode_id=global_ep, fold_loepto=fold,
                        samp_flat=flat, meta=json.dumps(meta))
    print(f"[collect] wrote {out} ({out.stat().st_size/1e6:.1f} MB)", flush=True)
    print("#### COLLECT DONE ####")


if __name__ == "__main__":
    main()
