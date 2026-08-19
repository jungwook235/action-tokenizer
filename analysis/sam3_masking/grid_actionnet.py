"""Build a PNG contact sheet from the outputs_actionnet overlay videos.

One row per combo (label printed to stdout in row order), N frames sampled evenly.
Usage: python grid_actionnet.py <episode_index> [--n-frames 6]
"""
import argparse
import glob
import os

import imageio.v3 as iio
import numpy as np

from run_sam3 import load_video

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", type=int)
    ap.add_argument("--n-frames", type=int, default=6)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "outputs_actionnet"))
    args = ap.parse_args()

    vids = sorted(glob.glob(os.path.join(args.out_dir, f"ep{args.episode:06d}_*.mp4")))
    rows = []
    for v in vids:
        frames, _ = load_video(v)
        idx = np.linspace(0, len(frames) - 1, args.n_frames).astype(int)
        rows.append(np.concatenate([frames[i] for i in idx], 1))
        print(f"row {len(rows)}: {os.path.basename(v)}  frames {list(idx)}")
    out = os.path.join(args.out_dir, f"grid_ep{args.episode:06d}.png")
    iio.imwrite(out, np.concatenate(rows, 0))
    print(out)


if __name__ == "__main__":
    main()
