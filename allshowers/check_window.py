"""
For each shower, get the unique layers, sort them, and check if any
consecutive unique layers have a gap > window (num_layer_cond//2 on each side).

Usage:
python /n/home04/hhanif/AllShowers/allshowers/check_window.py  /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_processed_step1_v3/merged_all_showers.h5 --num_layer_cond 8
"""

import argparse

import showerdata
import torch


def check_layer_windows(path: str, num_layer_cond: int = 4, stop: int | None = None):
    half = num_layer_cond // 2

    print(f"Loading data from {path} (stop={stop})")
    showers = showerdata.load(path, 0, stop)
    if showers.points.shape[2] == 5:
        showers.points = showers.points[:, :, :4]

    points = torch.from_numpy(showers.points)   # (N, max_hits, 4)
    mask   = points[:, :, 3] > 0                # (N, max_hits)
    layer  = (points[:, :, 2] + 0.1).long()     # (N, max_hits)

    N = points.shape[0]
    print(f"Loaded {N:,} showers. Checking window=±{half} (num_layer_cond={num_layer_cond})...\n")

    broken_showers = []

    for i in range(N):
        valid_layers = layer[i][mask[i]]         # only real hits
        if valid_layers.numel() == 0:
            continue

        unique_layers = valid_layers.unique(sorted=True).tolist()  # sorted unique layers

        # check each unique layer: does it have any neighbor within ±half?
        broken_layers = []
        for l in unique_layers:
            has_neighbor = any(
                abs(l - other) <= half and other != l
                for other in unique_layers
            )
            if not has_neighbor:
                broken_layers.append(l)

        if broken_layers:
            broken_showers.append({
                "idx": i,
                "total_hits": valid_layers.numel(),
                "unique_layers": unique_layers,
                "broken_layers": broken_layers,
            })

    print(f"Showers with broken window: {len(broken_showers)} / {N:,}")
    print()
    for s in broken_showers:
        print(f"Shower {s['idx']:>8,}  |  total hits: {s['total_hits']:>6,}  |  unique layers: {s['unique_layers']}  |  isolated layers: {s['broken_layers']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--num_layer_cond", type=int, default=4)
    parser.add_argument("--stop", type=int, default=None)
    args = parser.parse_args()

    check_layer_windows(args.path, args.num_layer_cond, args.stop)