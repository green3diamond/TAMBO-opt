"""
Remove showers that meet either condition:
  1. Only 1 unique layer (all hits in same layer)
  2. Any layer has no neighboring occupied layer within ±half

Usage:
    python filter_isolated_showers.py <input.h5> <output.h5> [--num_layer_cond 4]
"""

import argparse

import h5py
import numpy as np


def get_all_dataset_keys(f: h5py.File) -> list[str]:
    """Recursively get all dataset keys (not groups) in the file."""
    keys = []
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            keys.append(name)
    f.visititems(visitor)
    return keys


def find_bad_shower_indices(showers_obj: np.ndarray, num_layer_cond: int) -> np.ndarray:
    half = num_layer_cond // 2
    bad_indices = []

    print(f"Checking window=±{half} (num_layer_cond={num_layer_cond})...")

    for i, shower in enumerate(showers_obj):
        if i % 10000 == 0:
            print(f"  processed {i:,} / {len(showers_obj):,} showers, found {len(bad_indices)} bad so far")

        pts = shower.reshape(-1, 4)
        mask = pts[:, 3] > 0
        layers = (pts[:, 2] + 0.1).astype(int)
        valid_layers = layers[mask]
        total_hits = len(valid_layers)

        if total_hits == 0:
            print(f"  shower {i:>8,}  |  total hits:       0  |  unique layers: []  |  reason: empty")
            bad_indices.append(i)
            continue

        unique_layers = sorted(set(valid_layers.tolist()))

        # condition 1: only 1 unique layer
        if len(unique_layers) == 1:
            print(f"  shower {i:>8,}  |  total hits: {total_hits:>6,}  |  unique layers: {unique_layers}  |  reason: single unique layer")
            bad_indices.append(i)
            continue

        # condition 2: any layer has no neighbor within ±half
        isolated_layers = []
        for l in unique_layers:
            has_neighbor = any(
                0 < abs(l - other) <= half
                for other in unique_layers
            )
            if not has_neighbor:
                isolated_layers.append(l)

        if isolated_layers:
            print(f"  shower {i:>8,}  |  total hits: {total_hits:>6,}  |  unique layers: {unique_layers}  |  isolated layers: {isolated_layers}")
            bad_indices.append(i)

    return np.array(bad_indices, dtype=np.int64)


def filter_h5(input_path: str, output_path: str, num_layer_cond: int) -> None:
    with h5py.File(input_path, "r") as f_in:
        print(f"Opened {input_path}")
        showers_obj = f_in["showers"][:]
        N = len(showers_obj)
        print(f"Total showers: {N:,}\n")

        bad_idx = find_bad_shower_indices(showers_obj, num_layer_cond)
        print(f"\nFound {len(bad_idx)} showers to remove: indices {bad_idx.tolist()}")

        keep = np.ones(N, dtype=bool)
        keep[bad_idx] = False
        keep_idx = np.where(keep)[0]
        print(f"Keeping {keep.sum():,} showers\n")

        all_keys = get_all_dataset_keys(f_in)
        shape_keys = {"shape", "target/shape"}

        with h5py.File(output_path, "w") as f_out:
            for key in all_keys:
                if key in shape_keys:
                    continue
                data = f_in[key][:]
                filtered = data[keep_idx] if data.shape[0] == N else data
                # create parent groups if needed (e.g. "target/point_clouds")
                f_out.create_dataset(key, data=filtered)
                print(f"  wrote {key}: {filtered.shape}")

            # update shape metadata
            new_shape = f_in["shape"][:].copy()
            new_shape[0] = int(keep.sum())
            f_out.create_dataset("shape", data=new_shape)
            print(f"  wrote shape: {new_shape}")

            if "target/shape" in f_in:
                new_target_shape = f_in["target/shape"][:].copy()
                new_target_shape[0] = int(keep.sum())
                f_out.create_dataset("target/shape", data=new_target_shape)
                print(f"  wrote target/shape: {new_target_shape}")

    print(f"\nDone. Cleaned file saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input",  type=str, help="Input HDF5 file")
    parser.add_argument("output", type=str, help="Output HDF5 file")
    parser.add_argument("--num_layer_cond", type=int, default=6)
    args = parser.parse_args()

    filter_h5(args.input, args.output, args.num_layer_cond)