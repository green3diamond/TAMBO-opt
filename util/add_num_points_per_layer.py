#!/usr/bin/env python3

'''
python /n/home04/hhanif/AllShowers/allshowers/add_num_points_per_layer.py \
  --input /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_processed_step1_v5/merged_all_showers_test.h5 \
  --num-layers 24

python /n/home04/hhanif/AllShowers/allshowers/add_num_points_per_layer.py \
  --input /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/combined_electrons_balanced-test-file.h5  \
  --num-layers 24 --with-time
'''

import argparse
import os
import h5py
import numpy as np


# -------------------------------------------------
# num_points per layer (e > 0 mask, direct from h5py)
# -------------------------------------------------
def calc_num_points_per_layer_h5(
    h5_dataset,
    start: int,
    stop: int,
    num_layers: int,
    num_cols: int = 4,
) -> np.ndarray:
    """
    Count number of hits per layer for each shower using e > 0 mask.
    Reads directly from h5py dataset (variable-length flat arrays).

    layer index = (z + 0.1).astype(int32)

    Args:
        h5_dataset: h5py dataset of raw showers.
        start: First shower index (inclusive).
        stop: Last shower index (exclusive).
        num_layers: Number of calorimeter layers.
        num_cols: Number of point features — 4 (x,y,z,e) or 5 (x,y,z,e,t).
    """
    num_showers = stop - start
    points_per_layer = np.zeros((num_showers, num_layers), dtype=np.int32)

    for i, global_i in enumerate(range(start, stop)):
        shower = np.array(h5_dataset[global_i])
        points = shower.reshape(-1, num_cols)           # (n_hits, 4 or 5)
        layer_idx = np.clip(
            (points[:, 2] + 0.1).astype(np.int32), 0, num_layers - 1
        )
        mask = (points[:, 3] > 0).astype(np.int32)     # e > 0  (col 3 in both modes)
        np.add.at(points_per_layer[i], layer_idx, mask)

    return points_per_layer


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Copy HDF5 file (excluding 'showers') and add num_points_per_layer_corsika dataset"
    )
    parser.add_argument(
        "--input",
        default="/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_processed_step1_v3/merged_all_showers.h5",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output HDF5 file path. Defaults to <input_stem>_data_with_num_points.h5 in the same directory.",
    )
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--with-time",
        action="store_true",
        default=False,
        help=(
            "Treat shower data as 5-column format (x, y, z, e, t). "
            "Without this flag, the original 4-column format (x, y, z, e) is used."
        ),
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(args.input)

    # Number of columns per hit point
    num_cols = 5 if args.with_time else 4
    print(f"Mode: {'with time (x,y,z,e,t) — 5 cols' if args.with_time else 'original (x,y,z,e) — 4 cols'}")

    # Default output path
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_data_with_num_points{ext}"

    if os.path.isfile(args.output):
        if not args.overwrite:
            print(f"Output file '{args.output}' already exists. Use --overwrite to recompute.")
            return
        else:
            print(f"Output file '{args.output}' already exists — deleting and recomputing.")
            os.remove(args.output)

    DATASET_NAME = "num_points_per_layer_corsika"
    SKIP_DATASETS = {"showers", DATASET_NAME}  # skip showers + the dataset we're about to recompute

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")

    with h5py.File(args.input, "r") as h_in, h5py.File(args.output, "w") as h_out:

        N = h_in["showers"].shape[0]

        # -------------------------------------------------
        # Copy all datasets except 'showers'
        # -------------------------------------------------
        print("\nCopying datasets (excluding 'showers')...")
        for name in h_in:
            if name in SKIP_DATASETS:
                print(f"  - Skipping '{name}'")
                continue
            item = h_in[name]
            if isinstance(item, h5py.Dataset):
                print(f"  - Copying '{name}': {item.shape}")
            else:
                print(f"  - Copying '{name}' (group)")
            h_in.copy(name, h_out)

        # Copy root-level attributes, if any
        for key, val in h_in.attrs.items():
            h_out.attrs[key] = val

        # -------------------------------------------------
        # Create new dataset
        # -------------------------------------------------
        print(f"\nCreating '{DATASET_NAME}' with shape ({N}, {args.num_layers})...")
        d_np = h_out.create_dataset(
            DATASET_NAME,
            shape=(N, args.num_layers),
            dtype=np.int32,
            chunks=(min(args.chunk_size, N), args.num_layers),
            compression="gzip",
            shuffle=True,
        )

        # -------------------------------------------------
        # Compute in chunks (reads from input file)
        # -------------------------------------------------
        step = args.chunk_size
        for start in range(0, N, step):
            stop = min(N, start + step)

            np_chunk = calc_num_points_per_layer_h5(
                h_in["showers"],
                start=start,
                stop=stop,
                num_layers=args.num_layers,
                num_cols=num_cols,          # 4 or 5 depending on --with-time
            )
            d_np[start:stop] = np_chunk

            if stop % (step * 10) == 0 or stop == N:
                print(f"  Processed {stop}/{N}")

    print("\nDone.")
    print(f"Output file: {args.output}")
    with h5py.File(args.output, "r") as h:
        print("Datasets inside:")
        for name in h:
            print(f"  - {name}: {h[name].shape}")


if __name__ == "__main__":
    main()