#!/usr/bin/env python3

import os
import h5py
import numpy as np
import showerdata

num_layer_value = 24

input_files_list = [
        # "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/combined_electrons_balanced.h5",
        "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/combined_electrons_balanced-test-file.h5",
        "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/combined_muons_balanced.h5",
        "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/combined_muons_balanced-test-file.h5",
        "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/combined_photons_balanced.h5",
        "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/combined_photons_balanced-test-file.h5",
    ]

# -------------------------------------------------
# Label utilities
# -------------------------------------------------
def create_label_list_numpy(pdg_1d: np.ndarray) -> list[int]:
    unique = np.unique(pdg_1d).tolist()
    unique.sort(key=lambda x: (abs(int(x)), -int(x)))
    return [int(x) for x in unique]


def pdg_to_label_numpy(pdg_1d: np.ndarray, label_list: list[int]) -> np.ndarray:
    label_map = {pdg_val: i for i, pdg_val in enumerate(label_list)}
    return np.fromiter(
        (label_map[int(x)] for x in pdg_1d),
        dtype=np.int32,
        count=pdg_1d.size,
    )


# -------------------------------------------------
# num_points per layer (e > 0 mask, direct from h5py)
# -------------------------------------------------
def calc_num_points_per_layer_h5(h5_dataset, start: int, stop: int, num_layers: int) -> np.ndarray:
    num_showers = stop - start
    points_per_layer = np.zeros((num_showers, num_layers), dtype=np.int32)

    for i, global_i in enumerate(range(start, stop)):
        shower = np.array(h5_dataset[global_i])
        points = shower.reshape(-1, 5)
        layer_idx = np.clip(
            (points[:, 2] + 0.1).astype(np.int32), 0, num_layers - 1
        )
        mask = (points[:, 3] > 0).astype(np.int32)
        np.add.at(points_per_layer[i], layer_idx, mask)

    return points_per_layer


# -------------------------------------------------
# Auto-generate output path from input path
# -------------------------------------------------
def make_output_path(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    return f"{base}_for_pointcloud{ext}"


# -------------------------------------------------
# Process a single input file → output file
# -------------------------------------------------
def process_file(input_path: str, output_path: str, num_layers: int, chunk_size: int, overwrite: bool):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)

    if os.path.exists(output_path):
        if not overwrite:
            raise FileExistsError(f"Output exists: {output_path}. Set OVERWRITE = True to overwrite.")
        os.remove(output_path)

    print(f"\n{'='*60}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"{'='*60}")

    with h5py.File(input_path, "r") as hin:
        N = hin["pdg"].shape[0]
        pdg_all = hin["pdg"][:].astype(np.int32)

    label_list = create_label_list_numpy(pdg_all)

    with h5py.File(output_path, "w") as hout, h5py.File(input_path, "r") as hin:

        d_dir = hout.create_dataset(
            "directions",
            shape=hin["directions"].shape,
            dtype=hin["directions"].dtype,
            chunks=True,
            compression="gzip",
            shuffle=True,
        )

        d_en = hout.create_dataset(
            "energies",
            shape=hin["energies"].shape,
            dtype=hin["energies"].dtype,
            chunks=True,
            compression="gzip",
            shuffle=True,
        )

        d_lab = hout.create_dataset(
            "labels",
            shape=(N,),
            dtype=np.int32,
            chunks=True,
            compression="gzip",
            shuffle=True,
        )

        d_np = hout.create_dataset(
            "num_points",
            shape=(N, num_layers),
            dtype=np.int32,
            chunks=(min(chunk_size, N), num_layers),
            compression="gzip",
            shuffle=True,
        )

        hout.attrs["label_list"] = np.array(label_list, dtype=np.int32)
        hout.attrs["num_layers"] = np.int32(num_layers)

        for start in range(0, N, chunk_size):
            stop = min(N, start + chunk_size)

            d_dir[start:stop] = hin["directions"][start:stop]
            d_en[start:stop]  = hin["energies"][start:stop]

            d_lab[start:stop] = pdg_to_label_numpy(pdg_all[start:stop], label_list)

            d_np[start:stop] = calc_num_points_per_layer_h5(
                hin["showers"], start=start, stop=stop, num_layers=num_layers
            )

            if stop % (chunk_size * 10) == 0 or stop == N:
                print(f"  Processed {stop}/{N}")

    print(f"Done: {output_path}")
    with h5py.File(output_path, "r") as hout:
        print("  Datasets inside:")
        for name in ["directions", "energies", "labels", "num_points"]:
            print(f"    - {name}: {hout[name].shape}")


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    INPUT_FILES = input_files_list
    NUM_LAYERS = num_layer_value
    CHUNK_SIZE = 5000
    OVERWRITE  = False

    for input_path in INPUT_FILES:
        output_path = make_output_path(input_path)
        process_file(
            input_path=input_path,
            output_path=output_path,
            num_layers=NUM_LAYERS,
            chunk_size=CHUNK_SIZE,
            overwrite=OVERWRITE,
        )

    print(f"\nAll done. Processed {len(INPUT_FILES)} file(s).")


if __name__ == "__main__":
    main()