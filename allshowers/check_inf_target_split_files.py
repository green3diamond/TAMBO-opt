import h5py
import numpy as np
from pathlib import Path

BASE_DIR = Path(
    "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_ot_matched"
)

FILES = [
    # "merged_all_showers_part000.h5",
    # "merged_all_showers_part001.h5",
    # "merged_all_showers_part002.h5",
    # "merged_all_showers_part003.h5",
    # "merged_all_showers_part004.h5",
    # "merged_all_showers_part005.h5",
    # "merged_all_showers_part006.h5",
    # "merged_all_showers_part007.h5",
    # "merged_all_showers_part008.h5",
    # "merged_all_showers_part009.h5",
    # "merged_all_showers_part010.h5",
    # "merged_all_showers_part011.h5",
    # "merged_all_showers_part012.h5",
    # "merged_all_showers_part013.h5",
    # "merged_all_showers_part014.h5",
    # "merged_all_showers_part015.h5",
    # "merged_all_showers_part016.h5",
    # "merged_all_showers_part017.h5",
    # "merged_all_showers_part018.h5",
    # "merged_all_showers_part019.h5",
    # "merged_all_showers_part020.h5",
    # "merged_all_showers_part021.h5",
    # "merged_all_showers_part022.h5",
    # "merged_all_showers_part023.h5",
    # "merged_all_showers_part024.h5",
    # "merged_all_showers_part025.h5",
    # "merged_all_showers_part026.h5",
    # "merged_all_showers_part027.h5",
    # "merged_all_showers_part028.h5",
    # "merged_all_showers_part029.h5",
    "merged_all_showers.h5",
]


def check_file(file_path: Path):
    print(f"\nChecking: {file_path}")

    if not file_path.exists():
        print("  ❌ File not found")
        return

    with h5py.File(file_path, "r") as f:
        if "target/point_clouds" not in f:
            print("  ❌ target/point_clouds not found")
            return

        ds = f["target/point_clouds"]

        nan_count = 0
        inf_count = 0
        bad_entries = []

        for i in range(len(ds)):
            arr = ds[i]  # variable-length array

            if arr is None or len(arr) == 0:
                continue

            arr = np.asarray(arr)

            has_nan = np.isnan(arr).any()
            has_inf = np.isinf(arr).any()

            if has_nan:
                nan_count += 1
            if has_inf:
                inf_count += 1

            if has_nan or has_inf:
                bad_entries.append(i)

        print(f"  Total showers checked: {len(ds)}")
        print(f"  NaN entries: {nan_count}")
        print(f"  Inf entries: {inf_count}")

        if bad_entries:
            print(f"  First 10 bad indices: {bad_entries[:10]}")
        else:
            print("  ✅ No NaN or Inf found")


if __name__ == "__main__":
    for fname in FILES:
        check_file(BASE_DIR / fname)