import h5py
import numpy as np

file_path = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_ot_matched/merged_all_showers.h5"

CHUNK_SIZE = 5000


def check_numeric_array(data):
    """Check numpy array for negative, nan, inf."""
    return (
        np.any(data < 0),
        np.any(np.isnan(data)),
        np.any(np.isinf(data)),
    )


def check_dataset(name, dataset):
    if not isinstance(dataset, h5py.Dataset):
        return

    print(f"\nChecking dataset: {name}")
    print(f"Shape: {dataset.shape}, dtype: {dataset.dtype}")

    has_negative = False
    has_nan = False
    has_inf = False

    # ---------- NORMAL NUMERIC DATASET ----------
    if np.issubdtype(dataset.dtype, np.number):

        n = dataset.shape[0] if dataset.shape else 1

        if dataset.shape == ():
            neg, nan, inf = check_numeric_array(dataset[()])
            has_negative |= neg
            has_nan |= nan
            has_inf |= inf

        else:
            for i in range(0, n, CHUNK_SIZE):
                data = dataset[i:i + CHUNK_SIZE]
                neg, nan, inf = check_numeric_array(data)

                has_negative |= neg
                has_nan |= nan
                has_inf |= inf

                if has_negative and has_nan and has_inf:
                    break

    # ---------- OBJECT DATASET (YOUR SHOWERS / POINT_CLOUDS) ----------
    elif dataset.dtype == object:
        print("→ Object dataset detected (checking each entry)")

        n = dataset.shape[0]

        for i in range(0, n, CHUNK_SIZE):
            chunk = dataset[i:i + CHUNK_SIZE]

            for arr in chunk:
                if arr is None:
                    continue

                arr = np.asarray(arr)

                neg, nan, inf = check_numeric_array(arr)

                has_negative |= neg
                has_nan |= nan
                has_inf |= inf

                if has_negative and has_nan and has_inf:
                    break

            if has_negative and has_nan and has_inf:
                break

    else:
        print("→ Skipped (unsupported dtype)")
        return

    print(f"→ Negative values: {has_negative}")
    print(f"→ NaN values: {has_nan}")
    print(f"→ Inf values: {has_inf}")


with h5py.File(file_path, "r") as f:
    print("Scanning file...\n")
    f.visititems(check_dataset)

print("\nDone.")