import h5py
import numpy as np

file_path = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_ot_matched/merged_all_showers.h5"

expected = set(range(24))
missing_count = 0

with h5py.File(file_path, "r") as f:
    data = f["showers"]

    for shower in data:
        reshaped = shower.reshape(-1, 4)
        z_values = set(reshaped[:, 2].astype(int))

        if z_values != expected:
            missing_count += 1

print("Total showers:", len(data))
print("Showers missing at least one plane:", missing_count)
print("Showers with all 24 planes:", len(data) - missing_count)