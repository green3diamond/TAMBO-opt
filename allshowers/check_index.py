import h5py
import numpy as np

FILE_PATH = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_ot_matched/merged_all_showers.h5"

FIELDS_PER_PARTICLE = 4
Z_OFFSET = 2  # x, y, z, e -> z is index 2

with h5py.File(FILE_PATH, "r") as f:
    showers = f["showers"][:]  # shape: (120000,) dtype: object

n_showers = len(showers)
has_z24 = np.zeros(n_showers, dtype=bool)

for i, shower in enumerate(showers):
    # shower is a flat 1D array: x0,y0,z0,e0, x1,y1,z1,e1, ...
    z_values = shower[Z_OFFSET::FIELDS_PER_PARTICLE]  # every 4th value starting at index 2
    has_z24[i] = np.any(z_values == 23)

yes_count = has_z24.sum()
no_count  = (~has_z24).sum()

print(f"Showers WITH z == 24    : {yes_count}")
print(f"Showers WITHOUT z == 24 : {no_count}")
print(f"Shower indices WITH z == 24 (first 20): {np.where(has_z24)[0][:20]}")