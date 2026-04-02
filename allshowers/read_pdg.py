import h5py
import numpy as np

file_path = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_processed_step1_v3/merged_all_showers_randomized.h5"

with h5py.File(file_path, "r") as f:
    pdg = f["pdg"][:]

unique, counts = np.unique(pdg, return_counts=True)

print(f"Total showers: {len(pdg)}\n")
print(f"{'PDG Code':<12} {'Count':<10} {'Fraction':>8}")
print("-" * 32)
for code, count in zip(unique, counts):
    print(f"{code:<12} {count:<10} {count/len(pdg)*100:>7.2f}%")
