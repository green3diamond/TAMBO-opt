import numpy as np
import showerdata

FILE = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/h5_files_v2/combined_electrons.h5"

# load showers
showers = showerdata.load(FILE)

# columns:
# 0=x, 1=y, 2=layer, 3=energy, 4=time
points = showers.points

if points.shape[2] < 5:
    raise ValueError("Dataset has no time component.")

time = points[:, :, 4]
energy = points[:, :, 3]

# valid hits only
mask = energy > 0
time_valid = time[mask]

print("==== TIME STATS ====")
print("num valid hits:", len(time_valid))
print("mean:", np.mean(time_valid))
print("std:", np.std(time_valid))
print("min:", np.min(time_valid))
print("max:", np.max(time_valid))

print("\n==== PERCENTILES ====")
for p in [0, 0.1, 1, 5, 25, 50, 75, 95, 99, 99.9, 100]:
    print(f"{p:5.1f}% :", np.percentile(time_valid, p))

print("\n==== SPECIAL COUNTS ====")
print("num zeros:", np.sum(time_valid == 0))
print("num negative:", np.sum(time_valid < 0))
print("num nan:", np.sum(np.isnan(time_valid)))
print("num inf:", np.sum(np.isinf(time_valid)))

# check for extreme outliers
threshold = np.mean(time_valid) + 10 * np.std(time_valid)
num_outliers = np.sum(time_valid > threshold)

print("\n==== OUTLIERS ====")
print("threshold (mean + 10*std):", threshold)
print("num outliers:", num_outliers)