import showerdata
import numpy as np

showers = showerdata.load("/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_processed_step1_with_bins_256_with_time/merged_all_showers.h5", stop=10000)
t = showers.points[:, :, 4]
e = showers.points[:, :, 3]
mask = e > 0  # only real hits

t_hits = t[mask]
print(f"min:    {t_hits.min():.20f} s")
print(f"max:    {t_hits.max():.20f} s")
print(f"mean:   {t_hits.mean():.20f} s")
print(f"median: {np.median(t_hits):.20f} s")
print(f"any zero or negative: {(t_hits <= 0).sum()}")