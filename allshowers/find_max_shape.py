import showerdata

# path = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_processed_step1_v5/merged_all_showers.h5"
path = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_processed_step1_with_bins_256_with_time/merged_all_showers.h5"

# load all showers
showers = showerdata.load(path)

# print shape
print("showers.points.shape:", showers.points.shape)