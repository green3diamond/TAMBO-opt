import h5py
import numpy as np

input_path = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_processed_step1_v3/merged_all_showers_randomized.h5"
output_path = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_processed_step1_v3/subset_40k_per_pdg.h5"

TARGET_PDGS = [11, 111, -211]
N_PER_PDG = 40000
TOTAL = N_PER_PDG * len(TARGET_PDGS)

with h5py.File(input_path, "r") as f_in:
    pdg = f_in["pdg"][:]

    # Collect indices: 40k random samples per PDG
    selected_indices = []
    for code in TARGET_PDGS:
        idx = np.where(pdg == code)[0]
        chosen = np.random.choice(idx, size=N_PER_PDG, replace=False)
        print(f"PDG {code}: selected {len(chosen)} from {len(idx)} available")
        selected_indices.append(chosen)

    selected_indices = np.sort(np.concatenate(selected_indices))
    print(f"\nTotal selected showers: {len(selected_indices)}")

    with h5py.File(output_path, "w") as f_out:
        for name, obj in f_in.items():
            if isinstance(obj, h5py.Dataset):
                if name == "shape":
                    # Update first value to reflect new total showers
                    new_shape = obj[:].copy()
                    new_shape[0] = TOTAL
                    print(f"Updating 'shape' dataset: {obj[:]} -> {new_shape}")
                    f_out.create_dataset(name, data=new_shape, dtype=obj.dtype)
                else:
                    print(f"Saving dataset: {name} ...")
                    subset = obj[:][selected_indices]
                    f_out.create_dataset(name, data=subset, dtype=obj.dtype)

print("\nDone! File saved to:", output_path)