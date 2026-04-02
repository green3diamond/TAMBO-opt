
import h5py

# file_path = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_ot_matched/merged_all_showers_part000.h5"


# file_path = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_ot_matched/merged_all_showers.h5"

file_path = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_processed_step1_v3/merged_all_showers.h5"
def print_first_entries(name, obj):
    """
    Called for every object in the HDF5 file.
    Prints first 10 entries if it's a dataset.
    """
    if isinstance(obj, h5py.Dataset):
        print(f"\nDataset: {name}")
        print(f"Shape: {obj.shape}")
        print(f"Dtype: {obj.dtype}")

        # try:
        #     print("First 10 entries:")
        #     print(obj[:10])
        # except Exception as e:
        #     print(f"Could not read entries: {e}")

with h5py.File(file_path, "r") as f:
    print("File opened successfully.\n")
    print("Listing datasets and printing first 10 entries...\n")

    # Traverse all groups/datasets
    f.visititems(print_first_entries)
