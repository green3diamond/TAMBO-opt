import h5py
import numpy as np

files = [
    {
        "path": "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/combined_muons_balanced.h5",
        "pointcloud": 20000,
    },
    {
        "path": "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/combined_muons_balanced-test-file.h5",
        "pointcloud": 20000,
    },
    {
        "path": "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/combined_photons_balanced.h5",
        "pointcloud": 6000,
    },
    {
        "path": "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/combined_photons_balanced-test-file.h5",
        "pointcloud": 6000,
    },
]

for f in files:
    path = f["path"]
    pointcloud = f["pointcloud"]

    with h5py.File(path, "r+") as out:
        # Read existing shape to infer total
        old_shape = out["shape"][:]
        print(f"\n[{path}]")
        print(f"  Old shape dataset: {old_shape}")

        total = int(old_shape[0])  # infer total from existing shape[0]

        new_shape = np.array([total, pointcloud, 5], dtype=np.int64)

        # Delete old dataset and write new one
        del out["shape"]
        out.create_dataset("shape", data=new_shape)

        print(f"  New shape dataset: {new_shape}")

print("\nDone.")