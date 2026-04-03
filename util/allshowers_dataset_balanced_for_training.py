import h5py
import numpy as np
import os
import argparse
from tqdm import tqdm

'''
python /n/home04/hhanif/AllShowers/util/allshowers_dataset_balanced_for_training.py /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/combined_electrons.h5  --also-test
'''

parser = argparse.ArgumentParser()
parser.add_argument('--also-test', action='store_true', help='Also create a test file from the remaining samples')
parser.add_argument('--test-file', action='store_true', help='Only create the test file from remaining samples')
parser.add_argument('input', help='Path to input HDF5 file')
args = parser.parse_args()

path = args.input
base, ext = os.path.splitext(path)
out_path  = f"{base}_balanced{ext}"
test_path = f"{base}_balanced-test-file{ext}"

do_train = not args.test_file
do_test  = args.also_test or args.test_file

CHUNK_SIZE = 500

datasets = ['showers', 'directions', 'energies', 'pdg', 'actual_pdg', 'shower_ids', 'num_points']


def write_h5(out_path, f, idx_sorted, shuffle_seed, datasets):
    if os.path.exists(out_path):
        os.remove(out_path)
        print(f"Removed existing file: {out_path}")

    rng_shuffle = np.random.default_rng(shuffle_seed)
    shuffle_order = rng_shuffle.permutation(len(idx_sorted))

    idx_final = idx_sorted[shuffle_order]
    argsort = np.argsort(idx_final)
    idx_read_order = idx_final[argsort]
    restore_order = np.argsort(argsort)

    total = len(idx_read_order)

    with h5py.File(out_path, 'w') as out:
        print("Creating output datasets...")
        for name in datasets:
            ds = f[name]
            shape = (total,) + ds.shape[1:]
            out.create_dataset(name, shape=shape, dtype=ds.dtype, chunks=True)
        out.create_dataset('shape', data=np.array([total, 2000, 5], dtype=np.int64))

        for name in datasets:
            print(f"\nProcessing '{name}'...")
            ds = f[name]

            all_data = []
            for start in tqdm(range(0, total, CHUNK_SIZE), desc=f"  Reading '{name}'", unit="chunk"):
                end = min(start + CHUNK_SIZE, total)
                chunk_idx = idx_read_order[start:end]
                all_data.append(ds[chunk_idx])

            print(f"  Concatenating & shuffling '{name}'...")
            all_data = np.concatenate(all_data, axis=0)
            all_data = all_data[restore_order]

            print(f"  Writing '{name}'...")
            out[name][:] = all_data
            del all_data

    print(f"\nDone writing {out_path}! {total} entries saved.")
    print(f"Shape dataset: [{total}, 2000, 5]")


with h5py.File(path, 'r') as f:
    pdg = f['pdg'][:]

    idx_0 = np.where(pdg == 0)[0]
    idx_1 = np.where(pdg == 1)[0]

    rng = np.random.default_rng(42)
    idx_0_sampled = rng.choice(idx_0, size=365000, replace=False)
    idx_1_sampled = rng.choice(idx_1, size=365000, replace=False)

    if do_test:
        idx_0_remaining = np.setdiff1d(idx_0, idx_0_sampled)
        idx_1_remaining = np.setdiff1d(idx_1, idx_1_sampled)
        idx_test_sorted = np.sort(np.concatenate([idx_0_remaining, idx_1_remaining]))
        print(f"\nTest set: {len(idx_0_remaining)} class-0, {len(idx_1_remaining)} class-1 ({len(idx_test_sorted)} total)")

    if do_train:
        idx_train_sorted = np.sort(np.concatenate([idx_0_sampled, idx_1_sampled]))
        print(f"Train set: 365000 per class (730000 total)")
        write_h5(out_path, f, idx_train_sorted, shuffle_seed=99, datasets=datasets)

    if do_test:
        write_h5(test_path, f, idx_test_sorted, shuffle_seed=77, datasets=datasets)