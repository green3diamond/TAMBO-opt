import argparse
import os
from pathlib import Path

import showerdata

'''

python /n/home04/hhanif/AllShowers/allshowers/OT_match_split.py \
  --infile /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_processed_step1_v3/subset_40k_per_pdg.h5 \
  --outdir /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_ot_matched \
  --chunk 10000 

'''

def parse_args():
    p = argparse.ArgumentParser("Split a showerdata HDF5 file into smaller parts using showerdata.save_batch.")
    p.add_argument("--infile", required=True, help="Input HDF5 file path")
    p.add_argument("--outdir", required=True, help="Output directory for parts")
    p.add_argument("--chunk", type=int, default=20000, help="Showers per part file")
    p.add_argument("--prefix", default="merged_all_showers_part", help="Output filename prefix")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing part files")
    return p.parse_args()


def main():
    a = parse_args()
    in_path = a.infile
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    shape = showerdata.get_file_shape(in_path)  # (N, P, F)
    N, P, F = shape
    print(f"Input shape: N={N}, P={P}, F={F}")

    part = 0
    for start in range(0, N, a.chunk):
        stop = min(start + a.chunk, N)
        out_path = outdir / f"{a.prefix}{part:03d}.h5"

        if out_path.exists():
            if a.overwrite:
                out_path.unlink()
            else:
                print(f"Skipping existing {out_path}")
                part += 1
                continue

        # Create empty file of exact part shape
        part_shape = (stop - start, P, F)
        showerdata.create_empty_file(str(out_path), shape=part_shape)
        print(f"Created {out_path} with shape={part_shape}")

        # Read from input in batches and write into the part file
        # Use ShowerDataFile slicing so we don't load too much at once
        batch_size = 1024
        with showerdata.ShowerDataFile(in_path, "r") as fin:
            write_pos = 0
            for s in range(start, stop, batch_size):
                e = min(s + batch_size, stop)
                showers_batch = fin[s:e]  # returns a "Showers" object
                showerdata.save_batch(showers_batch, str(out_path), start=write_pos)
                write_pos += (e - s)

        print(f"Wrote part {part:03d}: [{start}:{stop}) -> {out_path}")
        part += 1


if __name__ == "__main__":
    main()