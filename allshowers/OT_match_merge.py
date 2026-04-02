#!/usr/bin/env python3
"""
merge_split_h5.py

Merge split HDF5 parts like:
  merged_all_showers_part*.h5
into a single:
  merged_all_showers.h5

Handles:
- numeric datasets: concatenated along axis=0
- ragged/vlen datasets stored as dtype=object: concatenated (stored as vlen float32)
- nested groups like target/*
- metadata datasets "shape" and "target/shape": updated after merge to merged N

Default input dir:
  /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_ot_matched/
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, Tuple

import h5py
import numpy as np


IN_DIR_DEFAULT = (
    "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_ot_matched/"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge split HDF5 part files into one file.")
    p.add_argument("--in_dir", type=str, default=IN_DIR_DEFAULT, help="Directory containing part files.")
    p.add_argument(
        "--pattern",
        type=str,
        default="merged_all_showers_part*.h5",
        help="Glob pattern for part files inside in_dir.",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output merged HDF5 path. Default: <in_dir>/merged_all_showers.h5",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists.")
    return p.parse_args()


def copy_attrs(src, dst) -> None:
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def ensure_parent_group(outf: h5py.File, ds_path: str) -> h5py.Group:
    """Ensure parent groups exist for ds_path, return the parent group."""
    parent_path = os.path.dirname(ds_path)
    if parent_path in ("", "/"):
        return outf
    g: h5py.Group = outf
    for part in parent_path.strip("/").split("/"):
        g = g.require_group(part)
    return g


def should_concat(ds: h5py.Dataset) -> bool:
    """
    Decide whether a dataset should be concatenated.
    - Scalars: copy
    - 1D small metadata named "shape": copy (we will update later)
    """
    if ds.shape is None or ds.shape == ():
        return False
    name = ds.name.split("/")[-1]
    if name == "shape" and ds.ndim == 1 and ds.shape[0] <= 16:
        return False
    return True


def infer_vlen_base_dtype(ds: h5py.Dataset) -> np.dtype:
    """
    For object datasets that are actually variable-length arrays, we store as vlen float32.
    Your examples show float32 arrays, so default to float32.
    """
    # If already a vlen dtype, try to keep the base type
    try:
        md = ds.dtype.metadata or {}
        if "vlen" in md:
            return np.dtype(md["vlen"])
    except Exception:
        pass
    return np.dtype(np.float32)


def main() -> None:
    args = parse_args()
    in_dir = args.in_dir
    out_path = args.out or os.path.join(in_dir, "merged_all_showers.h5")

    files = sorted(glob.glob(os.path.join(in_dir, args.pattern)))
    if not files:
        raise SystemExit(f"No files matched: {os.path.join(in_dir, args.pattern)}")

    if os.path.exists(out_path):
        if args.overwrite:
            os.remove(out_path)
        else:
            raise SystemExit(f"Output exists: {out_path}\nUse --overwrite to replace it.")

    # ---- Read dataset layout from first file ----
    with h5py.File(files[0], "r") as f0:
        dataset_paths: list[str] = []

        def visitor(_name: str, obj):
            if isinstance(obj, h5py.Dataset):
                dataset_paths.append(obj.name)

        f0.visititems(visitor)
        dataset_paths = sorted(set(dataset_paths))

        # Decide concat/copy and compute total lengths
        modes: Dict[str, str] = {}
        total_len: Dict[str, int] = {}

        for pth in dataset_paths:
            ds0 = f0[pth]
            mode = "concat" if should_concat(ds0) else "copy"
            modes[pth] = mode
            if mode == "concat":
                total_len[pth] = int(ds0.shape[0])

        # Add up lengths across all files for concat datasets
        for fp in files[1:]:
            with h5py.File(fp, "r") as fi:
                for pth, mode in modes.items():
                    if mode != "concat":
                        continue
                    if pth not in fi:
                        raise SystemExit(f"Dataset missing in {fp}: {pth}")
                    total_len[pth] += int(fi[pth].shape[0])

        # ---- Create output file and datasets ----
        with h5py.File(out_path, "w") as out:
            # Copy root attributes from first file
            copy_attrs(f0, out)

            # Create datasets
            for pth in dataset_paths:
                src = f0[pth]
                parent = ensure_parent_group(out, pth)
                name = os.path.basename(pth)

                if modes[pth] == "copy":
                    # Copy exactly (with special handling for object arrays)
                    if src.dtype.kind == "O":
                        base = infer_vlen_base_dtype(src)
                        dt = h5py.vlen_dtype(base)
                        data = src[...]
                        ds_new = parent.create_dataset(name, data=data, dtype=dt)
                    else:
                        data = src[...]
                        ds_new = parent.create_dataset(name, data=data, dtype=src.dtype)
                    copy_attrs(src, ds_new)
                    continue

                # concat dataset
                n_total = total_len[pth]
                if src.dtype.kind == "O":
                    base = infer_vlen_base_dtype(src)
                    dt = h5py.vlen_dtype(base)
                    ds_new = parent.create_dataset(name, shape=(n_total,), dtype=dt, chunks=True)
                else:
                    new_shape = (n_total,) + src.shape[1:]
                    ds_new = parent.create_dataset(
                        name,
                        shape=new_shape,
                        dtype=src.dtype,
                        chunks=True,
                        compression=None,
                    )
                copy_attrs(src, ds_new)

            # ---- Write concatenated data ----
            write_pos = {pth: 0 for pth, mode in modes.items() if mode == "concat"}

            for fp in files:
                with h5py.File(fp, "r") as fi:
                    for pth, mode in modes.items():
                        if mode != "concat":
                            continue
                        src = fi[pth]
                        n = int(src.shape[0])
                        start = write_pos[pth]
                        end = start + n
                        out_ds = out[pth]
                        out_ds[start:end] = src[...]
                        write_pos[pth] = end

            # ---- Sanity check ----
            for pth in write_pos:
                if write_pos[pth] != total_len[pth]:
                    raise RuntimeError(
                        f"Write mismatch for {pth}: wrote {write_pos[pth]} / expected {total_len[pth]}"
                    )

        # ---- Update metadata shapes after merge ----
        with h5py.File(out_path, "a") as out:
            # Prefer showers length; fallback to any known top-level vector dataset
            if "showers" in out:
                n_total = int(out["showers"].shape[0])
            elif "target/point_clouds" in out:
                n_total = int(out["target/point_clouds"].shape[0])
            else:
                raise RuntimeError("Could not infer merged N (missing 'showers' and 'target/point_clouds').")

            # Update /shape if present (e.g., [N, 6016, 4])
            if "shape" in out:
                shp = np.array(out["shape"][...], dtype=np.int32)
                if shp.ndim == 1 and shp.size >= 1:
                    shp[0] = n_total
                    out["shape"][...] = shp

            # Update /target/shape if present (e.g., [N, 6016, 3])
            if "target" in out and "shape" in out["target"]:
                tshp = np.array(out["target/shape"][...], dtype=np.int32)
                if tshp.ndim == 1 and tshp.size >= 1:
                    tshp[0] = n_total
                    out["target/shape"][...] = tshp

    print(f"✅ Merged {len(files)} files into: {out_path}")
    print("✅ Updated metadata: shape and target/shape first element set to merged N.")


if __name__ == "__main__":
    main()