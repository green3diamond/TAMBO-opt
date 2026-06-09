#!/usr/bin/env python3
"""
Check whether each shower satisfies a layer/z neighborhood condition, and
optionally write a filtered H5 file with bad showers removed.

This version supports the TAMBO/showerdata H5 format where the file contains:
    showers      Dataset {N/Inf}
    energies     Dataset {N/Inf, 1}
    directions   Dataset {N/Inf, 3}
    pdg          Dataset {N/Inf}
    actual_pdg   Dataset {N/Inf}
    num_points   Dataset {N/Inf}
    shower_ids   Dataset {N/Inf}
    target       Group
    shape        Dataset {3}

It reads shower points using showerdata.ShowerDataFile, so it works even when
the raw H5 "showers" dataset is not a simple [N, points, features] array.

Assumed point layout from showerdata:
    points[..., 0] = x
    points[..., 1] = y
    points[..., 2] = z / layer
    points[..., 3] = energy
    points[..., 4] = time, if present

Default condition:
    For every active layer z in a shower, all layers in
    [z - num_layer_cond, z + num_layer_cond] must also be present among active
    hits in that same shower, clipped to the global layer range.

Active hits are selected by energy > --energy-threshold.

Examples:
    python check_layer_window_showerdata.py file.h5 --num-layer-cond 4

    python check_layer_window_showerdata.py file.h5 --num-layer-cond 4 --remove -o filtered.h5

    python check_layer_window_showerdata.py file.h5 --num-layer-cond 4 --remove --in-place
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import Counter
from typing import Any

import h5py
import numpy as np
import showerdata


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Check z/layer neighborhood validity in showerdata H5 files."
    )
    p.add_argument("file", help="Input H5 file")
    p.add_argument(
        "--num-layer-cond",
        type=int,
        required=True,
        help="Required +/- layer window around each active layer",
    )
    p.add_argument(
        "--energy-threshold",
        type=float,
        default=0.0,
        help="Hits with energy > threshold are considered active. Default: 0.0",
    )
    p.add_argument(
        "--remove",
        action="store_true",
        help="Write a filtered output file with bad showers removed.",
    )
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output H5 file when --remove is used. Default: <input>_filtered.h5",
    )
    p.add_argument(
        "--in-place",
        action="store_true",
        help=(
            "Replace the input file with the filtered file. "
            "A .bak backup is created. Requires --remove."
        ),
    )
    p.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep showers with zero active hits. By default they are removed.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for scanning with showerdata. Default: 1024",
    )
    p.add_argument(
        "--max-print",
        type=int,
        default=20,
        help="Maximum number of bad shower indices to print.",
    )
    return p.parse_args()


def to_layer(z_values: np.ndarray) -> np.ndarray:
    # Match training code:
    # layer = (data["shower"][:, :, [2]] + 0.1).long()
    return (z_values + 0.1).astype(np.int64)


def get_num_showers(path: str) -> int:
    with h5py.File(path, "r") as h5:
        if "showers" in h5:
            return int(h5["showers"].shape[0])
        # fallback: first dataset with nonzero first dimension
        for _, obj in h5.items():
            if isinstance(obj, h5py.Dataset) and obj.shape:
                return int(obj.shape[0])
    raise RuntimeError("Could not determine number of showers.")


def pass1_global_layer_range(
    path: str,
    n: int,
    batch_size: int,
    energy_threshold: float,
) -> tuple[int | None, int | None]:
    global_min = None
    global_max = None

    with showerdata.ShowerDataFile(path, "r") as f:
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            batch = f[start:stop].points  # [batch, max_points, 4 or 5]
            active = batch[..., 3] > energy_threshold
            if np.any(active):
                layers = to_layer(batch[..., 2][active])
                mn = int(np.min(layers))
                mx = int(np.max(layers))
                global_min = mn if global_min is None else min(global_min, mn)
                global_max = mx if global_max is None else max(global_max, mx)

            print(f"pass 1: scanned {stop}/{n}", flush=True)

    return global_min, global_max


def check_one_shower(
    points: np.ndarray,
    num_layer_cond: int,
    energy_threshold: float,
    global_min_layer: int,
    global_max_layer: int,
    keep_empty: bool,
) -> tuple[bool, str]:
    if points.ndim != 2 or points.shape[-1] < 4:
        return False, f"bad_points_shape_{points.shape}"

    active = points[:, 3] > energy_threshold
    if not np.any(active):
        return (True, "empty") if keep_empty else (False, "no_active_hits")

    layers = to_layer(points[active, 2])
    present = set(int(z) for z in layers.tolist())

    # Condition: every active layer must have all neighboring layers
    # within +/- num_layer_cond, except outside global detector bounds.
    for z in sorted(present):
        lo = max(global_min_layer, z - num_layer_cond)
        hi = min(global_max_layer, z + num_layer_cond)
        required = set(range(lo, hi + 1))
        missing = sorted(required - present)
        if missing:
            return False, f"layer_{z}_missing_{missing[:10]}"

    return True, "ok"


def compute_valid_mask(
    path: str,
    n: int,
    batch_size: int,
    num_layer_cond: int,
    energy_threshold: float,
    keep_empty: bool,
) -> tuple[np.ndarray, list[str]]:
    global_min, global_max = pass1_global_layer_range(
        path=path,
        n=n,
        batch_size=batch_size,
        energy_threshold=energy_threshold,
    )

    valid = np.ones(n, dtype=bool)
    reasons = ["ok"] * n

    if global_min is None or global_max is None:
        valid[:] = keep_empty
        reasons = ["no_active_hits"] * n
        return valid, reasons

    print(f"global active layer range: [{global_min}, {global_max}]", flush=True)

    with showerdata.ShowerDataFile(path, "r") as f:
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            batch = f[start:stop].points

            for j in range(stop - start):
                ok, reason = check_one_shower(
                    batch[j],
                    num_layer_cond=num_layer_cond,
                    energy_threshold=energy_threshold,
                    global_min_layer=global_min,
                    global_max_layer=global_max,
                    keep_empty=keep_empty,
                )
                idx = start + j
                valid[idx] = ok
                reasons[idx] = reason

            print(f"pass 2: checked {stop}/{n}", flush=True)

    return valid, reasons


def copy_attrs(src: Any, dst: Any) -> None:
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def copy_dataset_filtered(
    src_ds: h5py.Dataset,
    dst_parent: h5py.Group,
    name: str,
    valid: np.ndarray,
    n: int,
) -> None:
    """
    Copy dataset. If first dimension is N, keep only valid entries.
    Works for normal and variable-length datasets.
    """
    kwargs = {}
    if src_ds.compression is not None:
        kwargs["compression"] = src_ds.compression
    if src_ds.compression_opts is not None:
        kwargs["compression_opts"] = src_ds.compression_opts
    if src_ds.shuffle:
        kwargs["shuffle"] = src_ds.shuffle
    if src_ds.fletcher32:
        kwargs["fletcher32"] = src_ds.fletcher32

    should_filter = src_ds.shape and src_ds.shape[0] == n

    if should_filter:
        idx = np.flatnonzero(valid)

        if src_ds.dtype.metadata and "vlen" in src_ds.dtype.metadata:
            # Variable length dataset: build object array.
            data = np.empty(len(idx), dtype=src_ds.dtype)
            for out_i, in_i in enumerate(idx):
                data[out_i] = src_ds[int(in_i)]
        else:
            # h5py supports sorted integer advanced indexing.
            data = src_ds[idx]
    else:
        data = src_ds[...]

    if src_ds.chunks is not None and np.shape(data):
        chunks = list(src_ds.chunks)
        chunks[0] = min(chunks[0], max(1, np.shape(data)[0]))
        kwargs["chunks"] = tuple(chunks)

    dst_ds = dst_parent.create_dataset(name, data=data, dtype=src_ds.dtype, **kwargs)
    copy_attrs(src_ds, dst_ds)


def filtered_copy(src_path: str, dst_path: str, valid: np.ndarray) -> None:
    n = len(valid)

    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        copy_attrs(src, dst)

        def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            parent_name = os.path.dirname(name)
            local_name = os.path.basename(name)
            parent = dst.require_group(parent_name) if parent_name else dst

            if isinstance(obj, h5py.Group):
                grp = dst.require_group(name)
                copy_attrs(obj, grp)
                return

            if isinstance(obj, h5py.Dataset):
                copy_dataset_filtered(obj, parent, local_name, valid, n)

        src.visititems(visitor)


def main() -> None:
    args = parse_args()

    if args.in_place and not args.remove:
        raise SystemExit("--in-place requires --remove")

    if args.num_layer_cond < 0:
        raise SystemExit("--num-layer-cond must be >= 0")

    n = get_num_showers(args.file)

    print(f"file: {args.file}")
    print(f"total showers: {n}")
    print(f"num_layer_cond: +/- {args.num_layer_cond}")
    print(f"energy threshold: {args.energy_threshold}")

    valid, reasons = compute_valid_mask(
        path=args.file,
        n=n,
        batch_size=args.batch_size,
        num_layer_cond=args.num_layer_cond,
        energy_threshold=args.energy_threshold,
        keep_empty=args.keep_empty,
    )

    bad = np.flatnonzero(~valid)
    print()
    print(f"valid showers: {int(valid.sum())}")
    print(f"bad showers:   {len(bad)}")

    if len(bad):
        print(f"first bad indices: {bad[:args.max_print].tolist()}")
        print("bad reason summary:")
        counts = Counter(reasons[i] for i in bad)
        for reason, count in counts.most_common(20):
            print(f"  {reason}: {count}")

    if args.remove:
        out = args.output
        if out is None:
            base, ext = os.path.splitext(args.file)
            out = f"{base}_filtered{ext or '.h5'}"

        if os.path.abspath(out) == os.path.abspath(args.file):
            raise SystemExit("Output path must differ from input path unless using --in-place.")

        print(f"\nwriting filtered file: {out}")
        filtered_copy(args.file, out, valid)

        if args.in_place:
            backup = args.file + ".bak"
            print(f"creating backup: {backup}")
            shutil.copy2(args.file, backup)
            print(f"replacing input with filtered file: {args.file}")
            shutil.move(out, args.file)

        print("done")


if __name__ == "__main__":
    main()