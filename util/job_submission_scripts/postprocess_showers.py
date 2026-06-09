#!/usr/bin/env python3
"""
Post-process a CHUNK of combined hit-level Parquet files into 3 HDF5 files.

Two-zone centroid approach
--------------------------
Hits are split per shower by Chebyshev distance:  cheb = max(|x|, |y|)

    zone "near"  :  cheb <  zone_boundary   → clustered with dx_near
    zone "far"   :  cheb >= zone_boundary   → clustered with dx_far

The two centroid pools are combined, truncated to top-Nmax by energy,
then sorted by plane_idx before writing to HDF5.

Per-particle CLI flags:
    --electrons-dx-near  --electrons-dx-far  --electrons-nmax
    --muons-dx-near      --muons-dx-far      --muons-nmax
    --photons-dx-near    --photons-dx-far    --photons-nmax
    --zone-boundary      (default 10 000 m = 10 km)

Usage:
    python postprocess_showers.py \\
        --chunk-list /path/to/chunk_0000.txt \\
        --incident-pdg 11 \\
        --chunk-id 0 \\
        --output-dir /path/to/output \\
        --electrons-dx-near 5  --electrons-dx-far 10  --electrons-nmax 4000 \\
        --muons-dx-near     10 --muons-dx-far     20  --muons-nmax     28000 \\
        --photons-dx-near   5  --photons-dx-far   10  --photons-nmax   8000 \\
        --zone-boundary     10000
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime
from typing import Optional


# =============================================================================
# Particle type configuration
# =============================================================================

PARTICLE_CONFIGS = {
    "electrons": {
        "pdg_values":    [11],
        "label":         "e± (pdg 11)",
        "default_nmax":  4000,
        "default_dx_near": 5.0,
        "default_dx_far":  10.0,
        "h5_suffix":     "electrons",
    },
    "muons": {
        "pdg_values":    [13],
        "label":         "μ± (pdg 13)",
        "default_nmax":  28000,
        "default_dx_near": 10.0,
        "default_dx_far":  20.0,
        "h5_suffix":     "muons",
    },
    "photons": {
        "pdg_values":    [22],
        "label":         "γ (pdg 22)",
        "default_nmax":  8000,
        "default_dx_near": 5.0,
        "default_dx_far":  10.0,
        "h5_suffix":     "photons",
    },
}

DEFAULT_ZONE_BOUNDARY = 10_000.0   # metres


# =============================================================================
# Lazy imports
# =============================================================================

def _lazy_imports():
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    import h5py
    return np, pd, pa, pq, pc, h5py


# =============================================================================
# Helpers
# =============================================================================

def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


def _pdg_class(pdg_primary: float) -> int:
    """Binary class label: 0 = e±/γ/π⁰, 1 = π±."""
    try:
        return 1 if abs(int(round(float(pdg_primary)))) == 211 else 0
    except Exception:
        return 0


# =============================================================================
# Event-CSV reader
# =============================================================================

def _event_csv_path_for(hits_parquet_path: str) -> str:
    base = hits_parquet_path
    if base.endswith("_hits.parquet"):
        return base[: -len("_hits.parquet")] + "_event.csv"
    root, _ = os.path.splitext(base)
    return root + "_event.csv"


def _load_event_row(csv_path: str) -> Optional[dict]:
    if not os.path.exists(csv_path):
        return None
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                return row
    except Exception:
        return None
    return None


def _float_or(default: float, val) -> float:
    try:
        v = float(val)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _extract_event_meta(csv_path: str) -> dict:
    row = _load_event_row(csv_path) or {}
    return {
        "shower_id":         str(row.get("shower_id", "")),
        "incident_energy":   _float_or(0.0, row.get("incident_energy")),
        "direction_x":       _float_or(0.0, row.get("direction_x")),
        "direction_y":       _float_or(0.0, row.get("direction_y")),
        "direction_z":       _float_or(1.0, row.get("direction_z")),
        "incident_pdg":      _float_or(0.0, row.get("incident_pdg")),
        "incident_class_id": _float_or(0.0, row.get("incident_class_id")),
    }


# =============================================================================
# Duplicate removal
# =============================================================================

def _drop_near_duplicates(df, np, pd,
                          time_tol: float, energy_rel_tol: float, xy_tol: float,
                          verbose: bool = False):
    if len(df) == 0:
        return df, 0

    sort_cols = ["plane_index", "pdg", "time", "kinetic_energy", "x", "y"]
    df_sorted = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    n = len(df_sorted)
    if n < 2:
        return df_sorted, 0

    t   = df_sorted["time"].to_numpy(dtype=np.float64)
    e   = df_sorted["kinetic_energy"].to_numpy(dtype=np.float64)
    xx  = df_sorted["x"].to_numpy(dtype=np.float64)
    yy  = df_sorted["y"].to_numpy(dtype=np.float64)
    pl  = df_sorted["plane_index"].to_numpy()
    pdg = df_sorted["pdg"].to_numpy()

    dt     = np.abs(np.diff(t))
    de     = np.abs(np.diff(e))
    emag   = np.maximum(np.abs(e[:-1]), np.abs(e[1:]))
    emag   = np.where(emag > 0, emag, 1.0)
    de_rel = de / emag
    dx_arr = np.abs(np.diff(xx))
    dy_arr = np.abs(np.diff(yy))

    same_plane = (pl[1:]  == pl[:-1])
    same_pdg   = (pdg[1:] == pdg[:-1])

    close = (
        same_plane & same_pdg
        & (dt     <= time_tol)
        & (de_rel <= energy_rel_tol)
        & (dx_arr <= xy_tol)
        & (dy_arr <= xy_tol)
    )

    dup_mask = np.zeros(n, dtype=bool)
    dup_mask[1:] = close

    n_removed = int(dup_mask.sum())
    if n_removed == 0:
        return df_sorted, 0

    if verbose:
        print(f"      dedup: removed {n_removed}/{n} near-duplicate hits "
              f"(time<{time_tol:.1e}s, dE/E<{energy_rel_tol:.1e}, dxy<{xy_tol:.1e}m)")

    return df_sorted.loc[~dup_mask].reset_index(drop=True), n_removed


# =============================================================================
# Zone split
# =============================================================================

def _split_by_zone(df_sub, zone_boundary: float, np):
    """
    Split hits by Chebyshev distance from origin.
        near : max(|x|, |y|) <  zone_boundary
        far  : max(|x|, |y|) >= zone_boundary
    Returns (df_near, df_far).
    """
    if len(df_sub) == 0:
        return df_sub, df_sub

    xy   = df_sub[["x", "y"]].to_numpy(dtype=np.float64)
    cheb = np.maximum(np.abs(xy[:, 0]), np.abs(xy[:, 1]))
    near_mask = cheb < zone_boundary
    return df_sub.iloc[near_mask], df_sub.iloc[~near_mask]


# =============================================================================
# Clustering
# =============================================================================

def _cluster_shower_part(xy, e, cell_size, shift, np, t=None):
    """Bin hits into (cell_size × cell_size) cells on one plane.
    Returns (xy_clustered, e_clustered, t_clustered)."""
    if len(xy) == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,),   dtype=np.float32),
            None if t is None else np.empty((0,), dtype=np.float32),
        )

    xy = xy.copy().astype(np.float32)
    xy += shift
    xy /= cell_size
    xy_idx = np.floor(xy).astype(np.int32)

    x_min  = int(xy_idx[:, 0].min())
    y_min  = int(xy_idx[:, 1].min())
    stride = int(xy_idx[:, 0].max()) - x_min + 2
    keys   = (xy_idx[:, 0].astype(np.int64) - x_min) * stride + \
             (xy_idx[:, 1].astype(np.int64) - y_min)

    unique_keys, inverse_idx = np.unique(keys, return_inverse=True)
    unique_x = (unique_keys // stride).astype(np.int32) + x_min
    unique_y = (unique_keys  % stride).astype(np.int32) + y_min

    e_clustered = np.zeros(len(unique_keys), dtype=np.float32)
    np.add.at(e_clustered, inverse_idx, e)

    if t is not None:
        t_clustered = np.full(len(unique_keys), np.inf, dtype=np.float32)
        np.minimum.at(t_clustered, inverse_idx, t)
    else:
        t_clustered = None

    xy_clustered = np.column_stack([unique_x, unique_y]).astype(np.float32)
    xy_clustered += 0.5
    xy_clustered *= cell_size
    xy_clustered -= shift
    return xy_clustered, e_clustered, t_clustered


def _cluster_hits_for_zone(df_sub, cell_size: float, np, include_time: bool):
    """
    Cluster hits in df_sub with the given cell_size.
    Returns ndarray (n_centroids, n_feat): x, y, plane_idx, energy [, time]
    """
    n_feat = 5 if include_time else 4
    if len(df_sub) == 0:
        return np.empty((0, n_feat), dtype=np.float32)

    pos   = df_sub[["x", "y", "plane_index"]].to_numpy(dtype=np.float32)
    e_all = np.maximum(df_sub["kinetic_energy"].to_numpy(dtype=np.float32), 0.0)
    t_all = (df_sub["time"].to_numpy(dtype=np.float32)
             if include_time and "time" in df_sub.columns else None)
    plane = pos[:, 2].astype(np.int32)

    parts = []
    for p_val in np.unique(plane):
        mask  = plane == p_val
        xy_p  = pos[mask, :2]
        e_p   = e_all[mask]
        t_p   = t_all[mask] if t_all is not None else None
        shift = np.array([0.0, 0.0], dtype=np.float32)

        xy_c, e_c, t_c = _cluster_shower_part(
            xy_p, e_p, cell_size, shift, np, t=t_p
        )
        if len(xy_c) == 0:
            continue

        cols = [xy_c[:, 0], xy_c[:, 1],
                np.full(len(xy_c), p_val, dtype=np.float32), e_c]
        if include_time:
            cols.append(t_c)
        parts.append(np.column_stack(cols).astype(np.float32))

    if not parts:
        return np.empty((0, n_feat), dtype=np.float32)
    return np.concatenate(parts, axis=0)


def _cluster_two_zones(df_sub, dx_near: float, dx_far: float,
                       zone_boundary: float, np, include_time: bool):
    """
    Split df_sub into near/far zones, cluster each with its own dx,
    then concatenate the centroid arrays.

    Returns combined ndarray (n_total_centroids, n_feat).
    """
    df_near, df_far = _split_by_zone(df_sub, zone_boundary, np)
    arr_near = _cluster_hits_for_zone(df_near, dx_near, np, include_time)
    arr_far  = _cluster_hits_for_zone(df_far,  dx_far,  np, include_time)

    if len(arr_near) == 0 and len(arr_far) == 0:
        n_feat = 5 if include_time else 4
        return np.empty((0, n_feat), dtype=np.float32)
    if len(arr_near) == 0:
        return arr_far
    if len(arr_far) == 0:
        return arr_near
    return np.concatenate([arr_near, arr_far], axis=0)


def _truncate_and_sort(arr, nmax: int, np, guarantee_all_planes: bool = False):
    """
    Keep top-nmax centroids by energy (global cut), then sort by plane_idx.

    If guarantee_all_planes=True:
        Each plane is first guaranteed its single highest-energy centroid
        (one seed per plane). The remaining budget (nmax - n_planes) is then
        filled globally from the highest-energy non-seed centroids.
        If nmax < n_planes, the nmax planes with the highest seed energy are kept.

    If guarantee_all_planes=False (default):
        Simple global top-Nmax cut. A plane with very high-energy centroids
        can consume the entire budget; other planes may get zero representation.
    """
    if len(arr) == 0:
        return arr

    if len(arr) <= nmax:
        order = np.argsort(arr[:, 2], kind="mergesort")
        return arr[order]

    if not guarantee_all_planes:
        # ── Simple global top-Nmax ────────────────────────────────────────
        top_idx = arr[:, 3].argpartition(-nmax)[-nmax:]
        result  = arr[top_idx]
        order   = np.argsort(result[:, 2], kind="mergesort")
        return result[order]

    # ── Guaranteed per-plane seed + global top-up ─────────────────────────
    plane_col  = arr[:, 2]
    energy_col = arr[:, 3]

    unique_planes = np.unique(plane_col)
    n_planes      = len(unique_planes)

    # Step 1: one seed per plane (highest-energy centroid)
    seed_indices = []
    for p in unique_planes:
        local_idx = np.where(plane_col == p)[0]
        best      = local_idx[energy_col[local_idx].argmax()]
        seed_indices.append(best)
    seed_indices = np.array(seed_indices, dtype=np.int64)

    if n_planes >= nmax:
        # Budget too small — keep the nmax planes with the highest seed energy
        seed_energies = energy_col[seed_indices]
        keep          = seed_indices[np.argsort(seed_energies)[::-1][:nmax]]
        result        = arr[keep]
        order         = np.argsort(result[:, 2], kind="mergesort")
        return result[order]

    # Step 2: fill remaining budget from the non-seed pool globally
    remaining_budget = nmax - n_planes
    seed_set         = set(seed_indices.tolist())
    all_idx          = np.arange(len(arr), dtype=np.int64)
    pool_indices     = all_idx[np.array([i not in seed_set for i in all_idx])]

    if len(pool_indices) <= remaining_budget:
        top_up = pool_indices
    else:
        pool_energies = energy_col[pool_indices]
        top_up        = pool_indices[
            np.argpartition(pool_energies, -remaining_budget)[-remaining_budget:]
        ]

    # Step 3: combine and sort by plane
    keep   = np.concatenate([seed_indices, top_up])
    result = arr[keep]
    order  = np.argsort(result[:, 2], kind="mergesort")
    return result[order]


# =============================================================================
# HDF5 writer
# =============================================================================

class ShowerH5Writer:
    """Appends one shower at a time into an H5 file with resizable datasets."""

    def __init__(self, path: str, nmax: int, n_feat: int, np, h5py,
                 extra_attrs: Optional[dict] = None):
        self.path   = path
        self.nmax   = int(nmax)
        self.n_feat = int(n_feat)
        self.np     = np
        self.h5py   = h5py

        self.f = h5py.File(path, "w")
        self.f.attrs["created"] = datetime.now().isoformat()
        self.f.attrs["nmax"]    = self.nmax
        self.f.attrs["n_feat"]  = self.n_feat
        if extra_attrs:
            for k, v in extra_attrs.items():
                try:
                    self.f.attrs[k] = v
                except Exception:
                    self.f.attrs[k] = str(v)

        dt_vlen = h5py.vlen_dtype(np.dtype("float32"))
        self.ds_showers    = self.f.create_dataset(
            "showers",    shape=(0,),    maxshape=(None,),    dtype=dt_vlen)
        self.ds_directions = self.f.create_dataset(
            "directions", shape=(0, 3),  maxshape=(None, 3),  dtype=np.float32)
        self.ds_energies   = self.f.create_dataset(
            "energies",   shape=(0, 1),  maxshape=(None, 1),  dtype=np.float32)
        self.ds_pdg        = self.f.create_dataset(
            "pdg",        shape=(0,),    maxshape=(None,),     dtype=np.int32)
        self.ds_actual_pdg = self.f.create_dataset(
            "actual_pdg", shape=(0,),    maxshape=(None,),     dtype=np.int32)
        self.ds_num_points = self.f.create_dataset(
            "num_points", shape=(0,),    maxshape=(None,),     dtype=np.int32)
        self.ds_shower_id  = self.f.create_dataset(
            "shower_id",  shape=(0,),    maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"))
        self.count = 0

    def append(self, arr, direction, energy, pdg_class, actual_pdg, shower_id=""):
        np   = self.np
        n_pts = int(arr.shape[0])
        flat  = arr.astype(np.float32, copy=False).ravel()
        new_n = self.count + 1

        self.ds_showers.resize((new_n,))
        self.ds_directions.resize((new_n, 3))
        self.ds_energies.resize((new_n, 1))
        self.ds_pdg.resize((new_n,))
        self.ds_actual_pdg.resize((new_n,))
        self.ds_num_points.resize((new_n,))
        self.ds_shower_id.resize((new_n,))

        self.ds_showers[self.count]     = flat
        self.ds_directions[self.count]  = np.asarray(direction, dtype=np.float32)[:3]
        self.ds_energies[self.count, 0] = np.float32(energy)
        self.ds_pdg[self.count]         = np.int32(pdg_class)
        self.ds_actual_pdg[self.count]  = np.int32(actual_pdg)
        self.ds_num_points[self.count]  = np.int32(n_pts)
        self.ds_shower_id[self.count]   = str(shower_id)
        self.count = new_n

    def close(self):
        import numpy as np
        if "shape" in self.f:
            del self.f["shape"]
        self.f.create_dataset(
            "shape",
            data=np.array([self.count, self.nmax, self.n_feat], dtype=np.int64),
            dtype=np.int64,
        )
        self.f.attrs["n_showers"] = self.count
        self.f.close()


# =============================================================================
# Core: process one chunk
# =============================================================================

def process_chunk(
    chunk_paths: list[str],
    output_dir: str,
    incident_pdg: int,
    chunk_id: int,
    nmax_per_particle: dict,
    dx_near_per_particle: dict,
    dx_far_per_particle: dict,
    zone_boundary: float = DEFAULT_ZONE_BOUNDARY,
    guarantee_all_planes: bool = False,
    include_time: bool = True,
    dedup_time_tol: float = 1e-15,
    dedup_energy_rel_tol: float = 1e-6,
    dedup_xy_tol: float = 1e-3,
    do_dedup: bool = True,
    particles: Optional[list[str]] = None,
    verbose: bool = True,
) -> dict:
    np, pd, pa, pq, pc, h5py = _lazy_imports()

    active = list(PARTICLE_CONFIGS.keys())
    if particles:
        bad = [p for p in particles if p not in PARTICLE_CONFIGS]
        if bad:
            raise ValueError(f"Unknown particle types: {bad}.")
        active = [p for p in PARTICLE_CONFIGS if p in particles]
        if not active:
            raise ValueError("Empty particle selection after filtering.")

    subdir = os.path.join(output_dir, f"pdg_{incident_pdg}")
    os.makedirs(subdir, exist_ok=True)

    result = {
        "incident_pdg":        incident_pdg,
        "chunk_id":            chunk_id,
        "n_parquets":          len(chunk_paths),
        "n_processed":         0,
        "n_skipped":           0,
        "n_dedup_removed":     0,
        "particles":           active,
        "zone_boundary":       zone_boundary,
        "guarantee_all_planes": guarantee_all_planes,
        "h5_files":            [],
        "h5_total_bytes":      0,
        "status":              "ok",
        "message":         "",
        "elapsed_s":       0.0,
    }
    t_start = time.time()
    n_feat  = 5 if include_time else 4

    # Open one H5 writer per active particle type
    writers = {}
    for pkey in active:
        cfg      = PARTICLE_CONFIGS[pkey]
        nmax     = int(nmax_per_particle.get(pkey,     cfg["default_nmax"]))
        dx_near  = float(dx_near_per_particle.get(pkey, cfg["default_dx_near"]))
        dx_far   = float(dx_far_per_particle.get(pkey,  cfg["default_dx_far"]))
        h5_path  = os.path.join(
            subdir, f"chunk_{chunk_id:04d}_{cfg['h5_suffix']}.h5")
        writers[pkey] = {
            "writer":  ShowerH5Writer(
                path=h5_path, nmax=nmax, n_feat=n_feat, np=np, h5py=h5py,
                extra_attrs={
                    "dx_near":             dx_near,
                    "dx_far":              dx_far,
                    "zone_boundary":       zone_boundary,
                    "guarantee_all_planes": guarantee_all_planes,
                    "nmax":                nmax,
                    "include_time":        include_time,
                    "incident_pdg":        incident_pdg,
                    "chunk_id":            chunk_id,
                    "particle_type":       pkey,
                },
            ),
            "nmax":    nmax,
            "dx_near": dx_near,
            "dx_far":  dx_far,
            "h5_path": h5_path,
        }

    required_cols = ["x", "y", "pdg", "time", "kinetic_energy", "plane_index"]

    for i, pq_path in enumerate(chunk_paths):
        if not os.path.exists(pq_path):
            result["n_skipped"] += 1
            if verbose:
                print(f"  [{i+1}/{len(chunk_paths)}] MISSING: {pq_path}")
            continue

        try:
            tbl = pq.read_table(pq_path, columns=required_cols)
        except Exception as e:
            result["n_skipped"] += 1
            if verbose:
                print(f"  [{i+1}/{len(chunk_paths)}] READ ERROR {pq_path}: {e}")
            continue

        if len(tbl) == 0:
            result["n_skipped"] += 1
            continue

        event_csv      = _event_csv_path_for(pq_path)
        meta           = _extract_event_meta(event_csv)
        actual_pdg_int = int(round(meta["incident_pdg"])) if meta["incident_pdg"] else incident_pdg
        pdg_class_int  = _pdg_class(meta["incident_pdg"] or actual_pdg_int)
        direction      = np.array(
            [meta["direction_x"], meta["direction_y"], meta["direction_z"]],
            dtype=np.float32)
        p_energy  = float(meta["incident_energy"])
        shower_id = meta["shower_id"] or os.path.basename(pq_path).replace("_hits.parquet", "")

        df = tbl.to_pandas()

        if do_dedup:
            df, n_rm = _drop_near_duplicates(
                df, np, pd,
                time_tol=dedup_time_tol,
                energy_rel_tol=dedup_energy_rel_tol,
                xy_tol=dedup_xy_tol,
                verbose=(verbose and i < 3),
            )
            result["n_dedup_removed"] += n_rm

        pdg_arr = df["pdg"].to_numpy()

        for pkey in active:
            cfg    = PARTICLE_CONFIGS[pkey]
            mask   = np.isin(pdg_arr, cfg["pdg_values"])
            df_sub = df[mask]
            w      = writers[pkey]

            # Two-zone clustering → combine → truncate → sort
            arr = _cluster_two_zones(
                df_sub,
                dx_near       = w["dx_near"],
                dx_far        = w["dx_far"],
                zone_boundary = zone_boundary,
                np            = np,
                include_time  = include_time,
            )
            arr = _truncate_and_sort(arr, w["nmax"], np,
                                     guarantee_all_planes=guarantee_all_planes)

            w["writer"].append(
                arr        = arr,
                direction  = direction,
                energy     = p_energy,
                pdg_class  = pdg_class_int,
                actual_pdg = actual_pdg_int,
                shower_id  = shower_id,
            )

        result["n_processed"] += 1

        if verbose and ((i + 1) % 20 == 0 or i + 1 == len(chunk_paths)):
            elapsed = time.time() - t_start
            rate    = (i + 1) / elapsed if elapsed > 0 else 0.0
            print(f"  [{i+1}/{len(chunk_paths)}] processed "
                  f"({rate:.2f} showers/s, dedup_removed={result['n_dedup_removed']})")

    for pkey, w in writers.items():
        w["writer"].close()
        size = os.path.getsize(w["h5_path"])
        result["h5_total_bytes"] += size
        result["h5_files"].append(w["h5_path"])
        if verbose:
            print(f"  wrote {os.path.basename(w['h5_path'])}: "
                  f"{w['writer'].count} showers, {_human_bytes(size)}")

    result["elapsed_s"] = time.time() - t_start
    if result["n_processed"] == 0:
        result["status"]  = "skipped"
        result["message"] = "no parquets processed"
    return result


# =============================================================================
# CLI
# =============================================================================

def _read_chunk_list(path: str) -> list[str]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Post-process hit parquets → 3 H5 files using two-zone centroid clustering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--chunk-list",    required=True)
    parser.add_argument("--incident-pdg",  type=int, required=True)
    parser.add_argument("--chunk-id",      type=int, required=True)
    parser.add_argument("--output-dir",    required=True)

    # Per-particle dx (near and far) and nmax
    for pkey, cfg in PARTICLE_CONFIGS.items():
        parser.add_argument(f"--{pkey}-dx-near", type=float,
                            default=cfg["default_dx_near"],
                            help=f"Near-zone cell size for {pkey} (cheb < zone-boundary).")
        parser.add_argument(f"--{pkey}-dx-far",  type=float,
                            default=cfg["default_dx_far"],
                            help=f"Far-zone cell size for {pkey} (cheb >= zone-boundary).")
        parser.add_argument(f"--{pkey}-nmax",    type=int,
                            default=cfg["default_nmax"],
                            help=f"Max centroids kept for {pkey}.")

    parser.add_argument("--zone-boundary", type=float, default=DEFAULT_ZONE_BOUNDARY,
                        help="Chebyshev distance threshold in metres (default 10 000 m).")
    parser.add_argument("--guarantee-all-planes", action="store_true",
                        help="Guarantee every plane has at least one centroid in the "
                             "output. One seed (highest-energy centroid) per plane is "
                             "reserved first; the remaining Nmax budget is then filled "
                             "by the globally highest-energy non-seed centroids. "
                             "Without this flag, a simple global top-Nmax cut is used "
                             "and low-energy planes may get zero representation.")
    parser.add_argument("--no-time",  action="store_true",
                        help="Exclude time feature (4 features instead of 5).")
    parser.add_argument("--particles", nargs="+", default=None,
                        choices=list(PARTICLE_CONFIGS.keys()))
    parser.add_argument("--no-dedup", action="store_true")
    parser.add_argument("--dedup-time-tol",        type=float, default=1e-15)
    parser.add_argument("--dedup-energy-rel-tol",  type=float, default=1e-6)
    parser.add_argument("--dedup-xy-tol",          type=float, default=1e-3)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.chunk_list):
        print(f"ERROR: chunk-list not found: {args.chunk_list}", file=sys.stderr)
        sys.exit(1)

    chunk_paths = _read_chunk_list(args.chunk_list)
    if not chunk_paths:
        print(f"ERROR: chunk-list is empty: {args.chunk_list}", file=sys.stderr)
        sys.exit(1)

    dx_near_per_particle = {
        "electrons": args.electrons_dx_near,
        "muons":     args.muons_dx_near,
        "photons":   args.photons_dx_near,
    }
    dx_far_per_particle = {
        "electrons": args.electrons_dx_far,
        "muons":     args.muons_dx_far,
        "photons":   args.photons_dx_far,
    }
    nmax_per_particle = {
        "electrons": args.electrons_nmax,
        "muons":     args.muons_nmax,
        "photons":   args.photons_nmax,
    }

    print(f"Chunk list    : {args.chunk_list}")
    print(f"N parquets    : {len(chunk_paths)}")
    print(f"Incident PDG  : {args.incident_pdg}")
    print(f"Chunk id      : {args.chunk_id:04d}")
    print(f"Output dir    : {args.output_dir}/pdg_{args.incident_pdg}/")
    print(f"Zone boundary : {args.zone_boundary:.0f} m")
    print(f"All-plane guarantee: {'ON' if args.guarantee_all_planes else 'OFF'} "
          f"(--guarantee-all-planes)")
    print(f"Particles     : {args.particles or list(PARTICLE_CONFIGS)}")
    for pkey in (args.particles or list(PARTICLE_CONFIGS)):
        print(f"  {pkey:<10}: dx_near={dx_near_per_particle[pkey]} m  "
              f"dx_far={dx_far_per_particle[pkey]} m  "
              f"nmax={nmax_per_particle[pkey]}")
    print(f"Dedup         : {'OFF' if args.no_dedup else 'ON'} "
          f"(time<{args.dedup_time_tol:.1e}s, dE/E<{args.dedup_energy_rel_tol:.1e}, "
          f"dxy<{args.dedup_xy_tol:.1e}m)")
    print()

    result = process_chunk(
        chunk_paths          = chunk_paths,
        output_dir           = args.output_dir,
        incident_pdg         = args.incident_pdg,
        chunk_id             = args.chunk_id,
        nmax_per_particle    = nmax_per_particle,
        dx_near_per_particle = dx_near_per_particle,
        dx_far_per_particle  = dx_far_per_particle,
        zone_boundary        = args.zone_boundary,
        guarantee_all_planes = args.guarantee_all_planes,
        include_time         = not args.no_time,
        dedup_time_tol       = args.dedup_time_tol,
        dedup_energy_rel_tol = args.dedup_energy_rel_tol,
        dedup_xy_tol         = args.dedup_xy_tol,
        do_dedup             = not args.no_dedup,
        particles            = args.particles,
        verbose              = not args.quiet,
    )

    print()
    print(f"Status        : {result['status']}")
    print(f"Processed     : {result['n_processed']}/{result['n_parquets']} showers")
    print(f"Skipped       : {result['n_skipped']}")
    print(f"Dedup removed : {result['n_dedup_removed']} hits")
    print(f"H5 written    : {_human_bytes(result['h5_total_bytes'])} "
          f"({len(result['h5_files'])} files)")
    print(f"Elapsed       : {result['elapsed_s']:.1f} s")
    if result["message"]:
        print(f"Detail        : {result['message']}")

    sys.exit(0 if result["status"] in ("ok", "partial") else 1)


if __name__ == "__main__":
    main()