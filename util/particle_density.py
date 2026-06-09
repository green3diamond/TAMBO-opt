#!/usr/bin/env python3
"""
Compute particle density in square (dx × dx) cells per detector plane.

For each shower and each secondary particle type (electrons, muons, photons):
  1. For every plane that has hits, anchor a square grid at (0, 0) in detector
     coordinates.  Cell column j covers [j·dx, (j+1)·dx), row k covers
     [k·dx, (k+1)·dx).  Negative coordinates get negative (but valid) indices,
     so the grid extends symmetrically around the origin.
  2. Assign each hit to exactly one cell via  ix = floor(x / dx),
     iy = floor(y / dx).  Boundary hits (x == j·dx exactly) go to cell j
     (left-inclusive, right-exclusive).  No hit is ever dropped or
     double-counted.
  3. Count the number of hits (particles) in each occupied cell.
     That count IS the particle density for that cell.
  4. Write one CSV row per occupied cell, recording its plane, grid indices,
     centre coordinates, and hit count.

Output CSV — written to <out>/pdg_<PDG>/:
    chunk_<XXXX>_density.csv

Columns:
    shower_id        – unique shower identifier
    incident_pdg     – PDG of the primary particle
    actual_pdg       – PDG recorded in the event CSV (may differ from incident)
    incident_energy  – primary particle energy [GeV]
    particle_type    – "electrons" | "muons" | "photons"
    dx               – cell side length [m]
    plane_index      – detector plane index
    cell_ix          – integer grid column  (floor(x / dx))
    cell_iy          – integer grid row     (floor(y / dx))
    cell_x_centre    – x coordinate of cell centre  [(ix + 0.5) * dx]
    cell_y_centre    – y coordinate of cell centre  [(iy + 0.5) * dx]
    n_hits           – number of particles in this cell (particle density)
    n_hits_plane     – total hits of this particle type on this plane
    n_cells_plane    – number of occupied cells on this plane

Row count per shower:
    sum over particle_types of (occupied cells across all planes)

Usage
-----
python particle_density.py \
    --chunk-list   registry_pdg_11.txt \
    --incident-pdg 11 \
    --chunk-id     0 \
    --output-dir   /path/to/output \
    --dx           100

Add --max-showers N [--random [--seed S]] to subsample.
Add --particles electrons muons  to restrict particle types.
Add --no-dedup to skip near-duplicate removal.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from typing import Optional


# =============================================================================
# Particle type configuration
# preprocess_showers.py collapses sign so parquet pdg ∈ {11, 13, 22}.
# =============================================================================

PARTICLE_CONFIGS = {
    "electrons": {"pdg_values": [11],  "label": "e± (pdg 11)"},
    "muons":     {"pdg_values": [13],  "label": "μ± (pdg 13)"},
    "photons":   {"pdg_values": [22],  "label": "γ (pdg 22)"},
}

DEFAULT_DX = 100.0   # metres


# =============================================================================
# Lazy imports
# =============================================================================

def _lazy_imports():
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    return np, pd, pq


# =============================================================================
# Event-CSV helpers  (identical to original scripts)
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
        "shower_id":       str(row.get("shower_id", "")),
        "incident_energy": _float_or(0.0, row.get("incident_energy")),
        "incident_pdg":    _float_or(0.0, row.get("incident_pdg")),
    }


# =============================================================================
# Near-duplicate removal  (identical to original scripts)
# =============================================================================

def _drop_near_duplicates(df, np, pd,
                          time_tol: float, energy_rel_tol: float,
                          xy_tol: float, verbose: bool = False):
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
        print(f"      dedup: removed {n_removed}/{n} near-duplicate hits")
    return df_sorted.loc[~dup_mask].reset_index(drop=True), n_removed


# =============================================================================
# Density computation — concentric square rings centred at (0, 0) per plane
#
# Ring k is the region strictly inside the square of half-side (k+1)·dx
# but outside the square of half-side k·dx:
#
#   ring 1:  |x| < 1·dx  AND  |y| < 1·dx        →  -100 to +100  (blue)
#   ring 2:  |x| < 2·dx  AND  |y| < 2·dx        →  -200 to +200  (red)
#            but NOT already inside ring 1
#
# Equivalently, a hit belongs to ring k (k = 1, 2, 3, ...) when:
#   max(|x|, |y|) ∈ [ (k-1)·dx, k·dx )
#
# i.e.  k = ceil( max(|x|, |y|) / dx )   with k=1 for the innermost ring.
#
# Ring 0 is a degenerate point (the origin itself) — in practice any hit
# exactly at (0,0) lands in ring 1 because ceil(0/dx)=0 → we clamp to 1.
#
# CSV columns written per ring:
#   ring_k        – ring index (1 = innermost, 2 = next, …)
#   half_side_in  – inner half-side = (k-1)·dx   [ring 1: 0]
#   half_side_out – outer half-side = k·dx
#   n_hits        – hits whose max(|x|,|y|) ∈ [half_side_in, half_side_out)
#   n_hits_cumulative – hits inside the full square of half-side k·dx
# =============================================================================

def _concentric_density_per_plane(xy, dx, np):
    """
    Assign hits to concentric square rings centred at (0,0).

    Ring k (k=1,2,3,...) covers:
        (k-1)·dx  <=  max(|x|, |y|)  <  k·dx

    Parameters
    ----------
    xy : (N, 2) float64 array of (x, y) hit positions
    dx : float, ring separation [same units as x, y]
    np : numpy module

    Returns
    -------
    list of dicts, one per occupied ring, sorted by ring_k ascending:
        ring_k           – ring index (1 = innermost)
        half_side_in     – inner boundary = (k-1) * dx
        half_side_out    – outer boundary = k * dx
        n_hits           – hits in this ring only
        n_hits_cumulative– hits inside full square of half-side k*dx
    """
    if len(xy) == 0:
        return []

    xy = np.asarray(xy, dtype=np.float64)

    # Chebyshev distance from origin = max(|x|, |y|)
    cheb = np.maximum(np.abs(xy[:, 0]), np.abs(xy[:, 1]))

    # Ring index: ceil(cheb / dx), clamped so origin → ring 1
    k_float = cheb / dx
    k_arr   = np.ceil(k_float).astype(np.int64)
    k_arr   = np.where(k_arr < 1, 1, k_arr)   # clamp origin to ring 1

    unique_k, counts = np.unique(k_arr, return_counts=True)

    # Build cumulative counts: n_hits_cumulative for ring k
    # = all hits with cheb < k*dx = sum of counts for rings 1..k
    k_to_count = dict(zip(unique_k.tolist(), counts.tolist()))
    all_k = list(range(1, int(unique_k.max()) + 1)) if len(unique_k) > 0 else []
    cumulative = 0
    rows = []
    for k in all_k:
        ring_count = k_to_count.get(k, 0)
        cumulative += ring_count
        if ring_count > 0:          # only write occupied rings
            rows.append({
                "ring_k":            k,
                "half_side_in":      (k - 1) * dx,
                "half_side_out":     k * dx,
                "n_hits":            ring_count,
                "n_hits_cumulative": cumulative,
            })

    return rows


def _concentric_density_for_type(df_sub, dx, np):
    """
    Run concentric ring density for one particle type across all planes.

    Returns list of dicts, one per (plane × occupied ring):
        plane_index, ring_k, half_side_in, half_side_out,
        n_hits, n_hits_cumulative,
        n_hits_plane, n_rings_plane
    """
    if len(df_sub) == 0:
        return []

    pos   = df_sub[["x", "y", "plane_index"]].to_numpy(dtype=np.float64)
    plane = pos[:, 2].astype(np.int32)

    all_rows = []
    for p_val in np.unique(plane):
        mask  = plane == p_val
        xy_p  = pos[mask, :2]
        rings = _concentric_density_per_plane(xy_p, dx, np)

        n_hits_plane  = int(mask.sum())
        n_rings_plane = len(rings)

        for r in rings:
            r["plane_index"]  = int(p_val)
            r["n_hits_plane"] = n_hits_plane
            r["n_rings_plane"]= n_rings_plane
            all_rows.append(r)

    return all_rows


# =============================================================================
# CSV columns
# =============================================================================

CSV_COLUMNS = [
    "shower_id",
    "incident_pdg",
    "actual_pdg",
    "incident_energy",
    "particle_type",
    "dx",
    "plane_index",
    "ring_k",             # 1 = innermost square [-dx,+dx]×[-dx,+dx]
    "half_side_in",       # inner boundary = (k-1)*dx   [0 for ring 1]
    "half_side_out",      # outer boundary = k*dx
    "n_hits",             # hits in this ring only
    "n_hits_cumulative",  # hits inside full square out to ring k
    "n_hits_plane",       # total hits of this type on this plane
    "n_rings_plane",      # number of occupied rings on this plane
]


# =============================================================================
# Core: process one chunk → one CSV
# =============================================================================

def _read_chunk_list(path: str) -> list[str]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def process_chunk(
    chunk_paths: list[str],
    output_dir: str,
    incident_pdg: int,
    chunk_id: int,
    dx: float,
    particles: Optional[list[str]] = None,
    max_showers: Optional[int] = None,
    random_sample: bool = False,
    seed: Optional[int] = None,
    do_dedup: bool = True,
    dedup_time_tol: float = 1e-15,
    dedup_energy_rel_tol: float = 1e-6,
    dedup_xy_tol: float = 1e-3,
    verbose: bool = True,
) -> dict:
    """Process a list of parquet files → one density CSV in output_dir/pdg_<N>/."""
    np, pd, pq = _lazy_imports()

    # Validate particle selection.
    active = list(PARTICLE_CONFIGS.keys())
    if particles:
        bad = [p for p in particles if p not in PARTICLE_CONFIGS]
        if bad:
            raise ValueError(f"Unknown particle types: {bad}. "
                             f"Choose from {list(PARTICLE_CONFIGS)}.")
        active = [p for p in PARTICLE_CONFIGS if p in particles]
        if not active:
            raise ValueError("Empty particle selection after filtering.")

    if dx <= 0:
        raise ValueError(f"dx must be positive, got {dx}")

    subdir   = os.path.join(output_dir, f"pdg_{incident_pdg}")
    os.makedirs(subdir, exist_ok=True)
    csv_path = os.path.join(subdir, f"chunk_{chunk_id:04d}_density.csv")

    result = {
        "incident_pdg":    incident_pdg,
        "chunk_id":        chunk_id,
        "n_parquets":      len(chunk_paths),
        "n_processed":     0,
        "n_skipped":       0,
        "n_dedup_removed": 0,
        "particles":       active,
        "dx":              dx,
        "csv_path":        csv_path,
        "n_rows_written":  0,
        "status":          "ok",
        "message":         "",
        "elapsed_s":       0.0,
    }
    t_start = time.time()
    required_cols = ["x", "y", "pdg", "time", "kinetic_energy", "plane_index"]

    # Subsample paths if requested.
    if max_showers is not None and max_showers < len(chunk_paths):
        if random_sample:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(chunk_paths), size=max_showers, replace=False)
            idx.sort()
            paths_to_use = [chunk_paths[i] for i in idx]
        else:
            paths_to_use = chunk_paths[:max_showers]
    else:
        paths_to_use = list(chunk_paths)
    result["n_selected"] = len(paths_to_use)

    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for i, pq_path in enumerate(paths_to_use):
            if not os.path.exists(pq_path):
                result["n_skipped"] += 1
                if verbose:
                    print(f"  [{i+1}/{len(paths_to_use)}] MISSING: {pq_path}")
                continue

            try:
                tbl = pq.read_table(pq_path, columns=required_cols)
            except Exception as exc:
                result["n_skipped"] += 1
                if verbose:
                    print(f"  [{i+1}/{len(paths_to_use)}] READ ERROR {pq_path}: {exc}")
                continue

            if len(tbl) == 0:
                result["n_skipped"] += 1
                continue

            # Event metadata.
            event_csv = _event_csv_path_for(pq_path)
            meta      = _extract_event_meta(event_csv)
            actual_pdg_int = (int(round(meta["incident_pdg"]))
                              if meta["incident_pdg"] else incident_pdg)
            p_energy  = float(meta["incident_energy"])
            shower_id = (meta["shower_id"]
                         or os.path.basename(pq_path).replace("_hits.parquet", ""))

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

                ring_rows = _concentric_density_for_type(df_sub, dx=dx, np=np)

                for r in ring_rows:
                    writer.writerow({
                        "shower_id":           shower_id,
                        "incident_pdg":        incident_pdg,
                        "actual_pdg":          actual_pdg_int,
                        "incident_energy":     p_energy,
                        "particle_type":       pkey,
                        "dx":                  dx,
                        "plane_index":         r["plane_index"],
                        "ring_k":              r["ring_k"],
                        "half_side_in":        r["half_side_in"],
                        "half_side_out":       r["half_side_out"],
                        "n_hits":              r["n_hits"],
                        "n_hits_cumulative":   r["n_hits_cumulative"],
                        "n_hits_plane":        r["n_hits_plane"],
                        "n_rings_plane":       r["n_rings_plane"],
                    })
                    result["n_rows_written"] += 1

            result["n_processed"] += 1

            if verbose and ((i + 1) % 50 == 0 or i + 1 == len(paths_to_use)):
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0.0
                print(f"  [{i+1}/{len(paths_to_use)}] processed "
                      f"({rate:.2f} showers/s, "
                      f"dedup_removed={result['n_dedup_removed']})")

    result["elapsed_s"] = time.time() - t_start
    if result["n_processed"] == 0:
        result["status"]  = "skipped"
        result["message"] = "no parquets processed"
    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute particle density in square (dx × dx) cells per plane.\n"
            "Grid anchored at (0,0); one CSV row per occupied cell per shower."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--chunk-list",    required=True,
                        help="Text file with one parquet path per line.")
    parser.add_argument("--incident-pdg",  type=int, required=True,
                        help="Incident PDG for this chunk (used for subdir name).")
    parser.add_argument("--chunk-id",      type=int, required=True,
                        help="Zero-padded chunk id (used in output filename).")
    parser.add_argument("--output-dir",    required=True,
                        help="Root output dir. CSV goes to <out>/pdg_<PDG>/.")
    parser.add_argument("--dx",            type=float, default=DEFAULT_DX,
                        help="Cell side length in metres. "
                             "Grid starts at (0,0); cells at 0, dx, 2dx, …")
    parser.add_argument("--particles",     nargs="+", default=None,
                        choices=list(PARTICLE_CONFIGS.keys()),
                        help="Subset of secondary types (default: all three).")

    # Subsampling
    parser.add_argument("--max-showers",   type=int, default=None,
                        help="Cap on showers consumed from chunk-list.")
    parser.add_argument("--random",        action="store_true",
                        help="With --max-showers: random sample instead of first N.")
    parser.add_argument("--seed",          type=int, default=None,
                        help="Seed for --random sampling.")

    # Dedup knobs
    parser.add_argument("--no-dedup",              action="store_true")
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

    if args.random and args.max_showers is None:
        print("WARNING: --random has no effect without --max-showers; ignored.",
              file=sys.stderr)

    print(f"Chunk list    : {args.chunk_list}")
    print(f"N parquets    : {len(chunk_paths)}"
          f"  cap={args.max_showers or 'none'}"
          f"{'  RANDOM' if args.random and args.max_showers else ''}"
          f"{f'  seed={args.seed}' if args.random and args.max_showers and args.seed is not None else ''}")
    print(f"Incident PDG  : {args.incident_pdg}")
    print(f"Chunk id      : {args.chunk_id:04d}")
    print(f"Output dir    : {args.output_dir}/pdg_{args.incident_pdg}/")
    print(f"Particles     : {args.particles or list(PARTICLE_CONFIGS)}")
    print(f"dx            : {args.dx} m  (square grid anchored at (0,0) per plane)")
    print(f"Dedup         : {'OFF' if args.no_dedup else 'ON'}"
          f"  (time<{args.dedup_time_tol:.1e}s,"
          f"  dE/E<{args.dedup_energy_rel_tol:.1e},"
          f"  dxy<{args.dedup_xy_tol:.1e}m)")
    print()

    result = process_chunk(
        chunk_paths           = chunk_paths,
        output_dir            = args.output_dir,
        incident_pdg          = args.incident_pdg,
        chunk_id              = args.chunk_id,
        dx                    = args.dx,
        particles             = args.particles,
        max_showers           = args.max_showers,
        random_sample         = args.random,
        seed                  = args.seed,
        do_dedup              = not args.no_dedup,
        dedup_time_tol        = args.dedup_time_tol,
        dedup_energy_rel_tol  = args.dedup_energy_rel_tol,
        dedup_xy_tol          = args.dedup_xy_tol,
        verbose               = not args.quiet,
    )

    print()
    print(f"Status        : {result['status']}")
    print(f"Selected      : {result.get('n_selected', result['n_parquets'])}"
          f"/{result['n_parquets']} showers")
    print(f"Processed     : {result['n_processed']} showers")
    print(f"Skipped       : {result['n_skipped']}")
    print(f"Dedup removed : {result['n_dedup_removed']} hits")
    print(f"CSV rows      : {result['n_rows_written']}  "
          f"(one row per occupied cell per shower)")
    print(f"CSV path      : {result['csv_path']}")
    print(f"Elapsed       : {result['elapsed_s']:.1f} s")
    if result["message"]:
        print(f"Detail        : {result['message']}")

    sys.exit(0 if result["status"] in ("ok", "partial") else 1)


if __name__ == "__main__":
    main()