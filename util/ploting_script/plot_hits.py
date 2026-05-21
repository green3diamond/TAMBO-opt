#!/usr/bin/env python3
"""
Quick plot of raw hits from a single shower parquet.

- One subplot per plane (xy scatter).
- Single secondary type (electrons / muons / photons).
- Marker size scales with log(kinetic_energy).
- Per-plane shower-axis center is projected from the sibling _event.csv
  using (incident_x/y/z, direction_x/y/z, z_depth_start, z_depth_step)
  and drawn as a red '+' on each subplot.

Usage:
    python plot_hits.py <shower>_hits.parquet --particle electrons
    python plot_hits.py <shower>_hits.parquet --particle muons --save hits.png
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt


PDG_MAP = {
    "electrons": [11, -11],
    "muons":     [13, -13],
    "photons":   [22],
}


def _event_csv_for(parquet_path: str) -> str:
    if parquet_path.endswith("_hits.parquet"):
        return parquet_path[: -len("_hits.parquet")] + "_event.csv"
    root, _ = os.path.splitext(parquet_path)
    return root + "_event.csv"


def _load_event_meta(csv_path: str) -> dict | None:
    if not os.path.exists(csv_path):
        return None
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return row
    return None


def _f(meta: dict, key: str, default: float = 0.0) -> float:
    try:
        v = float(meta.get(key, default))
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _plane_centers(meta: dict, planes: list[int]) -> dict[int, tuple[float, float]]:
    """Project shower axis onto each plane's z to get (cx, cy) per plane."""
    ix, iy, iz = _f(meta, "incident_x"),  _f(meta, "incident_y"),  _f(meta, "incident_z")
    dx, dy, dz = _f(meta, "direction_x"), _f(meta, "direction_y"), _f(meta, "direction_z", 1.0)
    z0 = _f(meta, "z_depth_start", 0.0)
    dz_step = _f(meta, "z_depth_step", 1.0)

    out: dict[int, tuple[float, float]] = {}
    if abs(dz) < 1e-12:
        # axis parallel to plane — fall back to incident xy on every plane
        for p in planes:
            out[int(p)] = (ix, iy)
        return out

    for p in planes:
        z_p = z0 + int(p) * dz_step
        t   = (z_p - iz) / dz
        out[int(p)] = (ix + t * dx, iy + t * dy)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet", help="Path to <shower>_hits.parquet")
    ap.add_argument("--particle", choices=list(PDG_MAP), default="electrons")
    ap.add_argument("--save", default=None, help="If given, save fig to this path instead of showing.")
    ap.add_argument("--xlim", type=float, default=None,
                    help="Symmetric xy limit. Default: auto from hit extent.")
    ap.add_argument("--ncols", type=int, default=4, help="Subplot columns.")
    ap.add_argument("--center-on-axis", action="store_true",
                    help="Center each subplot on the per-plane projected axis "
                         "(plot range = ±xlim around that center).")
    args = ap.parse_args()

    if not os.path.exists(args.parquet):
        print(f"ERROR: not found: {args.parquet}", file=sys.stderr)
        sys.exit(1)

    cols = ["x", "y", "plane_index", "kinetic_energy", "pdg"]
    df = pq.read_table(args.parquet, columns=cols).to_pandas()

    df = df[df["pdg"].isin(PDG_MAP[args.particle])]
    if len(df) == 0:
        print(f"No hits for particle={args.particle} in {args.parquet}")
        sys.exit(0)

    # event metadata for axis projection
    meta = _load_event_meta(_event_csv_for(args.parquet))
    if meta is None:
        print(f"WARNING: event csv not found next to {args.parquet}; centers will be (0,0)")
        meta = {}

    planes = sorted(df["plane_index"].unique().tolist())
    centers = _plane_centers(meta, planes)

    # marker size scaling (global log range)
    e_all = np.maximum(df["kinetic_energy"].to_numpy(), 1e-12)
    e_log = np.log10(e_all)
    e_lo, e_span = float(e_log.min()), max(float(e_log.max() - e_log.min()), 1e-9)

    # axis-limit strategy
    if args.center_on_axis:
        # half-window around each plane's projected center
        if args.xlim is None:
            # default half-window: spread of hits relative to their plane center
            dx_all, dy_all = [], []
            for p in planes:
                cx, cy = centers[int(p)]
                sub = df[df["plane_index"] == p]
                dx_all.append(np.abs(sub["x"].to_numpy() - cx))
                dy_all.append(np.abs(sub["y"].to_numpy() - cy))
            half = float(max(np.concatenate(dx_all).max(), np.concatenate(dy_all).max()) * 1.05)
        else:
            half = args.xlim
        share_xy = False  # different absolute coords per plane
    else:
        if args.xlim is None:
            half = float(max(abs(df["x"]).max(), abs(df["y"]).max()) * 1.05)
        else:
            half = args.xlim
        share_xy = True

    n     = len(planes)
    ncols = max(1, args.ncols)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.6 * ncols, 2.6 * nrows),
        squeeze=False,
        sharex=share_xy, sharey=share_xy,
    )

    for i, p in enumerate(planes):
        ax = axes[i // ncols][i % ncols]
        sub = df[df["plane_index"] == p]
        x = sub["x"].to_numpy()
        y = sub["y"].to_numpy()
        e = np.maximum(sub["kinetic_energy"].to_numpy(), 1e-12)
        s = 2 + 30 * (np.log10(e) - e_lo) / e_span

        ax.scatter(x, y, s=s, c="C0", alpha=0.5, edgecolors="none")

        cx, cy = centers[int(p)]
        ax.plot(cx, cy, marker="+", color="red", markersize=10, mew=1.5, zorder=5)

        if args.center_on_axis:
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)
        else:
            ax.set_xlim(-half, half)
            ax.set_ylim(-half, half)

        ax.set_title(f"plane {int(p)}  (n={len(sub)})", fontsize=8)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=7)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    sid = meta.get("shower_id", "") if meta else ""
    fig.suptitle(
        f"{os.path.basename(args.parquet)}  |  {args.particle}  |  "
        f"hits={len(df)}  |  shower_id={sid}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()