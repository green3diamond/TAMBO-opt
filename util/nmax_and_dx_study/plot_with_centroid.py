#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot energy-ignored histograms from the long-format CSVs produced by
postprocess_showers_csv.py / submit_energy_ignored.py.

Layout per figure (one figure per SECONDARY particle type):

    Row 1: combined e+ + e- + pi0     (incident_pdg in {-11, 11, 111})
    Row 2: combined pi+ + pi-          (incident_pdg in {-211, 211})
    Cols : one per (dx_near, dx_far) pair

Each panel overlays one step histogram per Nmax in --nmax-sweep
(or per-secondary: --electrons-nmax / --muons-nmax / --photons-nmax).

Input  : <csv-root>/pdg_<PDG>/chunk_*_energy_ignored.csv  (long format)
Output : <plot-dir>/<secondary>_ignored_energy.png

The CSV contains a "zone" column ("near" / "far") and "zone_boundary" (metres).
Use --zone to select which zone(s) to plot (default: both, one figure each).

python /n/home04/hhanif/TAMBO-opt/nmax_and_dx_study/plot_with_centroid.py \
    --csv-root /n/home04/hhanif/TAMBO-opt/results/nmax_study_with_centroids \
    --plot-dir /n/home04/hhanif/TAMBO-opt/results/nmax_study_with_centroids/plots \
    --dx-sweep           5  8  10  15  20 \
    --dx-after-10k-sweep 10 16  20  30  40 \
    --electrons-nmax 4000 6000 \
    --muons-nmax     10000 14000 18000 20000 25000 \
    --photons-nmax   6000 10000 14000


python /n/home04/hhanif/TAMBO-opt/nmax_and_dx_study/plot_with_centroid.py \
    --csv-root /n/home04/hhanif/TAMBO-opt/results/nmax_study_with_centroids_2 \
    --plot-dir /n/home04/hhanif/TAMBO-opt/results/nmax_study_with_centroids_2/plots3 \
    --dx-sweep           5  8  10  15  20 \
    --dx-after-10k-sweep 10 16  20  30  40 \
    --muons-nmax  24000    260000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


ROWS = [
    {
        "label":     r"$e^{\pm}$ + $\pi^{0}$",
        "pdgs":      [-11, 11, 111],
        "color":     "#4a90d9",
        "facecolor": "#eaf1fb",
    },
    {
        "label":     r"$\pi^{\pm}$",
        "pdgs":      [-211, 211],
        "color":     "#d94a4a",
        "facecolor": "#fbeaea",
    },
]

DEFAULT_DX_NEAR     = [5.0,  8.0,  10.0]   # near-zone dx values
DEFAULT_DX_FAR      = [10.0, 16.0, 20.0]   # paired far-zone dx values
DEFAULT_NMAX_SWEEP  = [2000, 4000, 6000, 10000, 14000, 18000, 20000]
DEFAULT_SECONDARIES = ["electrons", "muons", "photons"]
DEFAULT_ZONES       = ["near", "far"]   # "near": cheb < zone_boundary, "far": cheb >= zone_boundary


def load_pdg(csv_root: Path, pdg: int, zone: str | None = None) -> pd.DataFrame:
    """Concatenate all chunk CSVs for one incident PDG, optionally filtered
    to one zone ("near" or "far"). Empty DF if none found."""
    pdg_dir = csv_root / f"pdg_{pdg}"
    if not pdg_dir.is_dir():
        return pd.DataFrame()
    files = sorted(pdg_dir.glob("chunk_*_energy_ignored.csv"))
    if not files:
        return pd.DataFrame()

    parts = []
    for f in files:
        try:
            parts.append(pd.read_csv(f))
        except Exception as e:
            print(f"  WARN: failed to read {f}: {e}", file=sys.stderr)
    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)
    df["dx"]                  = df["dx"].astype(float)
    df["nmax"]                = df["nmax"].astype(int)
    df["energy_ignored_frac"] = df["energy_ignored_frac"].astype(float)

    # Handle CSVs that pre-date the zone column (treat as "near" by default).
    if "zone" not in df.columns:
        df["zone"]          = "near"
        df["zone_boundary"] = 10_000.0

    if zone is not None:
        df = df[df["zone"] == zone].reset_index(drop=True)
    return df


def load_rows(csv_root: Path, zone: str | None = None) -> list[pd.DataFrame]:
    """One concatenated DataFrame per row in ROWS (combining its pdgs),
    optionally filtered to one zone."""
    out = []
    for r in ROWS:
        parts = []
        for pdg in r["pdgs"]:
            df = load_pdg(csv_root, pdg, zone=zone)
            if not df.empty:
                parts.append(df)
        if parts:
            combined = pd.concat(parts, ignore_index=True)
        else:
            combined = pd.DataFrame()
            print(f"  WARN: no data for row {r['label']} "
                  f"(pdgs={r['pdgs']})", file=sys.stderr)
        out.append(combined)
    return out


def _nmax_color_map(nmax_sweep):
    palette = (plt.cm.tab10.colors + plt.cm.Set2.colors + plt.cm.Dark2.colors)
    return {n: palette[i % len(palette)] for i, n in enumerate(sorted(nmax_sweep))}


def _row_xmax(df, secondary, dx_pairs, nmax_sweep):
    """df contains both zones. dx_pairs = list of (dx_near, dx_far) tuples."""
    if df.empty:
        return 100.0
    near_dxs = [p[0] for p in dx_pairs]
    far_dxs  = [p[1] for p in dx_pairs]
    sub = df[
        (df["particle_type"] == secondary)
        & df["nmax"].isin(nmax_sweep)
        & (
            ((df["zone"] == "near") & df["dx"].isin(near_dxs)) |
            ((df["zone"] == "far")  & df["dx"].isin(far_dxs))
        )
    ]
    if sub.empty:
        return 100.0
    x = sub["energy_ignored_frac"].max() * 100.0 + 5.0
    return float(np.ceil(x / 5.0) * 5.0)


def plot_secondary(
    secondary: str,
    rows_df: list[pd.DataFrame],      # each df has BOTH zones (no pre-filtering)
    dx_pairs: list[tuple[float,float]],  # [(dx_near, dx_far), ...]
    nmax_sweep: list[int],
    plot_dir: Path,
    zone_boundary: float = 10_000.0,
    dpi: int = 150,
    bins: int = 8,
):
    """
    Each column corresponds to one (dx_near, dx_far) pair.
    For each pair the histogram is built by combining:
        near-zone rows where dx == dx_near
        far-zone  rows where dx == dx_far
    → one unified energy_ignored_frac distribution per Nmax.
    """
    n_rows = len(ROWS)
    n_cols = len(dx_pairs)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.6 * n_cols, 4.4 * n_rows),
        sharey=False, squeeze=False,
    )
    fig.suptitle(
        f"Energy Ignored — Secondary: {secondary.capitalize()}\n"
        f"Near zone dx ; Far zone dx  (boundary = {zone_boundary/1000:.0f} km)",
        fontsize=13, fontweight="bold", y=1.00,
    )

    color_for_nmax = _nmax_color_map(nmax_sweep)

    for row_idx, (row_cfg, df) in enumerate(zip(ROWS, rows_df)):
        xmax = _row_xmax(df, secondary, dx_pairs, nmax_sweep)

        for col_idx, (dx_near, dx_far) in enumerate(dx_pairs):
            ax = axes[row_idx][col_idx]
            if df.empty:
                ax.set_visible(False)
                continue

            # Combine near-zone rows (dx=dx_near) + far-zone rows (dx=dx_far)
            # into one unified distribution per Nmax.
            sub_near = df[
                (df["particle_type"] == secondary)
                & (df["zone"] == "near")
                & (df["dx"] == dx_near)
            ]
            sub_far = df[
                (df["particle_type"] == secondary)
                & (df["zone"] == "far")
                & (df["dx"] == dx_far)
            ]
            sub = pd.concat([sub_near, sub_far], ignore_index=True)

            for nmax in sorted(nmax_sweep):
                vals = sub.loc[sub["nmax"] == nmax, "energy_ignored_frac"].to_numpy()
                if vals.size == 0:
                    continue
                ax.hist(
                    100.0 * vals, bins=bins, histtype="step", linewidth=2.0,
                    color=color_for_nmax[nmax],
                    label=f"Nmax={nmax} (n={vals.size})",
                )

            ax.set_xlim(0, xmax)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="both", labelsize=8)
            ax.set_xlabel("Energy Ignored (%)", fontsize=9, fontweight="bold")
            ax.set_ylabel("Number of Showers", fontsize=9, fontweight="bold")
            ax.legend(loc="upper right", fontsize=7, frameon=True, framealpha=0.85)

            if row_idx == 0:
                ax.set_title(
                    f"dx near={dx_near:g} m ; dx far={dx_far:g} m",
                    fontsize=10, fontweight="bold", pad=4,
                )

            if col_idx == 0:
                ax.annotate(
                    f"Incident: {row_cfg['label']}",
                    xy=(0.0, 1.0), xycoords="axes fraction",
                    xytext=(0, 26 if row_idx == 0 else 12), textcoords="offset points",
                    fontsize=11, fontweight="bold", color="#222",
                    ha="left", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor=row_cfg["facecolor"],
                              edgecolor=row_cfg["color"],
                              linewidth=0.8, alpha=0.9),
                )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = plot_dir / f"{secondary}_ignored_energy.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv-root", required=True, type=Path,
                   help="Root containing pdg_<PDG>/chunk_*_energy_ignored.csv directories.")
    p.add_argument("--plot-dir", type=Path, default=None,
                   help="Output dir for PNGs (default: <csv-root>/plots).")
    p.add_argument("--dx-sweep", type=float, nargs="+", default=DEFAULT_DX_NEAR,
                   help="Near-zone dx values (paired 1-to-1 with --dx-after-10k-sweep).")
    p.add_argument("--dx-after-10k-sweep", type=float, nargs="+", default=DEFAULT_DX_FAR,
                   help="Far-zone dx values paired with --dx-sweep. "
                        "Must have the same length as --dx-sweep. "
                        "Pair i: (dx_sweep[i], dx_after_10k_sweep[i]).")
    p.add_argument("--nmax-sweep", type=int, nargs="+", default=DEFAULT_NMAX_SWEEP,
                   help="Default Nmax values to overlay (used for any secondary "
                        "without a per-secondary override).")
    p.add_argument("--electrons-nmax", type=int, nargs="+", default=None)
    p.add_argument("--muons-nmax",     type=int, nargs="+", default=None)
    p.add_argument("--photons-nmax",   type=int, nargs="+", default=None)
    p.add_argument("--secondaries", nargs="+", default=DEFAULT_SECONDARIES,
                   choices=["electrons", "muons", "photons"])
    p.add_argument("--bins", type=int, default=8)
    p.add_argument("--dpi",  type=int, default=150)

    args = p.parse_args()

    args.csv_root = args.csv_root.resolve()
    if args.plot_dir is None:
        args.plot_dir = args.csv_root / "plots"
    args.plot_dir = args.plot_dir.resolve()
    args.plot_dir.mkdir(parents=True, exist_ok=True)

    nmax_per_secondary = {
        "electrons": args.electrons_nmax if args.electrons_nmax else args.nmax_sweep,
        "muons":     args.muons_nmax     if args.muons_nmax     else args.nmax_sweep,
        "photons":   args.photons_nmax   if args.photons_nmax   else args.nmax_sweep,
    }

    sns.set_style("whitegrid")
    plt.rcParams["font.size"] = 10

    print(f"CSV root   : {args.csv_root}")
    print(f"Plot dir   : {args.plot_dir}")
    print(f"dx pairs   : {list(zip(args.dx_sweep, args.dx_after_10k_sweep))}")
    print(f"Default Nmax sweep : {args.nmax_sweep}")
    print("Per-secondary Nmax sweep:")
    for sec in args.secondaries:
        flag_used = (
            f"--{sec}-nmax" if {
                "electrons": args.electrons_nmax,
                "muons":     args.muons_nmax,
                "photons":   args.photons_nmax,
            }[sec] else "default"
        )
        print(f"  {sec:<10s}: {nmax_per_secondary[sec]}  ({flag_used})")
    print(f"Secondaries: {args.secondaries}")
    print()
    print("Row composition:")
    for r in ROWS:
        print(f"  {r['label']:<24s} <- pdgs {r['pdgs']}")
    print()

    # Validate pairs have equal length
    if len(args.dx_sweep) != len(args.dx_after_10k_sweep):
        print("ERROR: --dx-sweep and --dx-after-10k-sweep must have the same length.",
              file=sys.stderr)
        sys.exit(1)
    dx_pairs = list(zip(args.dx_sweep, args.dx_after_10k_sweep))

    # Load all data (both zones) once
    print("Loading data (both zones)...")
    rows_df = load_rows(args.csv_root, zone=None)
    for r, df in zip(ROWS, rows_df):
        print(f"  {r['label']:<24s}: {len(df):>9d} rows")
    if all(df.empty for df in rows_df):
        print("\nERROR: no data found. Check --csv-root.", file=sys.stderr)
        sys.exit(1)

    # Infer zone_boundary from data
    zone_boundary = 10_000.0
    for df in rows_df:
        if not df.empty and "zone_boundary" in df.columns:
            zone_boundary = float(df["zone_boundary"].iloc[0])
            break
    print(f"  Zone boundary: {zone_boundary:.0f} m")

    print("\nDrawing figures...")
    for sec in args.secondaries:
        plot_secondary(
            secondary     = sec,
            rows_df       = rows_df,
            dx_pairs      = dx_pairs,
            nmax_sweep    = nmax_per_secondary[sec],
            plot_dir      = args.plot_dir,
            zone_boundary = zone_boundary,
            dpi           = args.dpi,
            bins          = args.bins,
        )

    print(f"\nDone. {len(args.secondaries)} figure(s) in {args.plot_dir}")


if __name__ == "__main__":
    main()