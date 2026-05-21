#!/usr/bin/env python3
"""
Plot average particle density profile:  distance from origin  vs  mean n_hits

Reads all chunk_*_density.csv files (raw dx=100m rings) and REBINS them
into user-defined distance phases before plotting.

=============================================================================
CONFIGURATION  (edit the CONFIG block below)
=============================================================================

BIN_PHASES defines how the 100m raw rings are combined into plot bins.
Each phase is a dict with three keys:

    step   – bin width in metres  (must be a multiple of RAW_DX = 100 m)
    up_to  – upper boundary of this phase in metres
             use None for "all remaining hits"
    label  – short string used in axis tick labels for this phase

Default example — three phases:
    {"step":   500, "up_to":  10_000, "label": "500 m"},   # 20 bins
    {"step": 1_000, "up_to": 100_000, "label": "1 km"},    # 90 bins
    {"step":  None, "up_to":    None, "label": "remaining"},# 1 catch-all bin

This produces:
    bin  1:      0 –    500 m   (rings  1– 5  summed)
    bin  2:    500 –  1 000 m   (rings  6–10  summed)
    ...
    bin 20:  9500 – 10 000 m   (rings 96–100 summed)
    bin 21: 10000 – 11 000 m   (rings 101–110 summed)
    ...
    bin 110: 99000–100 000 m   (rings 991–1000 summed)
    bin 111: 100 000 m → ∞     (all remaining rings)

PDG_GROUPS controls which PDGs are combined into each plot panel.

PARTICLE_STYLES controls colours and labels per secondary type.

Usage
-----
python plot_density.py \
    --output-dir /path/to/results \
    --save-fig   density_profile.png

Add --log-y for log scale.
Add --raw-dx 100 if your CSVs used a different ring width.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# >>>  USER CONFIGURATION  <<<  edit anything in this block
# =============================================================================

# Raw ring width used when generating the CSVs (the --dx in particle_density.py)
RAW_DX = 100.0   # metres

# ---- Bin phases ----
# step  : bin width in metres (must be a multiple of RAW_DX). None = catch-all.
# up_to : outer boundary of this phase in metres.             None = catch-all.
# label : short label shown on x-axis ticks.
BIN_PHASES = [
    {"step":   500, "up_to":  20_000, "label": "500 m"},
    {"step": 1_000, "up_to": 100_000, "label": "1 km"},
    {"step":  None, "up_to":    None, "label": "remaining"},
]

# ---- PDG groups → one plot panel each ----
PDG_GROUPS = {
    "EM  (11, −11, 111)": [11, -11, 111],
    "Had (211, −211)":    [211, -211],
}

# ---- Particle types to plot ----
PARTICLE_STYLES = {
    "electrons": {"color": "#e41a1c", "label": "Electrons"},
    "muons":     {"color": "#377eb8", "label": "Muons"},
    "photons":   {"color": "#4daf4a", "label": "Photons"},
}

# =============================================================================
# END OF USER CONFIGURATION
# =============================================================================


# =============================================================================
# Bin-edge builder
# =============================================================================

def build_bin_edges(phases: list[dict], raw_dx: float) -> list[dict]:
    """
    Convert BIN_PHASES into a flat list of bin descriptors.

    Each bin dict contains:
        bin_id        – sequential integer starting at 1
        edge_in       – inner boundary in metres
        edge_out      – outer boundary in metres  (np.inf for catch-all)
        ring_k_min    – first raw ring_k in this bin
        ring_k_max    – last raw ring_k in this bin  (np.inf for catch-all)
        x_label       – multi-line tick label
        phase_label   – which phase this bin belongs to
    """
    bins   = []
    cursor = 0.0
    bid    = 1

    def _fmt(v):
        """Format metres as km string if >= 1000."""
        return f"{v/1000:.1f} km" if v >= 1000 else f"{v:.0f} m"

    for phase in phases:
        step   = phase["step"]
        up_to  = phase["up_to"]
        plabel = phase["label"]

        # Catch-all phase
        if step is None or up_to is None:
            k_min = int(cursor / raw_dx) + 1
            bins.append({
                "bin_id":     bid,
                "edge_in":    cursor,
                "edge_out":   np.inf,
                "ring_k_min": k_min,
                "ring_k_max": np.inf,
                "x_label":    f">{_fmt(cursor)}\n({plabel})",
                "phase_label": plabel,
            })
            break   # nothing after catch-all

        # Validate
        if step % raw_dx != 0:
            raise ValueError(
                f"BIN_PHASES step={step} m is not a multiple of "
                f"RAW_DX={raw_dx} m. Use a multiple of {raw_dx:.0f} m."
            )

        while cursor < up_to:
            inner = cursor
            outer = min(cursor + step, up_to)
            k_min = int(inner / raw_dx) + 1
            k_max = int(outer / raw_dx)
            bins.append({
                "bin_id":     bid,
                "edge_in":    inner,
                "edge_out":   outer,
                "ring_k_min": k_min,
                "ring_k_max": k_max,
                "x_label":    f"{_fmt(inner)}–{_fmt(outer)}\n({plabel})",
                "phase_label": plabel,
            })
            bid    += 1
            cursor  = outer

    return bins


# =============================================================================
# I/O helpers
# =============================================================================

def find_csv_files(output_dir: str, pdgs: list[int]) -> list[str]:
    files = []
    for pdg in pdgs:
        pattern = os.path.join(output_dir, f"pdg_{pdg}", "chunk_*_density.csv")
        found   = sorted(glob.glob(pattern))
        if not found:
            print(f"  WARNING: no CSVs for pdg={pdg} "
                  f"under {output_dir}/pdg_{pdg}/", file=sys.stderr)
        files.extend(found)
    return files


def load_csvs(files: list[str], label: str = "") -> pd.DataFrame:
    if not files:
        return pd.DataFrame()
    chunks = []
    for f in files:
        try:
            chunks.append(pd.read_csv(f, dtype={
                "shower_id": str, "incident_pdg": int, "actual_pdg": int,
                "particle_type": str, "plane_index": int, "ring_k": int,
                "n_hits": int, "n_hits_cumulative": int,
                "n_hits_plane": int, "n_rings_plane": int,
            }))
        except Exception as exc:
            print(f"  WARNING: cannot read {f}: {exc}", file=sys.stderr)
    if not chunks:
        return pd.DataFrame()
    df  = pd.concat(chunks, ignore_index=True)
    tag = f"[{label}] " if label else ""
    print(f"  {tag}{len(files)} files → {len(df):,} rows  "
          f"({df['shower_id'].nunique():,} showers)")
    return df


# =============================================================================
# Rebinning
# =============================================================================

def assign_bins(df: pd.DataFrame, bins: list[dict]) -> pd.DataFrame:
    """
    Vectorised mapping: ring_k  →  bin_id.
    Rings beyond the last explicit phase go to the catch-all bin.
    """
    if df.empty:
        return df

    max_k  = int(df["ring_k"].max())
    lookup = np.full(max_k + 2, -1, dtype=np.int64)

    for b in bins:
        k_min = int(b["ring_k_min"])
        k_max = b["ring_k_max"]
        end   = (max_k + 1) if (k_max is np.inf or k_max == np.inf) \
                else int(k_max) + 1
        end   = min(end, len(lookup))
        lookup[k_min:end] = b["bin_id"]

    ring_arr = df["ring_k"].to_numpy()
    clipped  = np.clip(ring_arr, 0, len(lookup) - 1)
    assigned = lookup[clipped]

    # Anything still unassigned → last bin (catch-all)
    last_bid = bins[-1]["bin_id"]
    assigned = np.where(assigned == -1, last_bid, assigned)

    out = df.copy()
    out["bin_id"] = assigned
    return out


def average_rebinned(df: pd.DataFrame, bins: list[dict],
                     particle_type: str,
                     plane: int | None = None) -> pd.DataFrame:
    """
    For one particle type (and optionally one plane):
      1. Filter to this particle type and plane (None = all planes).
      2. Sum n_hits per (shower_id, bin_id).
      3. Average across showers per bin_id → mean ± SEM.

    Returns DataFrame with columns:
        bin_id, edge_in, edge_out, x_label, phase_label,
        mean_n_hits, sem_n_hits, n_showers
    """
    sub = df[df["particle_type"] == particle_type]
    if plane is not None:
        sub = sub[sub["plane_index"] == plane]
    if sub.empty:
        return pd.DataFrame()

    per_shower = (
        sub.groupby(["shower_id", "bin_id"], as_index=False)["n_hits"]
        .sum()
        .rename(columns={"n_hits": "total_hits"})
    )
    stats = (
        per_shower.groupby("bin_id")["total_hits"]
        .agg(["mean", "sem", "count"])
        .reset_index()
        .rename(columns={"mean":  "mean_n_hits",
                         "sem":   "sem_n_hits",
                         "count": "n_showers"})
    )
    stats["sem_n_hits"] = stats["sem_n_hits"].fillna(0.0)
    meta  = pd.DataFrame(bins)[["bin_id", "edge_in", "edge_out",
                                 "x_label", "phase_label"]]
    stats = stats.merge(meta, on="bin_id", how="left")
    return stats.sort_values("bin_id").reset_index(drop=True)


# =============================================================================
# Plotting
# =============================================================================

def _build_ref_positions(bins: list[dict]) -> dict[int, int]:
    """Map bin_id → x position (1-based) for all bins."""
    return {b["bin_id"]: i + 1 for i, b in enumerate(bins)}


def plot_panel(ax, df: pd.DataFrame, bins: list[dict],
               row_label: str, plane: int | None,
               ref_positions: dict[int, int],
               log_y: bool = False,
               show_xlabel: bool = False,
               show_legend: bool = False):
    """
    Draw one subplot cell.

    Parameters
    ----------
    plane       : int → filter to that plane; None → all planes combined
    ref_positions : bin_id → x-position mapping (shared across all subplots)
    show_xlabel : only draw x-axis tick labels on the bottom row
    show_legend : only draw legend on the top row
    """
    has_data = False
    for ptype, style in PARTICLE_STYLES.items():
        stats = average_rebinned(df, bins, ptype, plane=plane)
        if stats.empty:
            continue
        has_data = True

        x   = stats["bin_id"].map(ref_positions).to_numpy(dtype=float)
        y   = stats["mean_n_hits"].to_numpy()
        err = stats["sem_n_hits"].to_numpy()
        n   = int(stats["n_showers"].max())

        ax.plot(x, y, color=style["color"], linewidth=1.2,
                marker="o", markersize=2,
                label=f"{style['label']} (N={n:,})")
        ax.fill_between(x, np.maximum(y - err, 0), y + err,
                        color=style["color"], alpha=0.12)

    if not has_data:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="grey")
        ax.set_yticks([])

    # Row label on the left y-axis
    ax.set_ylabel(row_label, fontsize=7, labelpad=3)
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)

    n_bins = len(bins)
    ax.set_xlim(0.5, n_bins + 0.5)

    # Phase boundary lines (thin dotted)
    prev_phase = bins[0]["phase_label"]
    for i, b in enumerate(bins[1:], start=2):
        if b["phase_label"] != prev_phase:
            ax.axvline(i - 0.5, color="grey", linestyle=":",
                       linewidth=0.6, alpha=0.6)
            prev_phase = b["phase_label"]

    if log_y and has_data:
        ax.set_yscale("log")

    # X-axis: only label the bottom row
    if show_xlabel:
        # Show only ~12 evenly spaced tick labels to avoid crushing
        step = max(1, n_bins // 12)
        sparse_positions = []
        sparse_labels    = []
        for i, b in enumerate(bins):
            if i % step == 0 or i == n_bins - 1:
                sparse_positions.append(ref_positions[b["bin_id"]])
                # Compact label: just the outer edge distance
                edge = b["edge_out"]
                if edge == np.inf:
                    lbl = f">{b['edge_in']/1000:.0f} km"
                elif edge >= 1000:
                    lbl = f"{edge/1000:.0f} km"
                else:
                    lbl = f"{edge:.0f} m"
                sparse_labels.append(lbl)

        ax.set_xticks(sparse_positions)
        ax.set_xticklabels(sparse_labels, fontsize=8, rotation=30, ha="right")
        ax.set_xlabel("Distance from origin (concentric square half-side)",
                      fontsize=9, labelpad=4)
    else:
        ax.set_xticks([])

    if show_legend and has_data:
        ax.legend(fontsize=6, framealpha=0.85, loc="upper right",
                  markerscale=1.2)


def make_figure(group_dfs: dict[str, pd.DataFrame],
                bins: list[dict],
                save_path: str,
                n_planes: int = 25,
                log_y: bool = False):
    """
    Build a grid of subplots:
        rows    = plane 0 … plane (n_planes-1)  +  last row = all planes
        columns = one per PDG group

    Total rows = n_planes + 1
    """
    n_rows   = n_planes + 1          # 25 plane rows + 1 "all planes" row
    n_cols   = len(group_dfs)
    col_labels = list(group_dfs.keys())

    # Shared x-position map
    ref_positions = _build_ref_positions(bins)

    # Figure: each row ~2 inches tall, each column ~9 inches wide
    fig_w = 9 * n_cols
    fig_h = 2.2 * n_rows
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        gridspec_kw={"hspace": 0.05, "wspace": 0.15},
    )

    # Column headers
    for col_idx, label in enumerate(col_labels):
        axes[0, col_idx].set_title(label, fontsize=11,
                                   fontweight="bold", pad=6)

    # Plane rows (0 … n_planes-1)
    for plane in range(n_planes):
        row_idx    = plane
        row_label  = f"Plane {plane}"
        is_bottom  = False          # never the true bottom row
        is_top     = (plane == 0)

        for col_idx, (group_label, df) in enumerate(group_dfs.items()):
            plot_panel(
                ax            = axes[row_idx, col_idx],
                df            = df,
                bins          = bins,
                row_label     = row_label,
                plane         = plane,
                ref_positions = ref_positions,
                log_y         = log_y,
                show_xlabel   = False,
                show_legend   = (is_top and col_idx == 0),
            )

    # Last row: all planes combined
    last_row = n_planes
    for col_idx, (group_label, df) in enumerate(group_dfs.items()):
        plot_panel(
            ax            = axes[last_row, col_idx],
            df            = df,
            bins          = bins,
            row_label     = "All planes",
            plane         = None,
            ref_positions = ref_positions,
            log_y         = log_y,
            show_xlabel   = True,
            show_legend   = (col_idx == 0),
        )

    # Style the "All planes" row to stand out
    for col_idx in range(n_cols):
        axes[last_row, col_idx].set_facecolor("#f0f4ff")
        for spine in axes[last_row, col_idx].spines.values():
            spine.set_edgecolor("#3060c0")
            spine.set_linewidth(1.2)

    # Overall title
    phase_parts = []
    for p in BIN_PHASES:
        if p["step"] is None:
            phase_parts.append(f"∞ ({p['label']})")
        else:
            phase_parts.append(
                f"{p['step']} m → {p['up_to']/1000:.0f} km ({p['label']})"
            )
    fig.suptitle(
        "Average particle density per plane  |  "
        "Binning: " + "  |  ".join(phase_parts),
        fontsize=11, y=1.002,
    )

    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}  ({n_rows} rows × {n_cols} cols)")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", required=True,
                        help="Root dir containing pdg_<N>/chunk_*_density.csv.")
    parser.add_argument("--save-fig", default="density_profile.png",
                        help="Output figure path (.png or .pdf).")
    parser.add_argument("--log-y", action="store_true",
                        help="Log scale on y-axis.")
    parser.add_argument("--raw-dx", type=float, default=RAW_DX,
                        help=f"Raw ring width in CSVs (default {RAW_DX:.0f} m).")
    parser.add_argument("--n-planes", type=int, default=24,
                        help="Number of individual plane rows to show (default 24, "
                             "i.e. planes 0–23). "
                             "The last row is always all-planes combined.")
    args = parser.parse_args()

    # Build bins
    bins = build_bin_edges(BIN_PHASES, raw_dx=args.raw_dx)
    print(f"Binning      : {len(bins)} bins across {len(BIN_PHASES)} phases")
    for b in bins[:3]:
        hi = "∞" if b["edge_out"] == np.inf else f"{b['edge_out']:.0f} m"
        kr = "∞" if b["ring_k_max"] == np.inf else str(b["ring_k_max"])
        print(f"  bin {b['bin_id']:>4}: {b['edge_in']:>8.0f} m – {hi:<12}  "
              f"rings {b['ring_k_min']}–{kr}")
    if len(bins) > 6:
        print(f"  ... ({len(bins)-6} bins not shown) ...")
    for b in bins[-3:]:
        hi = "∞" if b["edge_out"] == np.inf else f"{b['edge_out']:.0f} m"
        kr = "∞" if b["ring_k_max"] == np.inf else str(b["ring_k_max"])
        print(f"  bin {b['bin_id']:>4}: {b['edge_in']:>8.0f} m – {hi:<12}  "
              f"rings {b['ring_k_min']}–{kr}")
    print()

    # Load data per PDG group
    group_dfs: dict[str, pd.DataFrame] = {}
    for group_label, pdgs in PDG_GROUPS.items():
        print(f"Loading {group_label} ...")
        files = find_csv_files(args.output_dir, pdgs)
        df    = load_csvs(files, label=group_label)
        if not df.empty:
            df = assign_bins(df, bins)
        group_dfs[group_label] = df

    if all(df.empty for df in group_dfs.values()):
        print("ERROR: no data loaded — check --output-dir.", file=sys.stderr)
        sys.exit(1)

    print()
    print("Plotting...")
    make_figure(group_dfs, bins, save_path=args.save_fig,
                n_planes=args.n_planes, log_y=args.log_y)
    print("Done.")


if __name__ == "__main__":
    main()