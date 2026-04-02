"""
Comparison of ML (CNF-Transformer) vs CORSIKA shower observables.
Produces one PDF per shower index (shower_000.pdf … shower_019.pdf).

Panels:
  1. Energy per layer          — markers + Poisson error bars
  2. Energy per radial bin     — markers + Poisson error bars
  3. Center of Energy          — histogram (all 20 showers), current shower marked
  4. Layer energy ratio ML/COR — markers + propagated error bars
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import showerdata

# ── Config ────────────────────────────────────────────────────────────────────
ML_FILE      = "/n/home04/hhanif/AllShowers/results/20260301_171240_CNF-Transformer/samples01.h5"
CORSIKA_FILE = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_processed_step1_v3/merged_all_showers_test.h5"
N_SHOWERS    = 10
OUT_DIR      = "/n/home04/hhanif/AllShowers/plots"
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {"ML": "#E05C5C", "CORSIKA": "#4C9BE8"}
# ─────────────────────────────────────────────────────────────────────────────

print("Loading ML showers …")
showers_ml = showerdata.load(ML_FILE, stop=N_SHOWERS)

print("Loading CORSIKA showers …")
showers_corsika = showerdata.load(CORSIKA_FILE, stop=N_SHOWERS)

# ── Compute observables ───────────────────────────────────────────────────────
print("Computing observables …")
layer_energy_ml      = np.array(showerdata.observables.calc_energy_per_layer(showers_ml))
layer_energy_corsika = np.array(showerdata.observables.calc_energy_per_layer(showers_corsika))

radial_energy_ml      = np.array(showerdata.observables.calc_energy_per_radial_bin(showers_ml))
radial_energy_corsika = np.array(showerdata.observables.calc_energy_per_radial_bin(showers_corsika))

coe_ml      = np.array(showerdata.observables.calc_center_of_energy(showers_ml))
coe_corsika = np.array(showerdata.observables.calc_center_of_energy(showers_corsika))

def get_coe(coe_arr, idx):
    """Return scalar CoE for shower idx (handles 1-D and 2-D arrays)."""
    row = coe_arr[idx]
    return float(row[0]) if np.ndim(row) > 0 else float(row)

# Pre-compute CoE arrays for the histogram (same across all shower plots)
coe_ml_vals  = np.array([get_coe(coe_ml,      j) for j in range(N_SHOWERS)])
coe_cor_vals = np.array([get_coe(coe_corsika,  j) for j in range(N_SHOWERS)])
coe_bins = np.linspace(min(coe_ml_vals.min(), coe_cor_vals.min()),
                       max(coe_ml_vals.max(), coe_cor_vals.max()), 15)

# ── Per-shower plots ──────────────────────────────────────────────────────────
for i in range(N_SHOWERS):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Shower {i:03d} — ML (CNF-Transformer) vs CORSIKA",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    # ── 1. Energy per layer — markers + Poisson error bars ───────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    layers   = np.arange(layer_energy_ml.shape[1])
    le_ml_i  = layer_energy_ml[i]
    le_cor_i = layer_energy_corsika[i]

    ax1.errorbar(layers, le_ml_i,  yerr=np.sqrt(np.abs(le_ml_i)),
                 fmt="o", ms=4, lw=1.2, capsize=3,
                 color=COLORS["ML"],      label="ML (CNF-Transformer)")
    ax1.errorbar(layers, le_cor_i, yerr=np.sqrt(np.abs(le_cor_i)),
                 fmt="s", ms=4, lw=1.2, capsize=3,
                 color=COLORS["CORSIKA"], label="CORSIKA")
    ax1.set_xlabel("Layer index")
    ax1.set_ylabel("Energy deposit [a.u.]")
    ax1.set_title("Energy per Layer")
    ax1.set_yscale("log")
    ax1.legend(fontsize=9)

    # ── 2. Energy per radial bin — markers + Poisson error bars ──────────────
    ax2 = fig.add_subplot(gs[0, 1])
    rbins    = np.arange(radial_energy_ml.shape[1])
    re_ml_i  = radial_energy_ml[i]
    re_cor_i = radial_energy_corsika[i]

    ax2.errorbar(rbins, re_ml_i,  yerr=np.sqrt(np.abs(re_ml_i)),
                 fmt="o", ms=4, lw=1.2, capsize=3,
                 color=COLORS["ML"],      label="ML (CNF-Transformer)")
    ax2.errorbar(rbins, re_cor_i, yerr=np.sqrt(np.abs(re_cor_i)),
                 fmt="s", ms=4, lw=1.2, capsize=3,
                 color=COLORS["CORSIKA"], label="CORSIKA")
    ax2.set_xlabel("Radial bin index")
    ax2.set_ylabel("Energy deposit [a.u.]")
    ax2.set_title("Energy per Radial Bin")
    ax2.set_yscale("log")
    ax2.legend(fontsize=9)

    # ── 3. Center of Energy — histogram (all showers), current shower marked ─
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(coe_ml_vals,  bins=coe_bins, color=COLORS["ML"],
             alpha=0.6, label="ML (CNF-Transformer)", density=True)
    ax3.hist(coe_cor_vals, bins=coe_bins, color=COLORS["CORSIKA"],
             alpha=0.6, label="CORSIKA", density=True, histtype="step", lw=2)
    ax3.axvline(get_coe(coe_ml,      i), color=COLORS["ML"],
                lw=1.8, ls="--", label=f"Shower {i:03d} ML")
    ax3.axvline(get_coe(coe_corsika, i), color=COLORS["CORSIKA"],
                lw=1.8, ls=":",  label=f"Shower {i:03d} COR")
    ax3.set_xlabel("Center of Energy [a.u.]")
    ax3.set_ylabel("Normalised counts")
    ax3.set_title("Center of Energy Distribution")
    ax3.legend(fontsize=8)

    # ── 4. Layer energy ratio — markers + propagated error bars ──────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ratio = np.where(le_cor_i > 0, le_ml_i / le_cor_i, np.nan)
    ratio_err = np.where(
        (le_ml_i > 0) & (le_cor_i > 0),
        ratio * np.sqrt(1.0 / le_ml_i + 1.0 / le_cor_i),
        np.nan
    )
    ax4.axhline(1.0, color="gray", lw=1, ls="--")
    ax4.errorbar(layers, ratio, yerr=ratio_err,
                 fmt="o", ms=4, lw=1.2, capsize=3, color="#9B59B6")
    ax4.set_xlabel("Layer index")
    ax4.set_ylabel("ML / CORSIKA")
    ax4.set_title("Layer Energy Ratio (ML / CORSIKA)")
    ax4.set_ylim(0, 3)

    # ── Save ─────────────────────────────────────────────────────────────────
    out = os.path.join(OUT_DIR, f"shower_{i:03d}.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")

print(f"\nDone. {N_SHOWERS} plots saved to {OUT_DIR}")