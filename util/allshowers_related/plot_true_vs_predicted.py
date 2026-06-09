import showerdata
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

ML_FILE      = "/n/home04/hhanif/AllShowers/results/20260301_171240_CNF-Transformer/samples01.h5"
CORSIKA_FILE = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_processed_step1_v3/merged_all_showers_test.h5"
N_SHOWERS    = 20
OUT_DIR      = "/n/home04/hhanif/AllShowers/plots"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading ML showers …")
showers_ml = showerdata.load(ML_FILE, stop=N_SHOWERS)

print("Loading CORSIKA showers …")
showers_corsika = showerdata.load(CORSIKA_FILE, stop=N_SHOWERS)

layer_energy_ml      = np.array(showerdata.observables.calc_energy_per_layer(showers_ml))
layer_energy_corsika = np.array(showerdata.observables.calc_energy_per_layer(showers_corsika))

COLORS = {"ML": "#E05C5C", "CORSIKA": "#4C9BE8"}

def scatter_ax_log(ax, x, y, xlabel, ylabel, title):
    """For positive-only quantities (energy) — log scale."""
    ax.scatter(x, y, color=COLORS["ML"], s=40, zorder=3)
    lims = [min(x.min(), y.min()) * 0.9, max(x.max(), y.max()) * 1.1]
    ax.plot(lims, lims, "k--", lw=1.2, label="y = x")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

def scatter_ax_linear(ax, x, y, xlabel, ylabel, title):
    """For quantities that can be negative (x, y coords) — linear scale."""
    ax.scatter(x, y, color=COLORS["ML"], s=10, alpha=0.5, zorder=3)
    lims = [min(x.min(), y.min()) * 1.1 if min(x.min(), y.min()) < 0
            else min(x.min(), y.min()) * 0.9,
            max(x.max(), y.max()) * 1.1]
    ax.plot(lims, lims, "k--", lw=1.2, label="y = x")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

def hist_ax(ax, true_vals, pred_vals, xlabel, title, log=False):
    bins = np.logspace(np.log10(max(min(true_vals.min(), pred_vals.min()), 1e-10)),
                       np.log10(max(true_vals.max(), pred_vals.max())), 25) if log else 25
    ax.hist(true_vals, bins=bins, color=COLORS["CORSIKA"], alpha=0.6, label="True (CORSIKA)", density=True)
    ax.hist(pred_vals, bins=bins, color=COLORS["ML"],      alpha=0.6, label="Predicted (ML)", density=True)
    if log:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

for i in range(N_SHOWERS):
    e_true = layer_energy_corsika[i]
    e_pred = layer_energy_ml[i]

    hits_cor = showers_corsika[i].points.reshape(-1, 4)
    hits_ml  = showers_ml[i].points.reshape(-1, 4)
    x_true, y_true = hits_cor[:, 0], hits_cor[:, 1]
    x_pred, y_pred = hits_ml[:, 0],  hits_ml[:, 1]

    fig, axes = plt.subplots(3, 2, figsize=(12, 13))
    fig.suptitle(f"Shower {i:03d} — True vs Predicted", fontsize=13, fontweight="bold")

    # Row 1: Energy (log scale)
    scatter_ax_log(axes[0, 0], e_true, e_pred,
                   "True energy [a.u.]", "Predicted energy [a.u.]", "Energy per Layer — Scatter")
    hist_ax(axes[0, 1], e_true, e_pred,
            "Energy [a.u.]", "Energy Distribution", log=True)

    # Row 2: X coordinate (linear scale)
    scatter_ax_linear(axes[1, 0], x_true, x_pred,
                      "True X [a.u.]", "Predicted X [a.u.]", "X Coordinate — Scatter")
    hist_ax(axes[1, 1], x_true, x_pred,
            "X [a.u.]", "X Distribution", log=False)

    # Row 3: Y coordinate (linear scale)
    scatter_ax_linear(axes[2, 0], y_true, y_pred,
                      "True Y [a.u.]", "Predicted Y [a.u.]", "Y Coordinate — Scatter")
    hist_ax(axes[2, 1], y_true, y_pred,
            "Y [a.u.]", "Y Distribution", log=False)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"true_vs_pred_shower_{i:03d}.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")

print(f"\nDone. {N_SHOWERS} plots saved to {OUT_DIR}")