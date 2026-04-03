import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ml_file        = "/n/home04/hhanif/AllShowers/results/20260402_150113_CNF-Transformer/samples00.h5"
simulated_file = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/combined_electrons_balanced-test-file.h5"

CLASS_NAMES = {
    0: r"$e^\pm$/$\gamma$/$\pi^0$",
    1: r"$\pi^\pm$",
}
NUM_LAYERS = 24
US = 1e6   # seconds -> microseconds


# ------------------------------------------------------------------ utilities

def list_datasets(path):
    out = {}
    with h5py.File(path, "r") as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                out[name] = obj.shape
        f.visititems(visitor)
    return out


def compare_common_metadata(sim_path, ml_path, keys_to_check=None, atol=1e-6, rtol=1e-6):
    """
    Compare common datasets between files to make sure they represent the same sample.
    Useful for pdg, direction, energy, zenith, azimuth, etc.
    """
    print("\n" + "=" * 80)
    print("Comparing common metadata datasets")
    print("=" * 80)

    with h5py.File(sim_path, "r") as fs, h5py.File(ml_path, "r") as fm:
        sim_keys = set(fs.keys())
        ml_keys  = set(fm.keys())
        common   = sorted(sim_keys & ml_keys)

        print(f"Top-level common datasets: {common}")

        if keys_to_check is None:
            # check the most likely conditioning / metadata datasets if present
            preferred = [
                "pdg", "directions", "directions",
                "energy", "energies",
                "theta", "phi", "zenith", "azimuth",
                "condition", "conditions",
                "shape"
            ]
            keys_to_check = [k for k in preferred if k in common]

        if not keys_to_check:
            print("No requested metadata datasets found in both files.")
            return

        for key in keys_to_check:
            a = fs[key][:]
            b = fm[key][:]

            same_shape = a.shape == b.shape
            same_dtype = a.dtype == b.dtype

            print(f"\nDataset: {key}")
            print(f"  sim shape={a.shape}, dtype={a.dtype}")
            print(f"  ml  shape={b.shape}, dtype={b.dtype}")

            if not same_shape:
                print("  ❌ shape mismatch")
                continue

            if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
                exact = np.array_equal(a, b)
                close = np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True)
                max_abs = np.nanmax(np.abs(a - b)) if a.size else 0.0
                print(f"  exact match : {exact}")
                print(f"  allclose    : {close}")
                print(f"  max |diff|  : {max_abs:.6g}")
            else:
                exact = np.array_equal(a, b)
                print(f"  exact match : {exact}")

            # quick summary for direction-like arrays
            if key in {"direction", "directions"} and a.ndim >= 2:
                print(f"  sim first 3 rows:\n{a[:3]}")
                print(f"  ml  first 3 rows:\n{b[:3]}")


def assert_same_length_and_alignment(sim_path, ml_path, keys=("pdg", "direction", "directions")):
    """
    Hard fail if important metadata differ.
    """
    with h5py.File(sim_path, "r") as fs, h5py.File(ml_path, "r") as fm:
        for key in keys:
            if key in fs and key in fm:
                a = fs[key][:]
                b = fm[key][:]
                if a.shape != b.shape:
                    raise ValueError(f"{key}: shape mismatch {a.shape} vs {b.shape}")
                if np.issubdtype(a.dtype, np.number):
                    if not np.allclose(a, b, atol=1e-6, rtol=1e-6, equal_nan=True):
                        raise ValueError(f"{key}: values do not match between simulated and ML files")
                else:
                    if not np.array_equal(a, b):
                        raise ValueError(f"{key}: values do not match between simulated and ML files")


# ------------------------------------------------------------------ loaders

def _compute_energy_per_layer(pts_3d, num_layers=NUM_LAYERS):
    N = pts_3d.shape[0]
    energy_per_layer = np.zeros((N, num_layers), dtype=np.float32)
    layer_idx  = np.clip((pts_3d[..., 2] + 0.1).astype(np.int32), 0, num_layers - 1)
    energies   = pts_3d[..., 3].astype(np.float32)
    shower_idx = np.arange(N).reshape(-1, 1).repeat(pts_3d.shape[1], axis=1)
    np.add.at(energy_per_layer, (shower_idx, layer_idx), energies)
    return energy_per_layer


def _compute_energy_per_radial_bin(pts_3d, num_bins=200, r_max=400.0):
    N         = pts_3d.shape[0]
    bin_edges = np.linspace(0, r_max, num_bins + 1, dtype=np.float32)
    radial    = np.sqrt(pts_3d[..., 0]**2 + pts_3d[..., 1]**2).astype(np.float32)
    bin_idx   = np.digitize(radial, bins=bin_edges) - 1
    energies  = pts_3d[..., 3].astype(np.float32)
    oob = bin_idx >= num_bins
    energies[oob] = 0.0
    bin_idx[oob]  = 0
    energy_per_radial = np.zeros((N, num_bins), dtype=np.float32)
    shower_idx = np.arange(N).reshape(-1, 1).repeat(pts_3d.shape[1], axis=1)
    np.add.at(energy_per_radial, (shower_idx, bin_idx), energies)
    return energy_per_radial


def _compute_center_of_energy(pts_3d):
    energies = pts_3d[..., 3:4].astype(np.float32)
    total    = energies.sum(axis=1, keepdims=True) + 1e-8
    weighted = pts_3d[..., :3].astype(np.float32) * energies
    return (weighted.sum(axis=1) / total[:, 0]).astype(np.float32)


def load_all(path):
    print(f"  Reading {path} ...")
    with h5py.File(path, "r") as f:
        pdg   = f["pdg"][:]
        raw   = f["showers"][:]
        shape = f["shape"][:]

        # optional metadata
        direction = None
        if "direction" in f:
            direction = f["direction"][:]
        elif "directions" in f:
            direction = f["directions"][:]

    N, max_pts, ncols = int(shape[0]), int(shape[1]), int(shape[2])
    print(f"  Reshaping {N} showers → ({N}, {max_pts}, {ncols}) ...")

    pts = np.zeros((N, max_pts, ncols), dtype=np.float32)
    for i, flat in enumerate(raw):
        arr = np.asarray(flat, dtype=np.float32).reshape(-1, ncols)
        pts[i, :len(arr)] = arr

    print("  Computing energy_per_layer ...")
    layer = _compute_energy_per_layer(pts)

    print("  Computing energy_per_radial_bin ...")
    radial = _compute_energy_per_radial_bin(pts)

    print("  Computing center_of_energy ...")
    center = _compute_center_of_energy(pts)

    return layer, radial, center, pdg, raw, ncols, direction


def compute_time_obs(raw, ncols, num_layers=NUM_LAYERS):
    if ncols < 5:
        raise ValueError(f"ncols={ncols} — no time feature (col 4).")
    N = len(raw)
    mean_t          = np.zeros(N, dtype=np.float64)
    std_t           = np.zeros(N, dtype=np.float64)
    time_per_layer  = np.zeros((N, num_layers), dtype=np.float64)
    count_per_layer = np.zeros((N, num_layers), dtype=np.float64)

    for i, flat in enumerate(raw):
        pts = flat.reshape(-1, ncols)
        mask = pts[:, 3] > 0
        if mask.sum() == 0:
            continue

        t = pts[mask, 4]
        mean_t[i] = t.mean()
        std_t[i] = t.std()

        layer_idx = np.clip((pts[mask, 2] + 0.1).astype(np.int32), 0, num_layers - 1)
        np.add.at(time_per_layer[i], layer_idx, t)
        np.add.at(count_per_layer[i], layer_idx, 1)

    mean_t_per_layer = time_per_layer / count_per_layer.clip(min=1)
    return mean_t, std_t, mean_t_per_layer


# ------------------------------------------------------------------ checks before loading

print("Dataset inventory (simulated):")
for k, v in list_datasets(simulated_file).items():
    print(f"  {k}: {v}")

print("\nDataset inventory (ML):")
for k, v in list_datasets(ml_file).items():
    print(f"  {k}: {v}")

compare_common_metadata(
    simulated_file,
    ml_file,
    keys_to_check=["pdg", "direction", "directions", "shape"]
)

# Use this if you want the script to stop when metadata do not match
# assert_same_length_and_alignment(simulated_file, ml_file)


# ------------------------------------------------------------------ load

print("\nLoading Simulated file...")
s_layer, s_radial, s_center, s_pdg, s_raw, s_ncols, s_dir = load_all(simulated_file)

print("Loading ML file...")
m_layer, m_radial, m_center, m_pdg, m_raw, m_ncols, m_dir = load_all(ml_file)

print("Computing time observables (Simulated)...")
s_mean_t, s_std_t, s_t_layer = compute_time_obs(s_raw, s_ncols)

print("Computing time observables (ML)...")
m_mean_t, m_std_t, m_t_layer = compute_time_obs(m_raw, m_ncols)

print("Done.")


def _safe_norm(arr):
    s = arr.sum(1, keepdims=True)
    return np.where(s > 0, arr / s, 0.0)

s_layer  = _safe_norm(s_layer)
m_layer  = _safe_norm(m_layer)
s_radial = _safe_norm(s_radial)
m_radial = _safe_norm(m_radial)

layers = np.arange(1, NUM_LAYERS + 1)
rbins  = np.arange(1, s_radial.shape[1] + 1)

row_configs = [
    ("All", None),
    (f"Class 0: {CLASS_NAMES[0]}", 0),
    (f"Class 1: {CLASS_NAMES[1]}", 1),
]

NCOLS = 6
col_titles = [
    "Longitudinal Energy Profile",
    "Radial Energy Profile",
    r"CoE Radius $\sqrt{x^2+y^2}$",
    "Longitudinal Time Profile",
    "Mean Hit Time per Shower",
    "Time Spread per Shower",
]
col_xlabels = [
    "Layer",
    "Radial bin",
    "mm",
    "Layer",
    r"Mean $t$ [$\mu$s]",
    r"Std $t$ [$\mu$s]",
]


# ------------------------------------------------------------------ helpers

def mask_for(pdg_arr, pdg_val):
    return np.ones(len(pdg_arr), dtype=bool) if pdg_val is None else pdg_arr == pdg_val


def capped_indices(s_mask, m_mask):
    s_idx = np.where(s_mask)[0]
    m_idx = np.where(m_mask)[0]
    n = min(len(s_idx), len(m_idx))
    rng = np.random.default_rng(42)
    s_idx = rng.choice(s_idx, size=n, replace=False)
    m_idx = rng.choice(m_idx, size=n, replace=False)
    return s_idx, m_idx, n


def plot_row(axes, s_idx, m_idx, n):
    sl  = s_layer[s_idx]
    ml  = m_layer[m_idx]
    sr  = s_radial[s_idx]
    mr  = m_radial[m_idx]
    sc  = s_center[s_idx]
    mc  = m_center[m_idx]
    stl = s_t_layer[s_idx]
    mtl = m_t_layer[m_idx]
    smt = s_mean_t[s_idx]
    mmt = m_mean_t[m_idx]
    sst = s_std_t[s_idx]
    mst = m_std_t[m_idx]

    sLm, sLs = sl.mean(0), sl.std(0)
    mLm, mLs = ml.mean(0), ml.std(0)
    sTm, sTs = stl.mean(0), stl.std(0)
    mTm, mTs = mtl.mean(0), mtl.std(0)

    sCr = np.sqrt(sc[:, 0]**2 + sc[:, 1]**2)
    mCr = np.sqrt(mc[:, 0]**2 + mc[:, 1]**2)

    simulated_label = f"Simulated ({n})"
    ml_label        = f"ML ({n})"

    ax = axes[0]
    ax.plot(layers, sLm, marker='o', ms=3, lw=1.2, label=simulated_label)
    ax.fill_between(layers, sLm - sLs, sLm + sLs, alpha=0.2)
    ax.plot(layers, mLm, marker='s', ms=3, lw=1.2, label=ml_label)
    ax.fill_between(layers, mLm - mLs, mLm + mLs, alpha=0.2)
    ax.set_xticks(np.arange(1, 25, 4))
    ax.grid(True, lw=0.4)
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(rbins, np.nanmean(sr, axis=0), lw=1.2, label=simulated_label)
    ax.plot(rbins, np.nanmean(mr, axis=0), lw=1.2, label=ml_label)
    ax.grid(True, lw=0.4)
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.hist(sCr, bins=40, density=True, alpha=0.45, label=simulated_label)
    ax.hist(mCr, bins=40, density=True, alpha=0.45, label=ml_label)
    ax.grid(True, lw=0.4)
    ax.legend(fontsize=7)

    ax = axes[3]
    ax.plot(layers, sTm * US, marker='o', ms=3, lw=1.2, label=simulated_label)
    ax.fill_between(layers, (sTm - sTs) * US, (sTm + sTs) * US, alpha=0.2)
    ax.plot(layers, mTm * US, marker='s', ms=3, lw=1.2, label=ml_label)
    ax.fill_between(layers, (mTm - mTs) * US, (mTm + mTs) * US, alpha=0.2)
    ax.set_xticks(np.arange(1, 25, 4))
    ax.grid(True, lw=0.4)
    ax.legend(fontsize=7)

    ax = axes[4]
    ax.hist(smt * US, bins=40, density=True, alpha=0.45, label=simulated_label)
    ax.hist(mmt * US, bins=40, density=True, alpha=0.45, label=ml_label)
    ax.grid(True, lw=0.4)
    ax.legend(fontsize=7)

    ax = axes[5]
    ax.hist(sst * US, bins=40, density=True, alpha=0.45, label=simulated_label)
    ax.hist(mst * US, bins=40, density=True, alpha=0.45, label=ml_label)
    ax.grid(True, lw=0.4)
    ax.legend(fontsize=7)


# ------------------------------------------------------------------ figure

NROWS = len(row_configs)
fig = plt.figure(figsize=(NCOLS * 3.6, NROWS * 3.2))
gs = gridspec.GridSpec(
    NROWS, NCOLS,
    figure=fig,
    hspace=0.6, wspace=0.35,
    top=0.97, bottom=0.04, left=0.06, right=0.99,
)

for row_i, (row_label, class_val) in enumerate(row_configs):
    s_mask = mask_for(s_pdg, class_val)
    m_mask = mask_for(m_pdg, class_val)
    s_idx, m_idx, n = capped_indices(s_mask, m_mask)

    axes = [fig.add_subplot(gs[row_i, c]) for c in range(NCOLS)]
    plot_row(axes, s_idx, m_idx, n)

    header = (
        f"{'All Classes' if class_val is None else row_label}"
        f"  —  No. of Samples (capped at {n} each)"
    )
    axes[0].annotate(
        header,
        xy=(0, 1.18), xycoords="axes fraction",
        fontsize=9.5, fontweight="bold", color="#222222",
    )

    for ax in axes:
        ax.tick_params(labelsize=7)

    axes[0].set_ylabel("Norm. Energy", fontsize=8)
    axes[1].set_ylabel("Norm. Energy", fontsize=8)
    axes[2].set_ylabel("Density", fontsize=8)
    axes[3].set_ylabel(r"Mean $t$ [$\mu$s]", fontsize=8)
    axes[4].set_ylabel("Density", fontsize=8)
    axes[5].set_ylabel("Density", fontsize=8)

    for c, ax in enumerate(axes):
        ax.set_xlabel(col_xlabels[c], fontsize=8)
        ax.set_title(col_titles[c], fontsize=9, pad=4)

out = "shower_observables_by_class.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved → {out}")