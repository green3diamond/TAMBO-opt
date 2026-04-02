import h5py
import numpy as np
import torch
import yaml

from allshowers.preprocessing import compose

H5_FILE = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_ot_matched/merged_all_showers.h5"
TRAFO_PT = "/n/home04/hhanif/AllShowers/results/20260223_055410_CNF-Transformer/preprocessing/trafos.pt"
CFG_YAML = "/n/home04/hhanif/AllShowers/conf/transformer.yaml"

CHUNK_SHOWERS = 2000  # memory chunking only; still scans ALL showers
STOP_AFTER_FIRST_BAD = False  # True -> stop at first problem


def stats_flags(x: torch.Tensor):
    has_neg = bool((x < 0).any().item()) if (x.is_floating_point() or x.is_signed()) else False
    has_nan = bool(torch.isnan(x).any().item()) if x.is_floating_point() else False
    has_inf = bool(torch.isinf(x).any().item()) if x.is_floating_point() else False
    return has_neg, has_nan, has_inf


def to_points_matrix(arr: np.ndarray) -> np.ndarray:
    """
    Convert one object entry to (N, D).
    Supports:
      - already (N, D)
      - flat (N*D,) with D=4 or D=3
    """
    a = np.asarray(arr)
    if a.ndim == 2:
        return a
    if a.ndim == 1:
        for D in (4, 3):
            if a.size % D == 0:
                return a.reshape(-1, D)
    raise ValueError(f"Unrecognized point format: shape={a.shape}, size={a.size}")


def load_trafos(cfg_yaml: str, trafo_pt: str):
    with open(cfg_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    # Your YAML puts trafo specs under data:
    params = cfg.get("data", cfg.get("params", cfg))

    samples_energy_spec = params.get("samples_energy_trafo")
    samples_coordinate_spec = params.get("samples_coordinate_trafo")
    cond_spec = params.get("cond_trafo")

    missing = [k for k, v in [
        ("samples_energy_trafo", samples_energy_spec),
        ("samples_coordinate_trafo", samples_coordinate_spec),
        ("cond_trafo", cond_spec),
    ] if v is None]

    if missing:
        raise KeyError(
            f"Could not find trafo specs in YAML. Missing keys: {missing}\n"
            "Searched in: cfg['data'] then cfg['params'] then top-level."
        )

    samples_energy_trafo = compose(samples_energy_spec)
    samples_coordinate_trafo = compose(samples_coordinate_spec)
    cond_trafo = compose(cond_spec)

    state = torch.load(trafo_pt, map_location="cpu", weights_only=True)
    samples_energy_trafo.load_state_dict(state["samples_energy_trafo"])
    samples_coordinate_trafo.load_state_dict(state["samples_coordinate_trafo"])
    cond_trafo.load_state_dict(state["cond_trafo"])

    samples_energy_trafo.eval()
    samples_coordinate_trafo.eval()
    cond_trafo.eval()

    return samples_energy_trafo, samples_coordinate_trafo, cond_trafo


def check_object_dataset_with_trafos(name, dset, coord_trafo, energy_trafo):
    print(f"\n=== Checking OBJECT dataset (forward trafos): {name} ===")
    print(f"Shape: {dset.shape}, dtype: {dset.dtype}")

    any_neg = any_nan = any_inf = False
    first_bad = None  # (index, part, (neg,nan,inf))

    n = dset.shape[0]
    for i in range(0, n, CHUNK_SHOWERS):
        chunk = dset[i:i + CHUNK_SHOWERS]

        for k, entry in enumerate(chunk):
            idx = i + k
            if entry is None:
                continue

            pts = to_points_matrix(entry)  # (N, D)
            if pts.shape[1] < 3:
                continue

            # coords = first 2 columns
            xy = torch.from_numpy(pts[:, :2]).to(torch.float32)
            # energy heuristic: last column (works for [x,y,layer,E] or [x,y,E])
            edep = torch.from_numpy(pts[:, -1]).to(torch.float32)

            with torch.no_grad():
                xy_t = coord_trafo(xy)      # forward
                e_t = energy_trafo(edep)    # forward

            neg_xy, nan_xy, inf_xy = stats_flags(xy_t)
            neg_e, nan_e, inf_e = stats_flags(e_t)

            has_neg = neg_xy or neg_e
            has_nan = nan_xy or nan_e
            has_inf = inf_xy or inf_e

            if has_neg or has_nan or has_inf:
                any_neg |= has_neg
                any_nan |= has_nan
                any_inf |= has_inf

                if first_bad is None:
                    parts = []
                    if neg_xy or nan_xy or inf_xy:
                        parts.append("coords")
                    if neg_e or nan_e or inf_e:
                        parts.append("energy")
                    first_bad = (idx, ",".join(parts), (has_neg, has_nan, has_inf))

                if STOP_AFTER_FIRST_BAD:
                    break

        if STOP_AFTER_FIRST_BAD and first_bad is not None:
            break

    print(f"→ Negative (after trafo): {any_neg}")
    print(f"→ NaN (after trafo):      {any_nan}")
    print(f"→ Inf (after trafo):      {any_inf}")

    if first_bad is not None:
        print(f"First bad shower index: {first_bad[0]} (part={first_bad[1]}, flags neg/nan/inf={first_bad[2]})")
    else:
        print("No bad values found after forward transforms.")


def maybe_check_cond_trafo(f, cond_trafo):
    # try a few likely keys (adjust if yours differs)
    for key in ["energies", "target/energies", "incident_energies"]:
        if key in f and np.issubdtype(f[key].dtype, np.number):
            x = torch.from_numpy(f[key][:].astype(np.float32))
            with torch.no_grad():
                x_t = cond_trafo(x)
            neg, nan, inf = stats_flags(x_t)
            print(f"\n=== Checking cond_trafo on dataset: {key} ===")
            print(f"→ Negative (after trafo): {neg}")
            print(f"→ NaN (after trafo):      {nan}")
            print(f"→ Inf (after trafo):      {inf}")
            return
    print("\n(cond_trafo) No suitable numeric conditioning dataset found (skipped).")


def main():
    energy_trafo, coord_trafo, cond_trafo = load_trafos(CFG_YAML, TRAFO_PT)

    with h5py.File(H5_FILE, "r") as f:
        for key in ["showers", "target/point_clouds"]:
            if key in f:
                dset = f[key]
                if dset.dtype == object:
                    check_object_dataset_with_trafos(key, dset, coord_trafo, energy_trafo)
                else:
                    print(f"\nDataset {key} is dtype={dset.dtype}, not object (skipped by this checker).")
            else:
                print(f"\nDataset not found: {key}")

        maybe_check_cond_trafo(f, cond_trafo)


if __name__ == "__main__":
    main()