#!/usr/bin/env python3
"""
Diagnostic script for postprocess_showers.py output H5 files.
Reads showers in CHUNKS to keep memory bounded — works on huge files.

Checks for symptoms of:
  1. NaN / Inf values in any feature
  2. Out-of-physical-range x, y, plane, energy, time values
  3. Negative energies (should never appear; clamp at 0 in postprocess)
  4. Shape mismatches between num_points and the flat shower array
  5. plane_idx values that look corrupted (non-integer floats)
  6. Within-shower (x,y,plane) duplicates (post-clustering invariant)
  7. Per-plane y/x range ratio (risk indicator for the stride bug)
  8. Empty showers, num_points distribution

Usage:
    python inspect_h5.py file.h5
    python inspect_h5.py /path/to/pdg_11/                  # all .h5 in dir
    python inspect_h5.py file1.h5 file2.h5 ...
    python inspect_h5.py file.h5 --chunk-size 256          # tune memory
    python inspect_h5.py file.h5 --max-showers 5000        # cap inspection
    python inspect_h5.py file.h5 --max-showers -1          # all (no cap)
    python /n/home04/hhanif/TAMBO-opt/job_submission_scripts/inspect_h5.py /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/h5_files/combined_electrons_no_time_2048.h5 --max-showers -1   --quiet                   # less verbose

Memory notes:
  - num_points, energies, directions, pdg, actual_pdg are loaded once
    (small: O(N_showers)).
  - The variable-length 'showers' dataset is read SHOWER-BY-SHOWER inside a
    chunk loop. Peak memory is dominated by chunk-size * nmax * n_feat * 4 B.
    Default chunk-size=128, nmax=6016, n_feat=5 -> ~15 MB peak. Safe.
"""
from __future__ import annotations

import argparse
import os
import sys
from glob import glob
from collections import Counter

import numpy as np
import h5py


# ---------- physical sanity ranges (adjust if your detector differs) ---------
XY_ABS_MAX        = 1e4
ENERGY_MAX        = 1e6
PLANE_IDX_MAX     = 100_000
TIME_ABS_MAX      = 1e6
# -----------------------------------------------------------------------------


def _summary(arr, name):
    if arr.size == 0:
        return f"  {name:14s}: empty"
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    fa = arr[np.isfinite(arr)]
    if fa.size == 0:
        return f"  {name:14s}: ALL non-finite (nan={n_nan} inf={n_inf})"
    return (f"  {name:14s}: min={fa.min():.4g}  max={fa.max():.4g}  "
            f"mean={fa.mean():.4g}  std={fa.std():.4g}  "
            f"nan={n_nan}  inf={n_inf}")


def _check_one_shower(arr, n_feat):
    out = {
        "nan": 0, "inf": 0, "xy_oor": 0, "plane_oor": 0,
        "energy_oor": 0, "energy_neg": 0, "time_oor": 0,
        "duplicate": 0, "plane_non_int": 0, "y_x_skew": [],
    }
    if int(np.isnan(arr).sum()) > 0:
        out["nan"] = 1
    if int(np.isinf(arr).sum()) > 0:
        out["inf"] = 1

    xs    = arr[:, 0]
    ys    = arr[:, 1]
    planes= arr[:, 2]
    energ = arr[:, 3]
    times = arr[:, 4] if n_feat == 5 else None

    if (np.abs(xs) > XY_ABS_MAX).any() or (np.abs(ys) > XY_ABS_MAX).any():
        out["xy_oor"] = 1
    if (planes < 0).any() or (planes > PLANE_IDX_MAX).any():
        out["plane_oor"] = 1
    if np.any(np.abs(planes - np.round(planes)) > 1e-3):
        out["plane_non_int"] = 1
    if (energ < 0).any():
        out["energy_neg"] = 1
    if (energ > ENERGY_MAX).any():
        out["energy_oor"] = 1
    if times is not None:
        finite_t = times[np.isfinite(times)]
        if finite_t.size and (np.abs(finite_t) > TIME_ABS_MAX).any():
            out["time_oor"] = 1

    keys = np.stack([
        np.round(xs * 1e3).astype(np.int64),
        np.round(ys * 1e3).astype(np.int64),
        np.round(planes).astype(np.int64),
    ], axis=1)
    if len(np.unique(keys, axis=0)) != len(keys):
        out["duplicate"] = 1

    skews = []
    for p in np.unique(planes):
        sub = arr[planes == p]
        if len(sub) < 5:
            continue
        x_range = sub[:, 0].max() - sub[:, 0].min()
        y_range = sub[:, 1].max() - sub[:, 1].min()
        if x_range > 0:
            skews.append(y_range / x_range)
        elif y_range > 0:
            skews.append(np.inf)
    out["y_x_skew"] = skews
    return out


def inspect_one_file(path, chunk_size, max_showers, verbose):
    print(f"\n{'='*72}")
    print(f"FILE: {path}")
    print(f"{'='*72}")

    if not os.path.exists(path):
        print("  MISSING")
        return None

    size = os.path.getsize(path)
    print(f"  size on disk : {size/1024/1024:.2f} MB")

    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        print(f"  datasets     : {keys}")
        print(f"  attrs        : {dict(f.attrs)}")

        if "showers" not in f:
            print("  ERROR: no 'showers' dataset")
            return None

        n_showers = len(f["showers"])
        print(f"  n_showers    : {n_showers}")
        if n_showers == 0:
            print("  WARNING: empty file")
            return None

        nmax    = int(f.attrs.get("nmax",   -1))
        n_feat  = int(f.attrs.get("n_feat", -1))
        if n_feat <= 0:
            for j in range(min(n_showers, 100)):
                fl = f["showers"][j]
                np_j = int(f["num_points"][j])
                if np_j > 0 and fl.size % np_j == 0:
                    n_feat = fl.size // np_j
                    break
            print(f"  n_feat (inferred): {n_feat}")
        else:
            print(f"  nmax (attr)  : {nmax}")
            print(f"  n_feat (attr): {n_feat}  (4 = no time, 5 = with time)")

        # small datasets: load fully
        num_points = f["num_points"][:]
        energies   = f["energies"][:].ravel()
        directions = f["directions"][:]
        pdg        = f["pdg"][:]
        actual_pdg = f["actual_pdg"][:]

        print(f"\n  num_points stats:")
        print(f"    min={num_points.min()}  max={num_points.max()}  "
              f"mean={num_points.mean():.1f}  median={np.median(num_points):.0f}")
        n_at_cap = int((num_points >= nmax).sum()) if nmax > 0 else 0
        n_empty  = int((num_points == 0).sum())
        print(f"    showers at cap (num_points >= nmax): {n_at_cap} / {n_showers}")
        print(f"    empty showers (num_points == 0)    : {n_empty} / {n_showers}")

        print(_summary(energies, "energies"))
        print(_summary(directions, "directions"))
        dn = np.linalg.norm(directions, axis=1)
        print(f"  |direction|   : min={dn.min():.4g}  max={dn.max():.4g}  "
              f"mean={dn.mean():.4g}  (should be ~1.0)")
        print(f"  pdg classes   : {dict(Counter(pdg.tolist()))}")
        print(f"  actual_pdg    : {dict(Counter(actual_pdg.tolist()))}")

        # which indices to inspect
        if max_showers is None or max_showers >= n_showers:
            indices = np.arange(n_showers, dtype=np.int64)
        else:
            indices = np.linspace(0, n_showers - 1, max_showers, dtype=np.int64)
            indices = np.unique(indices)
        sample_n = len(indices)

        print(f"\n  Per-shower checks ({sample_n} showers, "
              f"chunk_size={chunk_size}):")

        totals = {
            "nan": 0, "inf": 0, "xy_oor": 0, "plane_oor": 0,
            "energy_oor": 0, "energy_neg": 0, "time_oor": 0,
            "duplicate": 0, "shape_mismatch": 0, "plane_non_int": 0,
        }
        examples = {k: [] for k in totals}
        all_skews = []
        showers_ds = f["showers"]

        n_processed = 0
        for chunk_start in range(0, sample_n, chunk_size):
            chunk_idxs = indices[chunk_start: chunk_start + chunk_size]

            for i in chunk_idxs:
                n_pts = int(num_points[i])
                if n_pts == 0:
                    continue
                flat = showers_ds[i]
                if flat.size != n_pts * n_feat:
                    totals["shape_mismatch"] += 1
                    if len(examples["shape_mismatch"]) < 3:
                        examples["shape_mismatch"].append(
                            (int(i), flat.size, n_pts * n_feat))
                    continue

                arr = flat.reshape(n_pts, n_feat)
                issues = _check_one_shower(arr, n_feat)

                for k in totals:
                    if k == "shape_mismatch":
                        continue
                    val = issues.get(k, 0)
                    if isinstance(val, int) and val > 0:
                        totals[k] += 1
                        if len(examples[k]) < 3:
                            if k == "xy_oor":
                                examples[k].append(
                                    (int(i),
                                     f"x[{arr[:,0].min():.3g},{arr[:,0].max():.3g}] "
                                     f"y[{arr[:,1].min():.3g},{arr[:,1].max():.3g}]"))
                            else:
                                examples[k].append((int(i), val))

                if issues["y_x_skew"]:
                    all_skews.extend(issues["y_x_skew"])

                n_processed += 1

            if verbose:
                done = min(chunk_start + chunk_size, sample_n)
                print(f"    ... {done}/{sample_n} showers checked", end="\r")

        print(f"    {n_processed}/{sample_n} showers checked       ")

        print(f"\n  ISSUES (showers affected, out of {sample_n}):")
        for k, v in totals.items():
            flag = "  <-- !" if v > 0 else ""
            print(f"    {k:22s}: {v}{flag}")

        for k, exlist in examples.items():
            if exlist:
                print(f"\n  {k} examples (idx, info):")
                for ex in exlist:
                    print(f"    {ex}")

        if all_skews:
            s = np.array(all_skews, dtype=np.float64)
            s_finite = s[np.isfinite(s)]
            print(f"\n  Per-plane y/x range ratio "
                  f"(stride-bug indicator; expect skew >> 1 for real showers):")
            if s_finite.size:
                med = float(np.median(s_finite))
                print(f"    median={med:.2f}  "
                      f"p90={np.percentile(s_finite, 90):.2f}  "
                      f"p99={np.percentile(s_finite, 99):.2f}  "
                      f"max={s_finite.max():.2f}")
                print(f"    fraction of planes with y/x > 1.5: "
                      f"{100*np.mean(s_finite > 1.5):.1f}%")
                print(f"    fraction of planes with y/x > 3.0: "
                      f"{100*np.mean(s_finite > 3.0):.1f}%")
                if med < 1.3:
                    print(f"    WARNING: median y/x ratio is suspiciously close "
                          f"to 1.0 -- signature of the stride bug.")

    return {
        "file": path, "n_showers": n_showers, "sampled": sample_n,
        "totals": totals,
        "skews_median": float(np.median(np.array(all_skews))) if all_skews else None,
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+",
                   help="H5 file(s) and/or directories to scan.")
    p.add_argument("--chunk-size", type=int, default=128,
                   help="How many showers per inner chunk (controls memory). "
                        "Default: 128.")
    p.add_argument("--max-showers", type=int, default=2000,
                   help="Max showers to deeply inspect per file. "
                        "Use -1 to inspect every shower. Default: 2000.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    max_showers = None if args.max_showers == -1 else args.max_showers

    files = []
    for x in args.paths:
        if os.path.isdir(x):
            files += sorted(glob(os.path.join(x, "*.h5")))
        else:
            files.append(x)

    if not files:
        print("No H5 files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {len(files)} file(s)  "
          f"(chunk_size={args.chunk_size}, max_showers={args.max_showers})")
    results = []
    for fp in files:
        try:
            r = inspect_one_file(fp, args.chunk_size, max_showers,
                                 verbose=not args.quiet)
            if r is not None:
                results.append(r)
        except Exception as e:
            print(f"\n  ERROR inspecting {fp}: {e}")

    print(f"\n{'='*72}")
    print("AGGREGATE SUMMARY")
    print(f"{'='*72}")
    grand = Counter()
    n_files_with_issues = 0
    for r in results:
        any_issue = False
        for k, v in r["totals"].items():
            grand[k] += v
            if v > 0:
                any_issue = True
        if any_issue:
            n_files_with_issues += 1

    print(f"  files scanned         : {len(results)}")
    print(f"  files with any issue  : {n_files_with_issues}")
    print(f"  total issue counts (showers affected, summed across files):")
    for k, v in grand.items():
        flag = "  <-- !" if v > 0 else ""
        print(f"    {k:22s}: {v}{flag}")

    skews = [r["skews_median"] for r in results if r["skews_median"] is not None]
    if skews:
        print(f"\n  per-file median y/x range ratio:")
        print(f"    files: min={min(skews):.2f}  "
              f"median={np.median(skews):.2f}  "
              f"max={max(skews):.2f}")
        if np.median(skews) < 1.3:
            print(f"    WARNING: most files show suspiciously low y/x skew.")

    sys.exit(1 if n_files_with_issues > 0 else 0)


if __name__ == "__main__":
    main()