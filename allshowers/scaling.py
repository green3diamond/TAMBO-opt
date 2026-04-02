#!/usr/bin/env python3
"""
Fit coordinate + energy transforms from an AllShowers HDF5 file
and save the fitted trafos into a .pt file.

NO dataset saving.
NO transformation pass.
Only fitting.

Usage:
  python /n/home04/hhanif/AllShowers/allshowers/scaling.py /n/home04/hhanif/AllShowers/conf/transformer.yaml \
      --out /n/home04/hhanif/AllShowers/data/fitted_trafos.pt \
      --fit-stop 100000
"""

from __future__ import annotations

import argparse
import yaml
import torch
import showerdata

from allshowers import preprocessing


def parse_args():
    p = argparse.ArgumentParser(description="Fit trafos and save them")
    p.add_argument("config", type=str, help="Path to YAML config")
    p.add_argument("--out", type=str, required=True, help="Output .pt file")
    p.add_argument(
        "--fit-stop",
        type=int,
        default=100000,
        help="Number of showers to use for fitting (0 = full file)",
    )
    return p.parse_args()


def load_conf(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def main():
    args = parse_args()
    conf = load_conf(args.config)

    data_path = conf["data"]["path"]
    shape = showerdata.get_file_shape(data_path)
    total_len = int(shape[0])

    fit_stop = args.fit_stop
    if fit_stop == 0:
        fit_stop = total_len
    fit_stop = min(fit_stop, total_len)

    print(f"[info] data.path  = {data_path}")
    print(f"[info] total_len = {total_len}")
    print(f"[info] fit_stop  = {fit_stop}")

    # -------------------------
    # Load data for fitting
    # -------------------------
    print("[info] loading data for fitting...")
    data = showerdata.load(path=data_path, stop=fit_stop).points[:, :, :4]
    x = torch.from_numpy(data).to(torch.float32)

    mask = x[:, :, 3] > 0.0

    # -------------------------
    # Build trafos from YAML
    # -------------------------
    coord_trafo = preprocessing.compose(
        transformation=conf["data"]["samples_coordinate_trafo"]
    )
    energy_trafo = preprocessing.compose(
        transformation=conf["data"]["samples_energy_trafo"]
    )

    coord_trafo.to(x.dtype)
    energy_trafo.to(x.dtype)

    # -------------------------
    # Fit coordinate trafo
    # -------------------------
    print("[info] fitting coordinate trafo...")
    coord_trafo.fit(
        x=x[:, :, :2],
        mask=mask[:, :, None].repeat(1, 1, 2),
    )

    # -------------------------
    # Fit energy trafo
    # -------------------------
    print("[info] fitting energy trafo...")
    energy_trafo.fit(
        x=x[:, :, 3],
        mask=mask,
    )

    # -------------------------
    # Save
    # -------------------------
    torch.save(
        {
            "samples_coordinate_trafo": coord_trafo.cpu(),
            "samples_energy_trafo": energy_trafo.cpu(),
            "meta": {
                "config": args.config,
                "data_path": data_path,
                "fit_stop": fit_stop,
            },
        },
        args.out,
    )

    print(f"[info] saved fitted trafos → {args.out}")


if __name__ == "__main__":
    main()