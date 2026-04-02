'''
python /n/home04/hhanif/AllShowers/allshowers/OT_match_with_scaling.py   /n/home04/hhanif/AllShowers/conf/transformer.yaml   --trafos /n/home04/hhanif/AllShowers/data/fitted_trafos.pt --reg 0.1 --sinkhorn-iters 50000
'''
import argparse
import multiprocessing
import os
import sys
import time
from collections.abc import Iterable, Iterator

import numpy as np
import numpy.typing as npt
import showerdata
import torch
import yaml
import ot

from allshowers import preprocessing  # noqa

start = time.time()


def print_time(*args, **kwargs) -> None:
    elapsed = time.time() - start
    print(f"[{elapsed: 7.2f}s]", *args, **kwargs)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("--trafos", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--reg", type=float, default=0.05)
    parser.add_argument("--sinkhorn-iters", type=int, default=1000)
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args(args)


# ---------------------------------------------------------
# PreProcessor
# ---------------------------------------------------------

class PreProcessor:

    def __init__(self, config_file: str, trafos_path_override=None):
        with open(config_file) as file:
            config = yaml.safe_load(file)

        self.file_path = config["data"]["path"]
        self.data_shape = showerdata.get_file_shape(self.file_path)

        trafos_path = trafos_path_override or config["data"].get("fitted_trafos")
        pack = torch.load(trafos_path, map_location="cpu", weights_only=False)

        if isinstance(pack, dict) and "trafos" in pack:
            pack = pack["trafos"]

        self.samples_coordinate_trafo = pack["samples_coordinate_trafo"]
        self.samples_energy_trafo = pack["samples_energy_trafo"]


    def __call__(self, x: npt.NDArray[np.float32]):
        x_tensor = torch.from_numpy(x).float()

        mask = x_tensor[:, 3] > 0.0

        coords = x_tensor[:, :2].permute(0, 2, 1)
        coords = self.samples_coordinate_trafo(coords)
        x_tensor[:, :2] = coords.permute(0, 2, 1)

        x_tensor[:, 3] = self.samples_energy_trafo(x_tensor[:, 3])

        layer = (x_tensor[:, 2] + 0.5).long()
        points = x_tensor[:, [0, 1, 3]]

        return points, mask, layer


# ---------------------------------------------------------
# DataLoader
# ---------------------------------------------------------

class DataLoader(Iterable):

    def __init__(self, data_file, batch_size):
        self.file_name = data_file
        self.batch_size = batch_size

    def __iter__(self):
        with showerdata.ShowerDataFile(self.file_name, "r") as file:
            for start in range(0, len(file), self.batch_size):
                end = min(start + self.batch_size, len(file))
                samples = file[start:end].points[:, :, :4]
                yield samples.transpose(0, 2, 1).astype(np.float32)


# ---------------------------------------------------------
# NoiseMatcher using ot.sinkhorn
# ---------------------------------------------------------

class NoiseMatcher:

    def __init__(self, pre_processor, device, reg=0.05, sinkhorn_iters=1000):
        self.pre_processor = pre_processor
        self.device = device
        self.num_layers = 24
        self.reg = reg
        self.sinkhorn_iters = sinkhorn_iters

        # Keep transforms on CPU; preprocessing always runs on CPU
        self.pre_processor.samples_coordinate_trafo.cpu()
        self.pre_processor.samples_energy_trafo.cpu()

    @torch.inference_mode()
    def __call__(self, samples):

        # Preprocessing runs on CPU
        points, mask, layer = self.pre_processor(samples)

        # Move results to target device
        points = points.to(self.device)
        mask = mask.to(self.device)
        layer = layer.to(self.device)

        B, _, P = points.shape
        noise = torch.randn((B, 3, P), device=self.device)

        for i in range(self.num_layers):
            mask_local = mask & (layer == i)

            for j in range(B):

                idx = torch.nonzero(mask_local[j], as_tuple=False).squeeze(1)
                N = int(idx.numel())

                if N <= 1:
                    continue

                points_j = points[j, :, idx].T
                noise_j = noise[j, :, idx].T

                # Cost matrix
                M = torch.cdist(points_j, noise_j, p=2)

                a = torch.ones(N, device=self.device) / N
                b = torch.ones(N, device=self.device) / N

                T = ot.emd(
                    a, b, M,
                    numItermax=self.sinkhorn_iters,
                )

                mapped = (T @ noise_j) * N
                noise[j, :, idx] = mapped.T

        noise = noise.masked_fill((~mask)[:, None, :], 0.0)

        return noise.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------
# Processing
# ---------------------------------------------------------

def process_file(
    data_file,
    data_shape,
    pre_processor,
    batch_size,
    reg,
    sinkhorn_iters,
    force_cpu,
):

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
    )

    print_time(f"Using device: {device}")

    noise_matcher = NoiseMatcher(
        pre_processor,
        device,
        reg=reg,
        sinkhorn_iters=sinkhorn_iters,
    )

    noise = np.empty((data_shape[0], 3, data_shape[1]), dtype=np.float32)

    loader = DataLoader(data_file, batch_size)

    if device.type == "cuda":
        # NO multiprocessing on GPU
        for i, batch in enumerate(loader):
            out = noise_matcher(batch)
            noise[i * batch_size:i * batch_size + len(batch)] = out
    else:
        with multiprocessing.Pool(os.cpu_count() - 1) as pool:
            for i, out in enumerate(pool.imap(noise_matcher, loader)):
                noise[i * batch_size:i * batch_size + len(out)] = out

    noise = noise.transpose(0, 2, 1)
    showerdata.save_target(noise, data_file, overwrite=True)

    print_time("Noise saved successfully.")


@torch.inference_mode()
def main(args=None):

    parsed_args = parse_args(args)

    pre_processor = PreProcessor(
        parsed_args.file,
        trafos_path_override=parsed_args.trafos,
    )

    process_file(
        data_file=pre_processor.file_path,
        data_shape=pre_processor.data_shape,
        pre_processor=pre_processor,
        batch_size=parsed_args.batch_size,
        reg=parsed_args.reg,
        sinkhorn_iters=parsed_args.sinkhorn_iters,
        force_cpu=parsed_args.force_cpu,
    )


if __name__ == "__main__":
    main()