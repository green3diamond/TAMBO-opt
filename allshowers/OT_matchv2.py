import argparse
import multiprocessing as mp
import os
import time

import numpy as np
import ot
import showerdata
import torch
import yaml

from allshowers import preprocessing


start_time = time.time()


def log(*args):
    print(f"[{time.time()-start_time:7.2f}s]", *args, flush=True)


def parse_args():
    p = argparse.ArgumentParser("OT match for ONE split file (uses save_target).")
    p.add_argument("--config", required=True)
    p.add_argument("--file", required=True, help="Split HDF5 file path")

    # NEW: file used ONLY for fitting transforms
    p.add_argument(
        "--fit-file",
        default="/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_processed_step1_v3/merged_all_showers_randomized.h5",
        help="HDF5 file used ONLY for fitting the preprocessing transforms.",
    )

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-proc", type=int, default=4)
    p.add_argument("--emd-numitermax", type=int, default=300000)
    p.add_argument("--fit-stop", type=int, default=100000)
    return p.parse_args()


###############################################################################
# PreProcessor
###############################################################################
class PreProcessor:
    def __init__(self, config_file: str, fit_file: str, fit_stop: int):
        with open(config_file) as f:
            cfg = yaml.safe_load(f)

        self.samples_energy_trafo = preprocessing.compose(
            cfg["data"]["samples_energy_trafo"]
        )
        self.samples_coordinate_trafo = preprocessing.compose(
            cfg["data"]["samples_coordinate_trafo"]
        )

        log("Fitting transforms...")
        log(f"  fit_file = {fit_file}")
        log(f"  fit_stop = {fit_stop}")

        showers = showerdata.load(path=fit_file, stop=fit_stop)
        pts = showers.points[:, :, :4]
        pts_t = torch.from_numpy(pts)

        mask = pts_t[:, :, 3] > 0.0
        self.samples_coordinate_trafo.to(pts_t.dtype)
        self.samples_energy_trafo.to(pts_t.dtype)

        self.samples_coordinate_trafo.fit(
            x=pts_t[:, :, :2],
            mask=mask[:, :, None].repeat(1, 1, 2),
        )
        self.samples_energy_trafo.fit(
            x=pts_t[:, :, 3],
            mask=mask,
        )

        layer = (pts_t[:, :, 2] + 0.5).to(torch.int64)
        self.num_layers = int(torch.max(layer).item() + 1)

    def __call__(self, x_bfp: np.ndarray):
        # x_bfp: (B,F,P)
        x = torch.from_numpy(x_bfp[:, :4, :]).permute(0, 2, 1)
        mask = x[:, :, 3] > 0.0

        x[:, :, :2] = self.samples_coordinate_trafo(x[:, :, :2])
        x[:, :, 3] = self.samples_energy_trafo(x[:, :, 3])

        layer = (x[:, :, 2] + 0.5).to(torch.int64)
        points = x[:, :, [0, 1, 3]]

        return points.numpy(), mask.numpy(), layer.numpy()


###############################################################################
# Noise matcher (EMD)
###############################################################################
class NoiseMatcherEMD:
    def __init__(self, pre: PreProcessor, emd_numitermax: int):
        self.pre = pre
        self.num_layers = pre.num_layers
        self.emd_numitermax = emd_numitermax

    def __call__(self, x_bfp: np.ndarray):
        points, mask, layer = self.pre(x_bfp)

        B, P, D = points.shape
        noise = np.random.randn(B, P, D).astype(np.float32)

        for li in range(self.num_layers):
            in_layer = (layer == li) & mask

            for j in range(B):
                idx = np.where(in_layer[j])[0]
                if idx.size <= 1:
                    continue

                pj = points[j, idx].astype(np.float64)
                nj = noise[j, idx].astype(np.float64)

                N = pj.shape[0]
                M = np.sqrt(np.sum((pj[:, None] - nj[None, :]) ** 2, axis=-1))

                wa = np.ones(N) / N
                wb = np.ones(N) / N

                T = ot.emd(wa, wb, M, numItermax=self.emd_numitermax)
                noise[j, idx] = (N * (T @ nj)).astype(np.float32)

        noise[~mask] = 0.0
        return noise


###############################################################################
# Batch loader
###############################################################################
def iter_batches(file_path, batch_size):
    with showerdata.ShowerDataFile(file_path, "r") as f:
        n = len(f)

        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            pts = f[s:e].points
            yield pts.transpose(0, 2, 1).astype(np.float32)


###############################################################################
# Main
###############################################################################
@torch.inference_mode()
def main():
    args = parse_args()

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    log("Initializing PreProcessor (fit transforms from --fit-file)")
    pre = PreProcessor(args.config, args.fit_file, args.fit_stop)

    N, P, _ = showerdata.get_file_shape(args.file)
    log(f"Split file shape: N={N}, P={P}")
    log(f"OT input/output file (--file): {args.file}")

    matcher = NoiseMatcherEMD(pre, args.emd_numitermax)

    log("Allocating target array (RAM required)")
    target = np.zeros((N, P, 3), dtype=np.float32)

    log("Running OT matching...")
    with mp.Pool(args.num_proc) as pool:
        for k, batch_noise in enumerate(
            pool.imap(matcher, iter_batches(args.file, args.batch_size))
        ):
            start = k * args.batch_size
            target[start : start + len(batch_noise)] = batch_noise

            if k % 5 == 0:
                log(f"Processed batch {k}")

    log("Saving target to split file...")
    showerdata.save_target(target, args.file, overwrite=True)

    log("Done.")


if __name__ == "__main__":
    main()