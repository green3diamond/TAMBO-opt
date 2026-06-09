import h5py
import numpy as np
from multiprocessing import Pool, cpu_count

FILE_PATH = "/n/home04/hhanif/train_130k.h5"
CHUNK_SIZE = 20_000


def process_chunk(args):
    start, end = args

    with h5py.File(FILE_PATH, "r") as f:
        showers = f["showers"]

        flat = np.concatenate(showers[start:end])

        if flat.size < 3:
            return -np.inf

        return np.nanmax(flat[2::4])


if __name__ == "__main__":

    with h5py.File(FILE_PATH, "r") as f:
        n_events = len(f["showers"])

    ranges = [
        (start, min(start + CHUNK_SIZE, n_events))
        for start in range(0, n_events, CHUNK_SIZE)
    ]

    with Pool(cpu_count()) as pool:
        maxima = pool.map(process_chunk, ranges)

    print(np.nanmax(maxima))