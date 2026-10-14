from svs import common
import numpy as np
import svs as ps
import os
import math
from data_extractor import *


def genGroundTrouth(dataset, K=None, query_batch=64, out_name="groundtruth_ip.ivecs"):
    """Exact top-K maximum inner product (MIPS) ground truth.

    Neighbors are ranked by <base, query>, largest first. Each row of the
    output ivecs file lists the K base ids for one query, best first.

    K           -- neighbors per query; defaults to the width of the existing
                   groundtruth.ivecs, as in the L2 version
    query_batch -- queries scored per matrix multiply; peak memory is about
                   N * query_batch * 4 bytes (1M base vectors, 64 queries -> 256 MB)
    out_name    -- written next to the dataset; kept separate from the
                   L2 groundtruth.ivecs so that file is not overwritten
    """
    os.chdir(dataset)
    base, query, groundtruth = get_data_generic(dataset)
    if K is None:
        K = groundtruth.shape[1]

    base = np.ascontiguousarray(base, dtype=np.float32)
    query = np.ascontiguousarray(query, dtype=np.float32)
    N, nq = base.shape[0], query.shape[0]
    K = min(K, N)
    print(base.shape, query.shape, "K =", K)

    ground_trouth = np.empty((nq, K), dtype=np.uint32)
    for start in range(0, nq, query_batch):
        end = min(start + query_batch, nq)
        # scores[i, j] = <base[i], query[start + j]>
        scores = base @ query[start:end].T
        for j in range(end - start):
            s = scores[:, j]
            # top-K largest scores, unordered
            top = np.argpartition(-s, K - 1)[:K]
            # order by score descending, ties broken by smaller id
            order = np.lexsort((top, -s[top].astype(np.float64)))
            ground_trouth[start + j] = top[order]
        print(f"\r{end}/{nq} queries done", end="", flush=True)
    print()

    out_dir = os.path.join(dataset_path, dataset)
    common.write_vecs(ground_trouth, out_name, skip_check=False)
    os.chdir('..')
    return ground_trouth


import argparse

parser = argparse.ArgumentParser(description="Run MRPT ANN search on a dataset")
parser.add_argument("--dataset", type=str, required=True,
                    help="Name of the dataset (e.g. 'imageNet', 'glove-100')")

args = parser.parse_args()
os.chdir('/home/cc/datasets/datasets')
genGroundTrouth(args.dataset)

