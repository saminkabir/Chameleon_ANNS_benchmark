from ann_search import *

import argparse
import os
import faiss

parser = argparse.ArgumentParser(description="Run FAISS IVF ANN search on a dataset")

parser.add_argument(
    "--dataset", type=str, required=True,
    help="Name of the dataset (e.g. 'imageNet', 'glove-100')"
)

parser.add_argument(
    "--nlist", type=int, required=True,
    help="Number of IVF clusters (e.g. 1024)"
)

parser.add_argument(
    "--nprobe", type=int, required=True,
    help="Number of IVF clusters searched per query (e.g. 16)"
)

parser.add_argument(
    "--cpu", type=int, required=True,
    help="Number of CPU threads (e.g. 1)"
)

args = parser.parse_args()

# Set CPU threads
faiss.omp_set_num_threads(args.cpu)
os.environ["OPENBLAS_NUM_THREADS"] = str(args.cpu)

# Run FAISS IVF
IVF(
    128,
    args.nlist,
    dataset_name=args.dataset,
    k=100,
    nprobe=args.nprobe
)