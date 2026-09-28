from ann_search import *


import argparse

parser = argparse.ArgumentParser(description="Run MRPT ANN search on a dataset")
parser.add_argument("--dataset", type=str, required=True,
                    help="Name of the dataset (e.g. 'imageNet', 'glove-100')")
parser.add_argument("--m", type=int, required=True,
                    help="Threshold value for autotune sample (e.g. 0.8)")
parser.add_argument("--ef_search", type=int, required=True,
                    help="Threshold value for autotune sample (e.g. 0.8)")
parser.add_argument("--ef_construction", type=int, required=True,
                    help="Threshold value for autotune sample (e.g. 0.8)")

# parser.add_argument("--cpu", type=int, required=True,
#                     help="Threshold value for autotune sample (e.g. 0.8)")
args = parser.parse_args()
faiss.omp_set_num_threads(1)
os.environ["OPENBLAS_NUM_THREADS"] = str(1)

HNSW2(128,args.m,dataset_name=args.dataset,k=100,ef_search=args.ef_search,ef_construction=args.ef_construction, cpu=1)

# taskset --cpu-list [cpu]-[cpu] python ann_search_hnsw.py --dataset glove --m 32 --ef_search 64 --ef_construction 100
# taskset --cpu-list [cpu1]-[cpu2] python ann_search_hnsw.py --dataset [dataset] --m [m] --ef_search [ef_search] --ef_construction [ef_construction]
