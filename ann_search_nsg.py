from ann_search import *


import argparse

parser = argparse.ArgumentParser(description="Run MRPT ANN search on a dataset")
parser.add_argument("--dataset", type=str, required=True,
                    help="Name of the dataset (e.g. 'imageNet', 'glove-100')")
parser.add_argument("--m", type=int, required=True,
                    help="Threshold value for autotune sample (e.g. 0.8)")



parser.add_argument("--cpu", type=int, required=True,
                    help="Number of CPU threads to use")

args = parser.parse_args()

faiss.omp_set_num_threads(args.cpu)
os.environ["OPENBLAS_NUM_THREADS"] = str(args.cpu)

NSG2(dataset_name=args.dataset,m=args.m,cpu=args.cpu)
    