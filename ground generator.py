from pysvs import common
import numpy as np
import heapq
import pysvs as ps
import os
import math
from data_extractor import *

            

def genGroundTrouth(dataset):
    os.chdir(dataset)
    
    groundtruth_ivec = "groundtruth.ivecs"
    base,query,groundtruth = get_data_generic(dataset)
    K = groundtruth.shape[1]
    ground_trouth = []
    print(base.shape, query.shape)
    for q in range(0,query.shape[0]):
        xq = query[q]
        heap = []
        heapq.heapify(heap)
        for b in range (0,base.shape[0]):
            xb = base[b]
            minus = xb - xq
            distance = np.dot(minus.T, minus)
            heapq.heappush(heap, (-distance,b))
            if len(heap)>K:
                heapq.heappop(heap)
        res = []
        for node in heapq.nlargest(K, heap, key=None):
            res.append(node[1])
        ground_trouth.append(res)
    ground_trouth=np.array(ground_trouth, dtype=np.uint32)
    common.write_vecs(ground_trouth, groundtruth_ivec, skip_check = False)
    os.chdir('..')