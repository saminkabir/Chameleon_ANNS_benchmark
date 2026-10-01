from ann_search import *


import flatnav
from flatnav.data_type import DataType 
import numpy as np
import argparse
import pickle
pickle_path='/data/kabir/similarity-search/models/results_pickles_v7/'
import heapq

def refine_response(Q, B, I, k):
    refined_result = []
    for nq in range(0, Q.shape[0]):
        h = []
        for index in I[nq]:
            cv = B[index]
            dist = np.linalg.norm(cv - Q[nq])
            heapq.heappush(h, (dist, index))
        n_s = heapq.nsmallest(k, h)
        temp = []
        for x in n_s:
            temp.append(x[1])
        refined_result.append(temp)
    return np.array(refined_result)

def flat_nav_test(dataset,max_edges_per_node=32,ef_construction = 100, ef_search= 150):
    xb,xq,gt=get_data_generic(dataset)
    n= xb.shape[0]
    d = xb.shape[1]
    k = gt.shape[1]
    info={}
    info['model']='FlatNav'
    dataset_size = n
    dataset_dimension = d
    dataset_to_index=xb
    index = flatnav.index.create(
        distance_type="l2",
        index_data_type=DataType.float32,
        dim=dataset_dimension,
        dataset_size=dataset_size,
        max_edges_per_node=max_edges_per_node,
        verbose=True,
        collect_stats=True,
    )
    index.set_num_threads(128)
    search_start_time = time.time()
    index.add(data=dataset_to_index, ef_construction=ef_construction)
    search_end_time = time.time()
    info['construction_time']=search_end_time-search_start_time
    
    queries = xq
    search_start_time = time.time()
    distances, indices = index.search(queries=queries, ef_search=ef_search, K=k)
    search_end_time = time.time()
    info['search_time']=search_end_time-search_start_time
    r = calculate_recall_at(gt, indices, k, k)
    info['recall']=r
    info['max_edges_per_node']=max_edges_per_node
    info['ef_construction']=ef_construction
    info['ef_search']=ef_search
    info['I'] = indices
    # test_id='FLATNAV-'+dataset+'-'+str(max_edges_per_node)+'-'+str(ef_construction)+'-'+str(ef_search)+'.pkl'
    
    for i in range(2,6):
        search_start_time = time.time()
        distances_1, indices_1 = index.search(queries=queries, ef_search=ef_search, K=k*i)
        search_end_time = time.time()
        indices_1=refine_response(xq,xb,indices_1,k)
        r = calculate_recall_at(gt, indices_1, k, k)
        info['search_timex'+str(i)]=search_end_time-search_start_time
        info['recallx'+str(i)]=r
        info['Ix'+str(i)] = indices
    # with open(pickle_path+test_id, 'wb') as f:
        # pickle.dump(info, f)
    print(info)
    return info
    
    


parser = argparse.ArgumentParser(description="Run MRPT ANN search on a dataset")
parser.add_argument("--dataset", type=str, required=True,
                    help="Name of the dataset (e.g. 'imageNet', 'glove-100')")
parser.add_argument("--max_edges_per_node", type=int, required=True,
                    help="Threshold value for autotune sample (e.g. 0.8)")
parser.add_argument("--ef_construction", type=int, required=True,
                    help="Threshold value for autotune sample (e.g. 0.8)")
parser.add_argument("--ef_search", type=int, required=True,
                    help="Threshold value for autotune sample (e.g. 0.8)")

args = parser.parse_args()
flat_nav_test(args.dataset, args.max_edges_per_node, args.ef_construction,args.ef_search)
    