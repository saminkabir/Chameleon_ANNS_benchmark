# scp ANN-search.py kabir@164.107.127.101:/data/kabir/similarity-search/models/faiss-library
import faiss
import time
import sys
from data_extractor import *
import heapq
import numpy as np
# from bolt import bolt_api as b
from annoy import AnnoyIndex
# import nanopq
# import diskannpy as dap

import os
from pathlib import Path
import matplotlib.pyplot as plt 
import os
import time
import os
# import glasspy as glass
# Set the number of threads for OpenBLAS
import scann
# from scipy.stats import bartlett, levene
index_path = '/data/kabir/similarity-search/diskANNIndexes/'
from pympler import asizeof
import pickle
pickle_path='/data/kabir/similarity-search/models/results_pickles_rebuttal/'

# log_file_name = "log1.txt"
# sys.stdout = open("logs/" + log_file_name, "a")
# print('Reach here 2')

def calculate_recall_at(ground_truth, I, k1, k2):
    return ps.k_recall_at(ground_truth, I, k1, k2)

def refine_response(Q, B, I, k):
    refined_result = []
    for nq in range(0, Q.shape[0]):
        h = []
        for index_ in I[nq]:
            index_=int(index_)
            cv = B[index_]
            dist = np.linalg.norm(cv - Q[nq])
            heapq.heappush(h, (dist, index_))
        n_s = heapq.nsmallest(k, h)
        temp = []
        for x in n_s:
            temp.append(x[1])
        refined_result.append(temp)
    return np.array(refined_result)
            
             
        

def calculate_map_at(ground_truth, I, k):
    mp = 0.0
    for q in range(0, ground_truth.shape[0]):
        ap = 0.0
        top_k_set = set(ground_truth[q])
        top_r_set = set()
        for r in range(1, k + 1):
            top_r_set.add(ground_truth[q][r-1])
            isRExact = I[q][r-1] in top_k_set
            if isRExact:
                ct = 0
                for i in range(0,r):
                    if (I[q][i] in top_r_set):
                        ct = ct + 1
                ap = ap + ct/r
        mp = mp + ap/k
    return mp/ground_truth.shape[0]               


def calculate_recall(ground_truth, I, k):
    ans = 0.0
    for q in range(0, ground_truth.shape[0]):
        truenn = I[q][0];
        for i in range(0,k):
            if ground_truth[q][i]==truenn:
                ans=ans+1
                break
    return ans/ground_truth.shape[0]

def calculate_recall_v2(ground_truth, I, k):
    ans = 0.0
    for q in range(0, ground_truth.shape[0]):
        for _k in range(0,k):
            for i in range(0,k):
                if ground_truth[q][i]==I[q][_k]:
                    ans=ans+1
                    break
    return ans/(ground_truth.shape[0]*k)    

def calculate_recall_dist(ground_truth, I, k):
    ans = 0.0
    for q in range(0, ground_truth.shape[0]):
        for j in range(0,k):
            truenn = I[q][j]
            for i in range(0,k):
                if ground_truth[q][i]>=truenn:
                    ans=ans+1
                    break
    return ans/(ground_truth.shape[0]*k)

def HNSW(data_loader, d, m, dataset_name, k = 100, ef_search = 8, ef_construction = 64):
    xb, xq, gt= data_loader()
    d = xb.shape[1]
    k = gt.shape[1]
    # set HNSW index parameters
    M = m  # number of connections each vertex will have
    train_start_time = time.time();
    index = faiss.IndexHNSWFlat(d, M)
    # set efConstruction and efSearch parameters
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    # add data to index
    index.add(xb)
    train_end_time = time.time();
    search_start_time = time.time()
    D_hnsw, I_hnsw = index.search(xq, k)
    search_end_time = time.time();
    r = calculate_recall_at(gt, I_hnsw, k, k)
    dict = {}
    dict['recall@'] = r
    dict['map@'] = calculate_map_at(gt, I_hnsw, k)
    dict['model_name'] = 'HNSW'
    dict['dataset_name'] = dataset_name
    dict['search-time'] =  search_end_time - search_start_time
    dict['training-time'] =  train_end_time - train_start_time
    dict['d'] = d
    dict['m'] = m
    dict['k'] = k
    dict['ef_search'] = ef_search
    dict['ef_construction'] = ef_construction
    return dict


def HNSW2(d, m, dataset_name, k = 100, ef_search = 8, ef_construction = 64, cpu=1):
    xb, xq, gt= get_data_common(dataset_name)
    d = xb.shape[1]
    k = gt.shape[1]
    # set HNSW index parameters
    M = m  # number of connections each vertex will have
    for sd in [1331]:
        train_start_time = time.time();
        index = faiss.IndexHNSWFlat(d, M)
        # set efConstruction and efSearch parameters
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
        # add data to index
        index.add(xb)
        train_end_time = time.time();
        search_start_time = time.time()
        D_hnsw, I_hnsw = index.search(xq, k)
        search_end_time = time.time();
        r = calculate_recall_at(gt, I_hnsw, k, k)
        dict = {}
        dict['recall@'] = r
        dict['map@'] = calculate_map_at(gt, I_hnsw, k)
        dict['model_name'] = 'HNSW'
        dict['dataset_name'] = dataset_name
        dict['search-time'] =  search_end_time - search_start_time
        dict['training-time'] =  train_end_time - train_start_time
        dict['d'] = d
        dict['m'] = m
        dict['k'] = k
        dict['ef_search'] = ef_search
        dict['ef_construction'] = ef_construction
        
        dict['I'] = I_hnsw
        for i in range(2,6):
            search_start_time = time.time()
            distances_1, indices_1 = index.search(xq, k*i)
            search_end_time = time.time()
            indices_1=refine_response(xq,xb,indices_1,k)
            r = calculate_recall_at(gt, indices_1, k, k)
            dict['search_timex'+str(i)]=search_end_time-search_start_time
            r = calculate_recall_at(gt, indices_1, k, k)
            dict['recallx'+str(i)]=r
            dict['Ix'+str(i)] = indices_1
    print(dict)

        # test_id='HNSW-SEED-EXPERIMENT-'+dataset_name+'-'+str(m)+'-'+str(ef_construction)+'-'+str(ef_search)+'-'+str(cpu)+'-'+str(sd)+'.pkl'
        # with open(pickle_path+test_id, 'wb') as f:
        #     pickle.dump(dict,f)

# def HNSW_GLASS(d, m, dataset_name, k = 100, ef_search = 8, ef_construction = 64):
#     xb, xq, gt= get_data_common(dataset_name)
#     d = xb.shape[1]
#     k = gt.shape[1]
#     # set HNSW index parameters
#     M = m  # number of connections each vertex will have
#     train_start_time = time.time();
#     # index = faiss.IndexHNSWFlat(d, M)
#     index = glass.Index(index_type="HNSW", metric="L2", R=M, L=ef_construction)
#     # set efConstruction and efSearch parameters
#     # index.hnsw.efConstruction = ef_construction
#     # index.hnsw.efSearch = ef_search
#     # add data to index
#     # index.add(xb)
#     graph = index.build(xb)
#     searcher = glass.Searcher(graph=graph, data=xb, metric="L2", quantizer="SQ4U")
#     searcher.set_ef(ef_search)
#     searcher.optimize()
#     train_end_time = time.time();
#     search_start_time = time.time()
#     ret = searcher.batch_search(query=xq, k=k)
#     I=np.array(ret[0])
#     search_end_time = time.time();
#     r = calculate_recall_at(gt, I, k, k)
#     dict = {}
#     dict['recall@'] = r
#     dict['map@'] = calculate_map_at(gt, I, k)
#     dict['model_name'] = 'GLASS-HNSW'
#     dict['dataset_name'] = dataset_name
#     dict['search-time'] =  search_end_time - search_start_time
#     dict['training-time'] =  train_end_time - train_start_time
#     dict['d'] = d
#     dict['m'] = m
#     dict['k'] = k
#     dict['ef_search'] = ef_search
#     dict['ef_construction'] = ef_construction
#     return dict

# def NSG_GLASS(d, m, dataset_name, k = 100, ef_search = 8, ef_construction = 64):
#     xb, xq, gt= get_data_common(dataset_name)
#     d = xb.shape[1]
#     k = gt.shape[1]
#     # set HNSW index parameters
#     M = m  # number of connections each vertex will have
#     train_start_time = time.time();
#     # index = faiss.IndexHNSWFlat(d, M)
#     index = glass.Index(index_type="NSG", metric="L2", R=M, L=ef_construction)
#     # set efConstruction and efSearch parameters
#     # index.hnsw.efConstruction = ef_construction
#     # index.hnsw.efSearch = ef_search
#     # add data to index
#     # index.add(xb)
#     graph = index.build(xb)
#     searcher = glass.Searcher(graph=graph, data=xb, metric="L2", quantizer="SQ4U")
#     searcher.set_ef(ef_search)
#     searcher.optimize()
#     train_end_time = time.time();
#     search_start_time = time.time()
#     ret = searcher.batch_search(query=xq, k=k)
#     I=np.array(ret[0])
#     search_end_time = time.time();
#     r = calculate_recall_at(gt, I, k, k)
#     dict = {}
#     dict['recall@'] = r
#     dict['map@'] = calculate_map_at(gt, I, k)
#     dict['model_name'] = 'GLASS-NSG'
#     dict['dataset_name'] = dataset_name
#     dict['search-time'] =  search_end_time - search_start_time
#     dict['training-time'] =  train_end_time - train_start_time
#     dict['d'] = d
#     dict['m'] = m
#     dict['k'] = k
#     dict['ef_search'] = ef_search
#     dict['ef_construction'] = ef_construction
#     return dict

def NSG(data_loader, d, m, dataset_name, k = 100, perams=[16, 32, 64]):
    xb, xq, gt= data_loader()
    d = xb.shape[1]
    k = gt.shape[1]
    # set HNSW index parameters
    M = m  # number of connections each vertex will have
    train_start_time = time.time();
    index = faiss.IndexNSGFlat(d, M)
    # add data to index
    index.add(xb)
    train_end_time = time.time();
    dicts = []
    for search_L in perams:
        search_start_time = time.time()
        index.nsg.search_L = search_L
        D_nsg, I_nsg = index.search(xq, k)
        search_end_time = time.time();
        r = calculate_recall_at(gt, I_nsg, k, k)
        dict = {}
        dict['recall@'] = r
        dict['map@'] = calculate_map_at(gt, I_nsg, k)
        dict['model_name'] = 'NSG'
        dict['dataset_name'] = dataset_name
        dict['search-time'] =  search_end_time - search_start_time
        dict['training-time'] =  train_end_time - train_start_time
        dict['d'] = d
        dict['m'] = m
        dict['k'] = k
        dict['search_L'] = search_L
        dicts.append(dict)
    return dicts

def NSG2(dataset_name, m, cpu, perams=[8, 16, 32, 64]):
    xb, xq, gt= get_data_common(dataset_name)
    d = xb.shape[1]
    k = gt.shape[1]
    # set HNSW index parameters
    M = m  # number of connections each vertex will have
    train_start_time = time.time();
    index = faiss.IndexNSGFlat(d, M)
    # add data to index
    index.add(xb)
    train_end_time = time.time()
    Iss=[]
    for refine in [1,2]:
        for search_L in perams:
            Is=[]
            for node in range(1,3):
                search_start_time = time.time()
                index.nsg.search_L = search_L
                index.nsg.enterpoint = node
                D_nsg, I_nsg = index.search(xq, k)
                I_nsg = refine_response(xq, xb, I_nsg, k)
                search_end_time = time.time();
                r = calculate_recall_at(gt, I_nsg, k, k)
                dict = {}
                dict['recall@'] = r
                dict['map@'] = calculate_map_at(gt, I_nsg, k)
                dict['model_name'] = 'NSG'
                dict['dataset_name'] = dataset_name
                dict['search-time'] =  search_end_time - search_start_time
                dict['training-time'] =  train_end_time - train_start_time
                dict['d'] = d
                dict['m'] = m
                dict['k'] = k
                dict['search_L'] = search_L
                dict['refine'] = refine
                dict['I'] = I_nsg
                test_id='NSG-'+dataset_name+'-'+str(m)+'-'+str(search_L)+'-'+str(refine)+'-'+str(node)+'-'+str(cpu)+'.pkl'
                with open(pickle_path+test_id, 'wb') as f:
                    pickle.dump(dict, f)
        
        
    
def PQ(data_loader, version, d, m, nbits, dataset_name, k = 20, refine = None):
    xb, xq, gt= data_loader(m)
    k = gt.shape[1]
    return (PQ_PROXY(data_loader, version, d, m, nbits, dataset_name, k, k),
            PQ_PROXY(data_loader, version, d, m, nbits, dataset_name, k, k*2))
    
def compute_closest_distances(xb,xq,gt):
    gt_dist = np.zeros([gt.shape[0], gt.shape[1]])
    for q in range(0,gt.shape[0]):
        x_q=xq[q]
        for i in range(0,gt.shape[1]):
            x_i = xb[gt[q][i]]
            minus = x_i - x_q
            distance = np.dot(minus.T, minus)
            # print(i,q,gt[q][i],distance)
            gt_dist[q][i]=distance
    return gt_dist

def PQ_PROXY(dataset_name, version, m, nbits, refine = None):
    xb, xq, gt= get_data_generic(dataset_name)
    gt_dist=compute_closest_distances(xb,xq,gt)
    d = xb.shape[1]
    k = gt.shape[1]
    if (xb.shape[1]%m)!=0:
        subSpaceSize = xb.shape[1]/m
        subSpaceSize = math.ceil(subSpaceSize)
        d = subSpaceSize * m
        xb = extend_time_series_with_zeros(xb,subSpaceSize*m)
        xq = extend_time_series_with_zeros(xq,subSpaceSize*m)
    if refine == None:
        refine = k
    index = None
    
    if version == 'PQ':
        index = faiss.IndexPQ(d, m, nbits)
    else:
        s  = "OPQ"+str(int(m))+",PQ"+str(int(m))+"x"+str(nbits)
        index = faiss.index_factory(d, s)
    index.is_trained
    train_start_time = time.time();
    index.train(xb)
    train_end_time = time.time();
    index.add(xb)
    encoding_end_time = time.time()
    dicts=[]
    for multiplier in range(1,6):
        refine=k*multiplier
        search_start_time = time.time();
        D_pq, I_pq = index.search(xq, refine)
        search_end_time = time.time();
        dict = {}
        I_pq = refine_response(xq, xb, I_pq, k)
        r = calculate_recall_at(gt, I_pq, k, k)
        dict['recall@'+'x'+str(refine)] = r
        dict['map@'] = calculate_map_at(gt, I_pq, k)
        dict['recall'] = calculate_recall(gt, I_pq, k)
        dict['model_name'] = version
        dict['dataset_name'] = dataset_name
        dict['training-time'] =  train_end_time - train_start_time
        dict['encoding-time'] =  encoding_end_time - train_end_time
        dict['construction-time'] =  encoding_end_time - train_start_time
        dict['search-time'+'x'+str(refine)] =  search_end_time - search_start_time
        dict['m'] = m
        dict['nbits'] = nbits
        dict['k'] = k
        dict['refine'] = refine
        dict['I']=I_pq
        dict['D']=D_pq
    print(dict)
    return dict

def scannGoogle(dataset_name, num_leaves_= 2000, num_leaves_to_search_=100, training_sample_size_=250000,num_neighbors_=2,anisotropic_quantization_threshold_=0.2):
    info={}
    neighbors_r=None
    try:
        info['dataset_name']=dataset_name
        info['num_leaves_']=num_leaves_
        info['num_leaves_to_search_']=num_leaves_to_search_
        info['training_sample_size_']=training_sample_size_
        info['num_neighbors_']=num_neighbors_
        info['anisotropic_quantization_threshold_']=anisotropic_quantization_threshold_
        xb, xq, gt= get_data_generic(dataset_name)
        d = xb.shape[1]
        k = gt.shape[1]
        distance_measure='squared_l2'
        searcher_without_built = scann.scann_ops_pybind.builder(xb, k, distance_measure).tree(num_leaves=num_leaves_,num_leaves_to_search=num_leaves_to_search_, training_sample_size=training_sample_size_).score_ah(num_neighbors_, anisotropic_quantization_threshold=anisotropic_quantization_threshold_).reorder(k)
        searcher_without_built.training_threads=1
        start = time.time()
        searcher=searcher_without_built.build()
        end = time.time()
        construction_time=end-start
        start = time.time()
        neighbors, distances = searcher.search_batched(xq, leaves_to_search=num_leaves_to_search_,final_num_neighbors=k,pre_reorder_num_neighbors=k+1)
        info['I_1'] = neighbors
        neighbors_r=neighbors
        end = time.time()
        search_time=end-start
        r = calculate_recall_at(gt, neighbors, k, k)

        # we are given top 100 neighbors in the ground truth, so select top 10
        info['recall']=r
        info['search_time']=search_time
        start = time.time()
        neighbors, distances = searcher.search_batched(xq, leaves_to_search=num_leaves_to_search_,final_num_neighbors=k,pre_reorder_num_neighbors=k*2)
        info['I_2'] = neighbors
        end = time.time()
        search_time=end-start
        r = calculate_recall_at(gt, neighbors, k, k)
        info['recallx2']=r
        info['search_timex2']=search_time
        start = time.time()
        neighbors, distances = searcher.search_batched(xq, leaves_to_search=num_leaves_to_search_,final_num_neighbors=k,pre_reorder_num_neighbors=k*3)
        info['I_3'] = neighbors
        end = time.time()
        search_time=end-start
        r = calculate_recall_at(gt, neighbors, k, k)
        info['recallx3']=r
        info['search_timex3']=search_time
        neighbors, distances = searcher.search_batched(xq, leaves_to_search=num_leaves_to_search_,final_num_neighbors=k,pre_reorder_num_neighbors=k*4)
        info['I_4'] = neighbors
        end = time.time()
        search_time=end-start
        r = calculate_recall_at(gt, neighbors, k, k)
        info['recallx4']=r
        info['search_timex4']=search_time
        neighbors, distances = searcher.search_batched(xq, leaves_to_search=num_leaves_to_search_,final_num_neighbors=k,pre_reorder_num_neighbors=k*5)
        info['I_5'] = neighbors
        end = time.time()
        search_time=end-start
        r = calculate_recall_at(gt, neighbors, k, k)
        info['recallx5']=r
        info['search_timex5']=search_time
        info['construction_time']=construction_time
    except:
        info={'error': True}
        print(info)
        return info
    print(info)
    return info
    
def scannTest1(dataset_name_):
    scannGoogle(dataset_name_, num_leaves_=2000, num_leaves_to_search_=10, num_neighbors_=2, anisotropic_quantization_threshold_=0.2)
    scannGoogle(dataset_name_, num_leaves_=2000, num_leaves_to_search_=25, num_neighbors_=2, anisotropic_quantization_threshold_=0.2)
    scannGoogle(dataset_name_, num_leaves_=2000, num_leaves_to_search_=50, num_neighbors_=2, anisotropic_quantization_threshold_=0.2)
    scannGoogle(dataset_name_, num_leaves_=2000, num_leaves_to_search_=75, num_neighbors_=2, anisotropic_quantization_threshold_=0.2)
    scannGoogle(dataset_name_, num_leaves_=2000, num_leaves_to_search_=100, num_neighbors_=2, anisotropic_quantization_threshold_=0.2)
    scannGoogle(dataset_name_, num_leaves_=2000, num_leaves_to_search_=200, num_neighbors_=2, anisotropic_quantization_threshold_=0.2)
    scannGoogle(dataset_name_, num_leaves_=2000, num_leaves_to_search_=400, num_neighbors_=2, anisotropic_quantization_threshold_=0.2)
    scannGoogle(dataset_name_, num_leaves_=2000, num_leaves_to_search_=600, num_neighbors_=2, anisotropic_quantization_threshold_=0.2)
    scannGoogle(dataset_name_, num_leaves_=2000, num_leaves_to_search_=1000, num_neighbors_=2, anisotropic_quantization_threshold_=0.2)
    scannGoogle(dataset_name_, num_leaves_=2000, num_leaves_to_search_=1500, num_neighbors_=2, anisotropic_quantization_threshold_=0.2)

def PQ_CENTERING(data_loader, version, d, m, nbits, dataset_name, k = 20, refine = None):
    xb, xq, gt= data_loader(m)
    k = gt.shape[1]
    return (PQ_PROXY_CENTERING(data_loader, version, d, m, nbits, dataset_name, k, k),
            PQ_PROXY_CENTERING(data_loader, version, d, m, nbits, dataset_name, k, k*2))
    

def PQ_PROXY_CENTERING(data_loader, version, d, m, nbits, dataset_name, k = 20, refine = None):
    xb, xq, gt= data_loader(m)
    xMean=np.average(xb,axis=0)
    xb=xb-xMean
    xq=xq-xMean
    d = xb.shape[1]
    k = gt.shape[1]
    if (xb.shape[1]%m)!=0:
        subSpaceSize = xb.shape[1]/m
        subSpaceSize = math.ceil(subSpaceSize)
        d = subSpaceSize * m
        xb = extend_time_series_with_zeros(xb,subSpaceSize*m)
        xq = extend_time_series_with_zeros(xq,subSpaceSize*m)
    if refine == None:
        refine = k
    index = None
    
    if version == 'PQ':
        index = faiss.IndexPQ(d, m, nbits)
    else:
        s  = "OPQ"+str(int(m))+",PQ"+str(int(m))+"x"+str(nbits)
        # index = faiss.index_factory(d, "OPQ16,IVF128,PQ16x8")
        # print(s)
        index = faiss.index_factory(d, s)
    index.is_trained
    train_start_time = time.time();
    index.train(xb)
    train_end_time = time.time();
    index.add(xb)
    encoding_end_time = time.time()
    search_start_time = time.time();
    D_pq, I_pq = index.search(xq, refine)
    I_pq = refine_response(xq, xb, I_pq, k)
    r = calculate_recall_at(gt, I_pq, k, k)
    search_end_time = time.time();
    # print(r)
    dict = {}
    dict['recall@'] = r
    dict['map@'] = calculate_map_at(gt, I_pq, k)
    dict['recall'] = calculate_recall(gt, I_pq, k)
    dict['model_name'] = version+'_centering'
    dict['dataset_name'] = dataset_name
    dict['training-time'] =  train_end_time - train_start_time
    dict['encoding-time'] =  encoding_end_time - train_end_time
    dict['construction-time'] =  encoding_end_time - train_start_time
    dict['search-time'] =  search_end_time - search_start_time
    dict['d'] = d
    dict['m'] = m
    dict['nbits'] = nbits
    dict['k'] = k
    dict['refine'] = refine
    return dict

def PQ_EIGEN(data_loader, version, d, m, nbits, dataset_name, k = 20, refine = None):
    xb, xq, gt= data_loader(m)
    k = gt.shape[1]
    return (PQ_PROXY_EIGEN(data_loader, version, d, m, nbits, dataset_name, k, k),
            PQ_PROXY_EIGEN(data_loader, version, d, m, nbits, dataset_name, k, k*2))
    

def PQ_PROXY_EIGEN(data_loader, version, d, m, nbits, dataset_name, k = 20, refine = None):
    xb, xq, gt= data_loader(m)
    Y=np.dot(np.transpose(xb),xb)
    w, v = np.linalg.eig(Y)
    xb=np.dot(xb,v)
    xq=np.dot(xq,v)
    d = xb.shape[1]
    k = gt.shape[1]
    if (xb.shape[1]%m)!=0:
        subSpaceSize = xb.shape[1]/m
        subSpaceSize = math.ceil(subSpaceSize)
        d = subSpaceSize * m
        xb = extend_time_series_with_zeros(xb,subSpaceSize*m)
        xq = extend_time_series_with_zeros(xq,subSpaceSize*m)
    if refine == None:
        refine = k
    index = None
    
    if version == 'PQ':
        index = faiss.IndexPQ(d, m, nbits)
    else:
        s  = "OPQ"+str(int(m))+",PQ"+str(int(m))+"x"+str(nbits)
        # index = faiss.index_factory(d, "OPQ16,IVF128,PQ16x8")
        # print(s)
        index = faiss.index_factory(d, s)
    index.is_trained
    train_start_time = time.time();
    index.train(xb)
    train_end_time = time.time();
    index.add(xb)
    encoding_end_time = time.time()
    search_start_time = time.time();
    D_pq, I_pq = index.search(xq, refine)
    I_pq = refine_response(xq, xb, I_pq, k)
    r = calculate_recall_at(gt, I_pq, k, k)
    search_end_time = time.time();
    # print(r)
    dict = {}
    dict['recall@'] = r
    dict['map@'] = calculate_map_at(gt, I_pq, k)
    dict['model_name'] = version+'-EIGEN'
    dict['dataset_name'] = dataset_name
    dict['training-time'] =  train_end_time - train_start_time
    dict['encoding-time'] =  encoding_end_time - train_end_time
    dict['construction-time'] =  encoding_end_time - train_start_time
    dict['search-time'] =  search_end_time - search_start_time
    dict['d'] = d
    dict['m'] = m
    dict['nbits'] = nbits
    dict['k'] = k
    dict['refine'] = refine
    return dict

def AQ(data_loader, version, d, m, nbits, dataset_name, k = 20, refine = None):
    xb, xq, gt= data_loader(m)
    k = gt.shape[1]
    return (PQ_PROXY(data_loader, version, d, m, nbits, dataset_name, k, k),
            PQ_PROXY(data_loader, version, d, m, nbits, dataset_name, k, k*2))
    

def AQ_PROXY(data_loader, version, d, m, nbits, dataset_name, k = 20, refine = None):
    xb, xq, gt= data_loader(m)
    d = xb.shape[1]
    k = gt.shape[1]
    if (xb.shape[1]%m)!=0:
        subSpaceSize = xb.shape[1]/m
        subSpaceSize = math.ceil(subSpaceSize)
        d = subSpaceSize * m
        xb = extend_time_series_with_zeros(xb,subSpaceSize*m)
        xq = extend_time_series_with_zeros(xq,subSpaceSize*m)
    if refine == None:
        refine = k
    index = None
    index = faiss.IndexAdditiveQuantizer(d, m, nbits)
    index.is_trained
    train_start_time = time.time();
    index.train(xb)
    train_end_time = time.time();
    index.add(xb)
    encoding_end_time = time.time()
    search_start_time = time.time();
    D_pq, I_pq = index.search(xq, refine)
    I_pq = refine_response(xq, xb, I_pq, k)
    r = calculate_recall_at(gt, I_pq, k, k)
    search_end_time = time.time();
    # print(r)
    dict = {}
    dict['recall@'] = r
    dict['map@'] = calculate_map_at(gt, I_pq, k)
    dict['model_name'] = "AQ"
    dict['dataset_name'] = dataset_name
    dict['training-time'] =  train_end_time - train_start_time
    dict['encoding-time'] =  encoding_end_time - train_end_time
    dict['construction-time'] =  encoding_end_time - train_start_time
    dict['search-time'] =  search_end_time - search_start_time
    dict['d'] = d
    dict['m'] = m
    dict['nbits'] = nbits
    dict['k'] = k
    dict['refine'] = refine
    return dict

def IMI_PQ(data_loader, version, d, m, nbits, dataset_name, imi_string = 'IMI2x1', k = 20, refine = None):
    xb, xq, gt= data_loader(m)
    d = xb.shape[1]
    k = gt.shape[1]
    if (xb.shape[1]%m)!=0:
        subSpaceSize = xb.shape[1]/m
        subSpaceSize = math.ceil(subSpaceSize)
        d = subSpaceSize * m
        xb = extend_time_series_with_zeros(xb,subSpaceSize*m)
        xq = extend_time_series_with_zeros(xq,subSpaceSize*m)
    if refine == None:
        refine = k
    index = None
    
    if version == 'PQ':
        s  = imi_string + ",PQ"+str(int(m))+"x"+str(nbits)
        # index = faiss.index_factory(d, "OPQ16,IVF128,PQ16x8")
        # print(s)
        index = faiss.index_factory(d, s)
    else:
        s  = "OPQ"+str(int(m))+ ',' + imi_string + ",PQ"+str(int(m))+"x"+str(nbits)
        # index = faiss.index_factory(d, "OPQ16,IVF128,PQ16x8")
        # print(s)
        index = faiss.index_factory(d, s)
    index.is_trained
    train_start_time = time.time();
    index.train(xb)
    train_end_time = time.time();
    index.add(xb)
    encoding_end_time = time.time()
    search_start_time = time.time();
    D_pq, I_pq = index.search(xq, refine)
    I_pq = refine_response(xq, xb, I_pq, k)
    r = calculate_recall_at(gt, I_pq, k, k)
    search_end_time = time.time();
    # print(r)
    dict = {}
    dict['recall@'] = r
    m1 = calculate_map_at(gt, I_pq, k)
    dict['map@'] = m1
    dict['recall'] = calculate_recall(gt, I_pq, k)
    dict['model_name'] = version + ','+ imi_string
    dict['dataset_name'] = dataset_name
    dict['training-time'] =  train_end_time - train_start_time
    dict['encoding-time'] =  encoding_end_time - train_end_time
    dict['construction-time'] =  encoding_end_time - train_start_time
    dict['search-time'] =  search_end_time - search_start_time
    dict['d'] = d
    dict['m'] = m
    dict['nbits'] = nbits
    dict['k'] = k
    dict['refine'] = refine
    # print('Reach here')
    return dict

def LSH(data_loader, d, nbits, dataset_name, k):
    xb, xq, gt= data_loader()
    k = gt.shape[1]
    d = xb.shape[1]
    index = faiss.IndexLSH(d, nbits)
    train_end_time = time.time();
    index.add(xb)
    encoding_end_time = time.time()
    search_start_time = time.time()
    D_pq, I_pq = index.search(xq, k)
    r = calculate_recall_at(gt, I_pq, k, k)
    search_end_time = time.time();
    # print(r)
    dict = {}
    dict['recall@'] = r
    dict['map@'] = calculate_map_at(gt, I_pq, k)
    dict['model_name'] = 'LSH'
    dict['dataset_name'] = dataset_name
    dict['encoding-time'] =  encoding_end_time - train_end_time
    dict['construction-time'] =  encoding_end_time - train_end_time
    dict['search-time'] =  search_end_time - search_start_time
    dict['d'] = d
    dict['nbits'] = nbits
    dict['k'] = k
    return dict


def PQFS(data_loader, version, d, m, nbits, dataset_name, k = 20, refine = None):
    xb, xq, gt= data_loader(m)
    k = gt.shape[1]
    return (PQFS_PROXY(data_loader, version, d, m, nbits, dataset_name, k, k),
            PQFS_PROXY(data_loader, version, d, m, nbits, dataset_name, k, k*2))

def PQFS_PROXY(data_loader, version, d, m, nbits, dataset_name, k = 20, refine = None):
    nbits = 4
    xb, xq, gt= data_loader(m)
    d = xb.shape[1]
    k = gt.shape[1]
    if (xb.shape[1]%m)!=0:
        subSpaceSize = xb.shape[1]/m
        subSpaceSize = math.ceil(subSpaceSize)
        d = subSpaceSize * m
        xb = extend_time_series_with_zeros(xb,subSpaceSize*m)
        xq = extend_time_series_with_zeros(xq,subSpaceSize*m)
    if refine == None:
        refine = k
    index = faiss.IndexPQFastScan(d, m, nbits)
    index.is_trained
    train_start_time = time.time();
    index.train(xb)
    train_end_time = time.time();
    index.add(xb)
    encoding_end_time = time.time()
    search_start_time = time.time();
    D_pq, I_pq = index.search(xq, refine)
    I_pq = refine_response(xq, xb, I_pq, k)
    r = calculate_recall_at(gt, I_pq, k, k)
    search_end_time = time.time();
    # print(r)
    dict = {}
    dict['recall@'] = r
    dict['map@'] = calculate_map_at(gt, I_pq, k)
    dict['model_name'] = 'PQFS'
    dict['dataset_name'] = dataset_name
    dict['training-time'] =  train_end_time - train_start_time
    dict['encoding-time'] =  encoding_end_time - train_end_time
    dict['construction-time'] =  encoding_end_time - train_start_time
    dict['search-time'] =  search_end_time - search_start_time
    dict['d'] = d
    dict['m'] = m
    dict['nbits'] = nbits
    dict['k'] = k
    dict['refine'] = refine
    return dict


def IVFPQ(dataset_name, d, m, nbits, k = 20, refine = None):
    xb, xq, gt= get_data_common(dataset_name)
    k = gt.shape[1]
    return (IVFPQ_PROXY(dataset_name, d, m, nbits, dataset_name, k, k),
            IVFPQ_PROXY(dataset_name, d, m, nbits, dataset_name, k, k*2))

def IVFPQ_PROXY(dataset_name, d, m, nbits, k = 20, refine = None):
    xb, xq, gt= get_data_common(dataset_name)
    d = xb.shape[1]
    k = gt.shape[1]
    if (xb.shape[1]%m)!=0:
        subSpaceSize = xb.shape[1]/m
        subSpaceSize = math.ceil(subSpaceSize)
        d = subSpaceSize * m
        xb = extend_time_series_with_zeros(xb,subSpaceSize*m)
        xq = extend_time_series_with_zeros(xq,subSpaceSize*m)
    if refine == None:
        refine = k
    nbits = min(8,nbits)
    vecs = faiss.IndexFlatL2(d)
    
    nlist = int(math.sqrt(xb.shape[0]))
    index = faiss.IndexIVFPQ(vecs, d, nlist, m, nbits)
    index.is_trained
    train_start_time = time.time();
    index.train(xb)
    train_end_time = time.time();
    index.add(xb)
    encoding_end_time = time.time()
    dicts=[]
    for multiplier in range(1,6):
        refine=k*multiplier
        search_start_time = time.time()
        index.nprobe = int(0.05*nlist)
        D_pq, I_pq = index.search(xq, refine)
        I_pq = refine_response(xq, xb, I_pq, k)
        r = calculate_recall_at(gt, I_pq, k, k)
        search_end_time = time.time();
        # print(r)
        dict = {}
        dict['recall@'+'x'+str(refine)] = r
        dict['map@'] = calculate_map_at(gt, I_pq, k)
        dict['model_name'] = 'IVFPQ'
        dict['dataset_name'] = dataset_name
        dict['training-time'] =  train_end_time - train_start_time
        dict['encoding-time'] =  encoding_end_time - train_end_time
        dict['construction-time'] =  encoding_end_time - train_start_time
        dict['search-time'+'x'+str(refine)] =  search_end_time - search_start_time
        dict['d'] = d
        dict['m'] = m
        dict['nbits'] = nbits
        dict['k'] = k
        dict['refine'] = refine
        dict['I_pq'] = I_pq
        dict['D_pq'] = D_pq
        dict['I_pq'] = I_pq
        dict['D_pq'] = D_pq
    print(dict)    
    return dict


# def diskANN(data_loader, dataset_name, caseId, _graph_max_degree = 64, _window_size = 128, _search_window_size = 30):
#     xb,xq,gt = data_loader()
#     d = xb.shape[1]
#     k = gt.shape[1]
#     rel_dir = index_path + "index-"+dataset_name+'-'+str(caseId)
#     Path(rel_dir).mkdir(exist_ok=True)
    
#     print("\n\n====================== +BUILD")
#     dap.build_disk_index(
#         data=xb,
#         distance_metric="l2",
#         index_directory=rel_dir,
#         graph_degree=_graph_max_degree,
#         complexity=_window_size,
#         vector_dtype=np.float32,
#         search_memory_maximum=0.05,
#         build_memory_maximum=0.05,
#         num_threads=1,
#         pq_disk_bytes=0
#     )
    
#     print("\n\n====================== +LOAD")

#     index = dap.StaticDiskIndex(
#             distance_metric="l2",
#             vector_dtype=np.float32,
#             index_directory=Path(rel_dir).resolve(),
#             num_threads=1,
#             num_nodes_to_cache=1
#         )

#     print("\n\n====================== +SEARCH")
#     ids, dists = index.batch_search(
#                     xq, k_neighbors=k, complexity=_window_size, beam_width=_search_window_size, num_threads=1)
#     print(ids.shape)

def vamana_ovq(data_loader, dataset_name, compressed_loader = None,_graph_max_degree = 64, _window_size = 128, _search_window_size = 30, k = 10):
    parameters = ps.VamanaBuildParameters(
        graph_max_degree = _graph_max_degree,
        window_size = _window_size,
    )
    xb,xq,gt = data_loader()
    # print(xb.shape, xq.shape, gt.shape)
    construction_start_time = time.time();
    if compressed_loader:
        xb = compressed_loader
    index = ps.Vamana.build(
        parameters,
        xb,
        ps.DistanceType.L2,
        num_threads = 1,
    )
    construction_end_time = time.time();
    index.search_window_size = _search_window_size
    
    search_start_time = time.time();
    I, D = index.search(xq, k)
    search_end_time = time.time();
    # Compare with the groundtruth.
    r = calculate_recall_at(gt, I, k, k)
    dict = {}
    dict['recall@'] = r
    dict['map@'] = calculate_map_at(gt, I, k)
    dict['construction-time'] =  construction_end_time - construction_start_time
    dict['search-time'] =  search_end_time - search_start_time
    dict['dataset_name'] = dataset_name
    dict['_graph_max_degree'] = _graph_max_degree
    dict['_window_size'] = _window_size
    dict['_search_window_size'] = _search_window_size
    dict['k'] = k
    dict['index_size']=asizeof.asizeof(index)
    version = 'vamana'
    if compressed_loader:
        version = version + '_LVQ'
    dict['model_name'] = version;
    return dict

def ovq_proxy(data_loader, dataset_name, compressed_loader = None,_graph_max_degree = 64, _window_size = 128, _search_window_size = 30, k = 10):
    try:
        xb,xq,gt = data_loader()
        k = gt.shape[1]
        ret = vamana_ovq(data_loader, dataset_name, compressed_loader,_graph_max_degree, _window_size, _search_window_size, k)
        log_file_name = "logVAMANA.txt"
        sys.stdout = open("logs/" + log_file_name, "a")
        print(ret)
        ret = vamana_ovq(data_loader, dataset_name, None,_graph_max_degree, _window_size, _search_window_size, k)
        sys.stdout = open("logs/" + log_file_name, "a")
        print(ret)
    except:
        print("Error occurred on ", dataset_name)

def ovq_svs_lvq(data_loader, dataset_name, compressed_loader = None,_graph_max_degree = 64, _window_size = 128, _search_window_size = 30, k = 10):
    try:
        xb,xq,gt = data_loader()
        k = gt.shape[1]
        ret = vamana_ovq(data_loader, dataset_name, get_compressed_loader(data_src_lvq_train[dataset_name], xb.shape[1], data_loader=data_loader),_graph_max_degree, _window_size, _search_window_size, k)
        log_file_name = "logVAMANALVQmem.txt"
        sys.stdout = open("logs/" + log_file_name, "a")
        print(ret)
    except:
        print("Error occurred on ", dataset_name)


def ovq_svs(data_loader, dataset_name, compressed_loader = None,_graph_max_degree = 64, _window_size = 128, _search_window_size = 30, k = 10):
    try:
        xb,xq,gt = data_loader()
        k = gt.shape[1]
        log_file_name = "logVAMANAmem.txt"
        ret = vamana_ovq(data_loader, dataset_name, None,_graph_max_degree, _window_size, _search_window_size, k)
        sys.stdout = open("logs/" + log_file_name, "a")
        print(ret)
    except:
        print("Error occurred on ", dataset_name)

class Accuracy:
    LOWEST = 'lowest'
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    
_acc_to_nbytes = {
    Accuracy.LOWEST: 2,
    Accuracy.LOW: 8,
    Accuracy.MEDIUM: 16,
    Accuracy.HIGH: 32,
}
    
# def bolt(data_loader, dataset_name, bit=128, k=20, accuracy='medium', refine = None):
#     xb,xq,gt = data_loader()
#     k = gt.shape[1]
#     d = xb.shape[1]
#     if refine == None:
#         refine = k
#     construction_start_time = time.time();
#     enc = b.Encoder('l2', accuracy=accuracy).fit(xb)
#     construction_end_time = time.time();
    
#     search_start_time = time.time()
#     bolt_knn = [enc.knn(q, refine) for q in xq]
#     search_end_time = time.time();
#     I = np.array(bolt_knn)
#     I = refine_response(xq, xb, I, k)
#     r = calculate_recall_at(gt, I, k, k)
#     dict = {}
#     dict['recall@'] = r
#     dict['map@'] = calculate_map_at(gt, I, k)
#     dict['construction-time'] =  construction_end_time - construction_start_time
#     dict['search-time'] =  search_end_time - search_start_time
#     dict['dataset_name'] = dataset_name
#     dict['k'] = k
#     dict['B'] = accuracy
#     dict['model_name'] = 'bolt';
#     return dict




def annoySearch(dataset_name, tree = 10, k=20):
    xb,xq,gt = get_data_generic(dataset_name)
    k = gt.shape[1]
    I = np.zeros((xq.shape[0],k))
    index = AnnoyIndex(xq.shape[1], 'euclidean')
    construction_start_time = time.time()
    for i in range(0,xb.shape[0]):
        index.add_item(i,xb[i])
    index.build(tree, n_jobs=1)
    construction_end_time = time.time();
    search_start_time = time.time()
    for i in range(0,xq.shape[0]):
        I[i]=np.array(index.get_nns_by_vector(xq[i], k))
    search_end_time = time.time();
    I = np.array(I)
    r = calculate_recall_at(gt, I, k, k)
    dict = {}
    dict['recall'] = r
    dict['map@'] = calculate_map_at(gt, I, k)
    dict['construction-time'] =  construction_end_time - construction_start_time
    dict['search-time'] =  search_end_time - search_start_time
    dict['dataset_name'] = dataset_name
    dict['k'] = k
    dict['tree'] = tree
    dict['model_name'] = 'annoy'
    dict['I']=I
    test_id='annoy-'+dataset_name+'-'+str(tree)+'.pkl'
    search_start_time = time.time()
    I1 = np.zeros((xq.shape[0],k*2))
    for i in range(0,xq.shape[0]):
        I1[i]=np.array(index.get_nns_by_vector(xq[i], k*2))
    I1=refine_response(xq,xb,I1,k)    
    search_end_time = time.time();
    r1 = calculate_recall_at(gt, I1, k, k)
    dict['recallx2'] = r1
    dict['search-timex2'] =  search_end_time - search_start_time
    
    
    search_end_time = time.time();
    with open(pickle_path+test_id, 'wb') as f:
        pickle.dump(dict, f)
    print(dict)
    return dict

def OPQ(data_loader, d, m, nbits, dataset_name, k = 20, refine = None, iter=50):
    xb, xq, gt= data_loader(m)
    d = xb.shape[1]
    k = gt.shape[1]
    if refine == None:
        refine = k
    if (xb.shape[1]%m)!=0:
        subSpaceSize = xb.shape[1]/m
        subSpaceSize = math.ceil(subSpaceSize)
        d = subSpaceSize * m
        xb = extend_time_series_with_zeros(xb,subSpaceSize*m)
        xq = extend_time_series_with_zeros(xq,subSpaceSize*m)
    index_pq = faiss.IndexPQ(d, m, nbits)
    opq_matrix = faiss.OPQMatrix(d, m)
    # opq_matrix.verbose = true
    opq_matrix.niter = iter
    opq_matrix.niter_pq = 4
    index = faiss.IndexPreTransform(opq_matrix, index_pq)
    index.is_trained
    train_start_time = time.time();
    index.train(xb)
    train_end_time = time.time();
    index.add(xb)
    encoding_end_time = time.time()
    search_start_time = time.time();
    D_pq, I_pq = index.search(xq, refine)
    I_pq = refine_response(xq, xb, I_pq, k)
    r = calculate_recall_at(gt, I_pq, k, k)
    search_end_time = time.time();
    # print(r)
    dict = {}
    dict['recall@'] = r
    dict['map@'] = calculate_map_at(gt, I_pq, k)
    dict['model_name'] = 'OPQv1'
    dict['dataset_name'] = dataset_name
    dict['training-time'] =  train_end_time - train_start_time
    dict['encoding-time'] =  encoding_end_time - train_end_time
    dict['construction-time'] =  encoding_end_time - train_start_time
    dict['search-time'] =  search_end_time - search_start_time
    dict['d'] = d
    dict['m'] = m
    dict['nbits'] = nbits
    dict['k'] = k
    dict['refine'] = refine
    return dict

def OPQv1(data_loader, d, m, nbits, dataset_name, k = 20, refine = None, iter=50):
    xb, xq, gt= data_loader(m)
    d = xb.shape[1]
    k = gt.shape[1]
    if refine == None:
        refine = k
    if (xb.shape[1]%m)!=0:
        subSpaceSize = xb.shape[1]/m
        subSpaceSize = math.ceil(subSpaceSize)
        d = subSpaceSize * m
        xb = extend_time_series_with_zeros(xb,subSpaceSize*m)
        xq = extend_time_series_with_zeros(xq,subSpaceSize*m)
    index = faiss.IndexPQ(d, m, nbits)
    opq_matrix = faiss.OPQMatrix(d, m)
    # opq_matrix.verbose = true
    opq_matrix.niter = iter
    opq_matrix.train(xb)
    xb=opq_matrix.apply_py(xb)
    xq=opq_matrix.apply_py(xq)
    train_start_time = time.time();
    index.train(xb)
    train_end_time = time.time();
    index.add(xb)
    encoding_end_time = time.time()
    search_start_time = time.time();
    D_pq, I_pq = index.search(xq, refine)
    I_pq = refine_response(xq, xb, I_pq, k)
    r = calculate_recall_at(gt, I_pq, k, k)
    search_end_time = time.time();
    # print(r)
    dict = {}
    dict['recall@'] = r
    dict['map@'] = calculate_map_at(gt, I_pq, k)
    dict['model_name'] = 'OPQv2'
    dict['dataset_name'] = dataset_name
    dict['training-time'] =  train_end_time - train_start_time
    dict['encoding-time'] =  encoding_end_time - train_end_time
    dict['construction-time'] =  encoding_end_time - train_start_time
    dict['search-time'] =  search_end_time - search_start_time
    dict['d'] = d
    dict['m'] = m
    dict['nbits'] = nbits
    dict['k'] = k
    dict['refine'] = refine
    return dict

# def OPQv2(data_loader, d, m, nbits, dataset_name, k = 20, refine = None, iter=50):
#     xb, xq, gt= data_loader(m)
#     d = xb.shape[1]
#     k = gt.shape[1]
#     if refine == None:
#         refine = k
#     if (xb.shape[1]%m)!=0:
#         subSpaceSize = xb.shape[1]/m
#         subSpaceSize = math.ceil(subSpaceSize)
#         d = subSpaceSize * m
#         xb = extend_time_series_with_zeros(xb,subSpaceSize*m)
#         xq = extend_time_series_with_zeros(xq,subSpaceSize*m)
#     index = nanopq.OPQ(M=m)
#     index.verbose = False
#     # opq_matrix.verbose = true
    
#     train_start_time = time.time();
#     index.fit(xb, rotation_iter=50)
#     train_end_time = time.time();
#     X_code = index.encode(xb)
#     I_pq = np.zeros((xq.shape[0],k), dtype=np.int32)
#     encoding_end_time = time.time()
#     search_start_time = time.time();
#     for i in range(0,xq.shape[0]):
#         query=xq[i]
#         dists = index.dtable(query).adist(X_code)  # (10000, ) 
#         # print(dists)
#         sorted_index_array = np.argsort(dists)
#         I_pq[i]=np.array(sorted_index_array[:k])
#         # min_n = np.argmin(dists)
#         # print(sorted_index_array[:k])
#         # print(min_n)
#     # print(I_pq.dtype)
#     # print(I_pq)
#     I_pq = refine_response(xq, xb, I_pq, k)
#     r = calculate_recall_at(gt, I_pq, k, k)
#     search_end_time = time.time();
    
#     dict = {}
#     dict['recall@'] = r
#     dict['map@'] = calculate_map_at(gt, I_pq, k)
#     dict['model_name'] = 'OPQv2'
#     dict['dataset_name'] = dataset_name
#     dict['training-time'] =  train_end_time - train_start_time
#     dict['encoding-time'] =  encoding_end_time - train_end_time
#     dict['construction-time'] =  encoding_end_time - train_start_time
#     dict['search-time'] =  search_end_time - search_start_time
#     dict['d'] = d
#     dict['m'] = m
#     dict['nbits'] = nbits
#     dict['k'] = k
#     dict['refine'] = refine
#     return dict

# def PQv2(data_loader, dataset_name, d, m, nbits, k = 20, refine = None):
#     xb, xq, gt= data_loader(m)
#     d = xb.shape[1]
#     k = gt.shape[1]
#     if refine == None:
#         refine = k
#     if (xb.shape[1]%m)!=0:
#         subSpaceSize = xb.shape[1]/m
#         subSpaceSize = math.ceil(subSpaceSize)
#         d = subSpaceSize * m
#         xb = extend_time_series_with_zeros(xb,subSpaceSize*m)
#         xq = extend_time_series_with_zeros(xq,subSpaceSize*m)
#     xb=np.float32(xb)
#     xq=np.float32(xq)
#     index = nanopq.PQ(M=m, Ks=2**nbits, verbose=True)
#     index.verbose = False
#     # opq_matrix.verbose = true
    
#     train_start_time = time.time();
#     index.fit(vecs=xb, iter=25, seed=123)
#     train_end_time = time.time();
#     X_code = index.encode(xb)
#     I_pq = np.zeros((xq.shape[0],k), dtype=np.int32)
#     encoding_end_time = time.time()
#     search_start_time = time.time();
#     for i in range(0,xq.shape[0]):
#         query=xq[i]
#         dists = index.dtable(query).adist(X_code)  # (10000, ) 
#         # print(dists)
#         sorted_index_array = np.argsort(dists)
#         I_pq[i]=np.array(sorted_index_array[:k])
#         # min_n = np.argmin(dists)
#         # print(sorted_index_array[:k])
#         # print(min_n)
#     # print(I_pq.dtype)
#     # print(I_pq)
#     I_pq = refine_response(xq, xb, I_pq, k)
#     r = calculate_recall_at(gt, I_pq, k, k)
#     search_end_time = time.time();
    
#     dict = {}
#     dict['recall@'] = r
#     dict['map@'] = calculate_map_at(gt, I_pq, k)
#     dict['model_name'] = 'PQv2'
#     dict['dataset_name'] = dataset_name
#     dict['training-time'] =  train_end_time - train_start_time
#     dict['encoding-time'] =  encoding_end_time - train_end_time
#     dict['construction-time'] =  encoding_end_time - train_start_time
#     dict['search-time'] =  search_end_time - search_start_time
#     dict['d'] = d
#     dict['m'] = m
#     dict['nbits'] = nbits
#     dict['k'] = k
#     dict['refine'] = refine
#     return dict

def linearScan(data_loader, name):
    base, query, gt= data_loader()
    d = base.shape[1]
    K = gt.shape[1]
    
    
    ground_trouth = []
    ground_trouth_dist = []
    print(base.shape, query.shape)
    search_start_time = time.time()
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
        res_dist = []
        for node in heapq.nlargest(K, heap, key=None):
            res.append(node[1])
            res_dist.append(-node[0])
        ground_trouth.append(res)
        ground_trouth_dist.append(res_dist)
    search_end_time = time.time();
    ground_trouth=np.array(ground_trouth, dtype=np.uint32)
    ground_trouth_dist=np.array(ground_trouth_dist)
    r = calculate_recall_at(gt, ground_trouth, K, K)
    dict = {}
    dict['search-time'] =  search_end_time - search_start_time
    dict['recall@'] = r
    dict['recall@dist'] = calculate_recall_dist(gt, ground_trouth_dist, K)
    dict['name'] = name
    return dict

def linearScanPCA(data_loader, name):
    base, query, gt= data_loader()
    d = base.shape[1]
    K = gt.shape[1]
    Y=np.dot(np.transpose(base),base)
    w, v = np.linalg.eig(Y)
    base=np.dot(base,v)
    query=np.dot(query,v)
    check=np.dot(np.transpose(v),v)
    print('Check eigen vector')
    print(check)
    
    ground_trouth = []
    print(base.shape, query.shape)
    search_start_time = time.time()
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
    search_end_time = time.time();
    ground_trouth=np.array(ground_trouth, dtype=np.uint32)
    r = calculate_recall_at(gt, ground_trouth, K, K)
    dict = {}
    dict['search-time'] =  search_end_time - search_start_time
    dict['recall@'] = r
    dict['name'] = name
    return dict

def linearScanPCA2(data_loader, name):
    base, query, gt= data_loader()
    d = base.shape[1]
    K = gt.shape[1]
    Y=np.dot(np.transpose(base),base)
    w, v = np.linalg.eig(Y)
    dim=base.shape[1]
    sort_order=np.argsort(-w)
    rotation_matrix=np.zeros((dim,dim))
    for i in range(0,len(sort_order)):
        rotation_matrix[i][sort_order[i]]=1
    
    v=np.dot(v,rotation_matrix)
    print('rotation_matrix')
    print(rotation_matrix)
    base=np.dot(base,v)
    query=np.dot(query,v)
    check=np.dot(np.transpose(v),v)
    print('Check eigen vector')
    print(check)
    
    ground_trouth = []
    print(base.shape, query.shape)
    search_start_time = time.time()
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
    search_end_time = time.time();
    ground_trouth=np.array(ground_trouth, dtype=np.uint32)
    r = calculate_recall_at(gt, ground_trouth, K, K)
    dict = {}
    dict['search-time'] =  search_end_time - search_start_time
    dict['recall@'] = r
    dict['name'] = name
    return dict

def disPlayGraph(y1,y2,name):
    fig = plt.figure() 
    # data to be plotted
    x = np.arange(1, len(y2)+1)
    
    # plotting
    plt.title("Variance per dim ("+name+")") 
    plt.xlabel("dim") 
    plt.ylabel("variance") 
    plt.plot(x, y1, color ="green")
    plt.plot(x, y2, color ="blue") 
    plt.savefig("variance-graph/variance"+name+".png")

def analyzeVariance(data_loader, name):
    base, query, gt= data_loader()
    dict = {}
    dict['name']=name
    dict['var-raw']=np.var(base, axis = 0)
    dict['var-raw-sum']=np.sum(dict['var-raw'])
    Y=np.dot(np.transpose(base),base)
    w, v = np.linalg.eig(Y)
    base=np.dot(base,v)
    query=np.dot(query,v)
    check=np.dot(np.transpose(v),v)
    dict['var-pca']=np.var(base, axis = 0)
    dict['var-pca-sum']=np.sum(dict['var-pca'])
    dict['eigen']=w
    disPlayGraph(dict['var-raw'],dict['var-pca'],name)
    return dict

# d = 128
# nbits = 8
# dataset= get_data_audio
# name='audio'
# k=20

# print(OPQv2(dataset, d, 32, nbits, name, k = k))   
# print(OPQv2(dataset, d, 16, nbits, name, k = k)) 
# def bartllet_test(data_loader, name):
#     info={}
#     base, query, gt= data_loader()
#     data=base
#     # Perform Bartlett's test across the features (columns)
#     # Bartlett's test expects multiple arrays (one for each feature)
#     stat, p_value = levene(*[data[:, i] for i in range(data.shape[1])])

#     print(f"Bartlett's test statistic: {stat}")
#     print(f"P-value: {p_value}")
#     info['dataset']=name
#     info['stat']=stat
#     info['p_value']=p_value
#     # Interpretation
#     alpha = 0.05  # Significance level
#     if p_value < alpha:
#         print("Reject the null hypothesis: The variances are significantly different across the features.")
#     else:
#         print("Fail to reject the null hypothesis: The variances are equal across the features, suitable for PCA.")
#     return info

