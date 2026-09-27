import sys
dataset_infos=['OBST2024', 'space1V', 'siftsmall', 'imagenet-align-640-normalized', 'seismic1m', 'gooaq-distilroberta-768-normalized', 'astro1m', 'glove-200-cosine', 'agnews-mxbai-1024-euclidean', 'ccnews-nomic-768-normalized', 'sald1m']
excludes=[]
cnt=21
import os
tmp='/usr/bin/time -v python ann_search_flatnav.py --dataset [dataset] --max_edges_per_node [max_edges_per_node] --ef_construction [ef_construction]  --ef_search [ef_search]  &> flatnavogs/[dataset]-[max_edges_per_node]-[ef_construction]-[ef_search].txt'
sizes = ['_rc_1.25','_rc_1.5','_rc_2.0']
sizes = ['_rc_2.0']
sizes = ['']

for size in sizes:
    for th in [(16,8,16),(24,12,24),(32,16,32),(24,24,24),(32,32,32),(48,24,48),(64,32,64),(48,48,48),(64,64,64),(96,48,96),(96,96,96)]:
        for dataset_info in dataset_infos:
            dataset=dataset_info+size
            if not dataset in excludes:
                text=tmp+''
                text=text.replace('[dataset]',dataset)
                text=text.replace('[max_edges_per_node]',str(th[0]))
                text=text.replace('[ef_search]',str(th[1]))
                text=text.replace('[ef_construction]',str(th[2]))
                fileName='[dataset]-[max_edges_per_node]-[ef_construction]-[ef_search].txt'
                fileName=fileName.replace('[dataset]',dataset)
                fileName=fileName.replace('[max_edges_per_node]',str(th[0]))
                fileName=fileName.replace('[ef_search]',str(th[1]))
                fileName=fileName.replace('[ef_construction]',str(th[2]))
                print(text,end=' & ')
         
print('echo 1')
