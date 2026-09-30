
import sys
dataset_infos=[]

dataset_infoss=['dpr']

for datasest in dataset_infoss:
    dataset_infos.append(datasest)

tmp='/usr/bin/time -v taskset --cpu-list 0-127 python ann_search_hnsw.py --dataset [dataset] --m [m] --ef_search [ef_search]  --ef_construction  [ef_construction] &> logHNSW/[dataset]-128-[m]-[ef_search]-[ef_construction].txt'
cnt=-1
for th in [(16,16,16),(16,8,16),(24,12,24),(32,16,32),(48,24,48),(64,32,64),(96,96,96)]:
    for dataset_info in dataset_infos:
        dataset=dataset_info
        text=tmp+''
        text=text.replace('[dataset]',dataset)
        text=text.replace('[m]',str(th[0]))
        text=text.replace('[ef_search]',str(th[1]))
        text=text.replace('[ef_construction]',str(th[2]))
        print(text,end=' ; ')
print('echo 1')
