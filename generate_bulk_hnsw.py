
import sys
dataset_infos=[]

dataset_infoss=['deep', 'glove', 'sun', 'audio', 'millionSong', 'nuswide', 'MNIST', 'notre', 'sift', 'imageNet', 'llama-128-ip_20_variants', 'coco-nomic-768-normalized_20_variants', 'yi-128-ip_20_variants', 'agnews-mxbai-1024-euclidean_20_variants', 'gooaq-distilroberta-768-normalized_20_variants', 'ccnews-nomic-768-normalized_20_variants', 'yandex-200-cosine_20_variants', 'arxiv-nomic-768-normalized_20_variants', 'laion-clip-512-normalized_20_variants','yahoomusic_20_variants']
dataset_infoss=[
    'notre'
]
sizes = ['_rc_1.25','_rc_1.5','_rc_2.0']
sizes = ['_rc_2.0']

def get_cpu(cpu):
    cpu_infos=[(255,0)]
    cpu=cpu+1
    for cpu_info in cpu_infos:
        if cpu_info[0]==cpu:
            # print(cpu)
            return cpu_info[1]
    return cpu

for datasest in dataset_infoss:
    dataset_infos.append(datasest)
excludes=[]

tmp='/usr/bin/time -v taskset --cpu-list [cpu]-[cpu] python ann_search_hnsw.py --dataset [dataset] --m [m] --ef_search [ef_search]  --ef_construction  [ef_construction] &> logHNSW/[dataset]-[m]-[ef_search]-[ef_construction].txt'
cnt=-1
for size in sizes:
    for th in [(16,16,16),(16,8,16),(24,12,24),(32,16,32),(48,24,48),(64,32,64),(96,96,96)]:
        for dataset_info in dataset_infos:
            dataset=dataset_info+size
            if not dataset in excludes:
                text=tmp+''
                cnt=get_cpu(cnt)
                text=text.replace('[dataset]',dataset)
                text=text.replace('[m]',str(th[0]))
                text=text.replace('[ef_search]',str(th[1]))
                text=text.replace('[ef_construction]',str(th[2]))
                text=text.replace('[cpu]',str(cnt))
                
                print(text,end=' & ')
print('echo 1')
