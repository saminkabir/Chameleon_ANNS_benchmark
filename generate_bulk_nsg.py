import sys
dataset_infos=[
    'dpr'
]

tmp='/usr/bin/time -v taskset --cpu-list 0-127 python ann_search_nsg.py --dataset [dataset] --m [m] --cpu 128> logNSGBigDataset/[dataset]-[m].txt'
for th in [12,24,48,64]:
    for dataset_info in dataset_infos:
        dataset=dataset_info+''
        text=tmp+''
        text=text.replace('[dataset]',dataset)
        text=text.replace('[m]',str(th))
        print(text,end=' ; ')
print('echo 1')
