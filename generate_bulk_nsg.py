import sys
dataset_infos=[
    'notre'
]
sizes = ['_rc_1.25','_rc_1.5','_rc_2.0']
sizes = ['_rc_2.0']

def get_cpu(cpu):
    cpu_infos=[(255,1)]
    cpu=cpu+1
    for cpu_info in cpu_infos:
        if cpu_info[0]==cpu:
            # print(cpu)
            return cpu_info[1]
    return cpu



excludes=['crawl','gist','tiny5m','enron','trevi']
tmp='/usr/bin/time -v taskset --cpu-list [cpu]-[cpu] python ann_search_nsg.py --dataset [dataset] --m [m] &> logNSG/[dataset]-[m].txt'
cnt=10
for size in sizes:
    for th in [12,24,48,64]:
        for dataset_info in dataset_infos:
            dataset=dataset_info+size
            if not dataset in excludes:
                cnt=get_cpu(cnt)
                text=tmp+''
                text=text.replace('[dataset]',dataset)
                text=text.replace('[m]',str(th))
                text=text.replace('[cpu]',str(cnt))
                print(text,end=' & ')
print('echo 1')
