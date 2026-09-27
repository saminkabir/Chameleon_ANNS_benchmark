dataset_infos=['lendb', 'text-to-image', 'llama-128-ip', 'deep', 'simplewiki-openai-3072-normalized', 'yi-128-ip', 'PNW', 'notre', 'landmark-nomic-768-normalized', 'Music', 'audio', 'instancegm', 'yandex-200-cosine', 'OBS', 'random', 'netflix', 'laion-clip-512-normalized', 'imageNet', 'NEIC', 'imagenet-clip-512-normalized', 'gooaq-distilroberta-768-normalized', 'landmark-dino-768-cosine', 'ukbench', 'nuswide', 'gist', 'word2vec', 'coco-nomic-768-normalized', 'space1V', 'celeba-resnet-2048-cosine', 'sun', 'yahoomusic', 'bigann', 'cifar', 'nytimes', 'movielens', 'MNIST', 'OBST2024', 'sift', 'uqv', 'txed', 'seismic1m', 'imagenet-align-640-normalized', 'geofon', 'vcseis', 'sald1m', 'millionSong', 'Meier2019JGR', 'ISC_EHB_DepthPhases', 'arxiv-nomic-768-normalized', 'agnews-mxbai-1024-euclidean', 'yahoo-minilm-384-normalized', 'glove', 'stead', 'ethz', 'Iquique']
excludes=[]
tmp='/usr/bin/time -v python ann_search_flatnav.py --dataset [dataset] --max_edges_per_node [max_edges_per_node] --ef_construction [ef_construction]  --ef_search [ef_search]  &> flatnavogs/[dataset]-[max_edges_per_node]-[ef_construction]-[ef_search].txt'

for th in [(16,8,16),(24,12,24),(32,16,32),(24,24,24),(32,32,32),(48,24,48),(64,32,64),(48,48,48),(64,64,64),(96,48,96),(96,96,96)]:
    for dataset_info in dataset_infos:
        dataset=dataset_info
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
