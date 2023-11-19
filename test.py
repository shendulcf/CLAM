import os
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm

slide_dir = '/home/sci/Disk_data/Datasets/TCGA-NSCLC-sub/FEATURES_level1/pt_files'
num_clusters = 10

for slide_file in tqdm(os.listdir(slide_dir)):

    features = torch.load(os.path.join(slide_dir, slide_file))
    
    kmeans = KMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(features)
    # 转换为张量
    labels = torch.from_numpy(labels)
    # 获取每个类别的patch数量
    label_counts = torch.bincount(labels)
    size = (num_clusters, 1, 1024)
    clustered_feats = torch.zeros(*size).repeat(1, label_counts.size(0), 1)

    for i, label in enumerate(labels):
        if i < clustered_feats.shape[1]:
            clustered_feats[label, i, :] = features[i]

    slide_name = slide_file.split('.')[0]
    torch.save(clustered_feats, f'./dataset_cluster/TCGA/pt_files/{slide_name}.pt')