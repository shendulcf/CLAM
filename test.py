import os
from sklearn.cluster import KMeans, DBSCAN
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np

# 假设 features 是你的特征数据（每行代表一个 patch 的特征向量）
path = r'/home/sci/Disk_data/Datasets/TCGA-NSCLC-sub/FEATURES_level1/pt_files/'

# 选择聚类数量
k = 50
# DBSCAN 参数
eps = 0.5
min_samples = 5

# 保存所有聚类的列表
clustered_features_list = []

for slide in tqdm(os.listdir(path)):
    slide_feat = torch.load(os.path.join(path, slide))
    slide_name = os.path.splitext(slide)[0]
    
    # 使用 K 均值聚类
    # kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(slide_feat)

    # # 获取聚类中心的特征向量
    # cluster_centers = kmeans.cluster_centers_

    # 将每个聚类的特征组织成一个新的数据结构
    clustered_features = []
    for cluster_id in np.unique(cluster_labels):
        if cluster_id == -1:
            continue
        # 获取属于当前聚类的索引
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        # 获取属于当前聚类的特征
        cluster_features = slide_feat[cluster_indices, :]
        # 添加到新的数据结构中
        clustered_features.append([c.clone().detach() for c in cluster_features])

    # 将聚类堆叠在一起
    clustered_features_tensor = pad_sequence([torch.stack(c) for c in clustered_features], batch_first=True)
    print(clustered_features_tensor.shape)
    torch.save(clustered_features_tensor, f'./dataset_cluster/TCGA/pt_files/{slide_name}.pt')

    # 添加到整体列表中
    #clustered_features_list.append(clustered_features_tensor)

    # 将整体列表堆叠在一起
    #all_clustered_features = pad_sequence(clustered_features_list, batch_first=True)

    # 在这里，可以添加代码来删除填充的部分，例如通过计算非零元素的数量，截取相应的长度
    #if not os.path.exists(f'./dataset_cluster/TCGA/pt_files/{slide_name}.pt'):
    # 保存 PyTorch 张量为文件
        #torch.save(all_clustered_features, f'./dataset_cluster/TCGA/pt_files/{slide_name}.pt')