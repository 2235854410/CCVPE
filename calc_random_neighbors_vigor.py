import random

import numpy as np
import torch
from sklearn.metrics import DistanceMetric
import pickle
# from vigor import VigorDatasetTrain
from datasets_with_sampling import VIGORDataset
from util import geodistance

TOP_K = 128
BATCH_SIZE = 10

# -----------------------------------------------------------------------------#
# Cross Area                                                                   #
# -----------------------------------------------------------------------------#

SAME_AREA = False

batch_size = 4
dataset_root = '/data/dataset/VIGOR'

dataset = VIGORDataset(dataset_root, split="crossarea", train=True, pos_only=True,
                     transform=None, ori_noise=0,
                     random_orientation=None)

# df_sat = dataset.df_sat
df_ground = dataset.df_ground
print(f"length of df_ground : {len(df_ground)}")

idx2sat = dataset.idx2sat
train_sat_ids = df_ground["sat"].unique()
np.random.shuffle(train_sat_ids)
random.shuffle(train_sat_ids)

print("Length Train Ids Cross Area:", len(train_sat_ids))

gps_coords = {}
gps_coords_list = []

for idx in train_sat_ids:
    _, lat, long = idx2sat[idx][:-4].split('_')

    gps_coords[idx] = (float(lat), float(long))
    gps_coords_list.append((float(lat), float(long)))

print("Length of gps coords : " + str(len(gps_coords_list)))
print("Calculation...")

# 计算距离矩阵
dist = DistanceMetric.get_metric('haversine')
dm = dist.pairwise(gps_coords_list)
print("Distance Matrix:", dm.shape)

dm_torch = torch.from_numpy(dm)

# 遍历每个样本，找到距离大于100米的邻居
near_neighbors = dict()

for i, idx in enumerate(train_sat_ids):
    selected_ids = []
    _, lat_main, long_main = idx2sat[idx][:-4].split('_')

    # 遍历所有其他样本
    for j, other_idx in enumerate(train_sat_ids):
        if i == j:
            continue  # 跳过自己

        _, lat, long = idx2sat[other_idx][:-4].split('_')
        dis = geodistance(lat_main, long_main, lat, long)

        # 判断是否大于100米
        if dis > 100:
            selected_ids.append(j)

        if len(selected_ids) > BATCH_SIZE:
            selected_ids = random.sample(selected_ids, BATCH_SIZE)
            break



    near_neighbors[idx] = [train_sat_ids[i] for i in selected_ids]

print("Saving...")
with open("dataset/vigor_random_dict_cross_debug.pkl", "wb") as f:
    pickle.dump(near_neighbors, f)
