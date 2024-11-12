import numpy as np
import torch
from sklearn.metrics import DistanceMetric
import pickle
from vigor import VigorDatasetTrain
from util import geodistance

TOP_K = 64
BATCH_SIZE = 20

# -----------------------------------------------------------------------------#
# Cross Area                                                                   #
# -----------------------------------------------------------------------------#

SAME_AREA = False

dataset = VigorDatasetTrain(data_folder="/data/dataset/VIGOR",
                            same_area=SAME_AREA)

df_sat = dataset.df_sat
df_ground = dataset.df_ground
print(f"length of df_ground : {len(df_ground)}")

idx2sat = dataset.idx2sat
train_sat_ids = np.sort(df_ground["sat"].unique())

print("Length Train Ids Cross Area:", len(train_sat_ids))

gps_coords = {}
gps_coords_list = []

for idx in train_sat_ids:
    _, lat, long = idx2sat[idx][:-4].split('_')

    gps_coords[idx] = (float(lat), float(long))
    gps_coords_list.append((float(lat), float(long)))

print("Length of gps coords : " + str(len(gps_coords_list)))
print("Calculation...")

dist = DistanceMetric.get_metric('haversine')
dm = dist.pairwise(gps_coords_list, gps_coords_list)
print("Distance Matrix:", dm.shape)

dm_torch = torch.from_numpy(dm)
dm_torch = dm_torch.fill_diagonal_(dm.max())

values, ids = torch.topk(dm_torch, k=TOP_K, dim=1, largest=False)

values_near = values.numpy()
ids_near = ids.numpy()

near_neighbors = dict()

for i, idx in enumerate(train_sat_ids):
    selected_ids = []
    _, lat_main, long_main = idx2sat[idx][:-4].split('_')
    count = 0
    for id in ids_near[i]:
        _, lat, long = idx2sat[train_sat_ids[id]][:-4].split('_')
        # print(f"lat_main: {lat_main}, long_main: {long_main}, lat: {lat}, long: {long}")
        dis = geodistance(lat_main, long_main, lat, long)
        if dis > 100 and count <= BATCH_SIZE:
            # print(f"add, dis = {dis}")
            selected_ids.append(id)
            count += 1
            if count == BATCH_SIZE:
                break
        # else:
        #     print("not add")
    near_neighbors[idx] = train_sat_ids[selected_ids].tolist()
    # print(f"lat_main: {lat_main}, long_main: {long_main}")
    # for i in near_neighbors[idx]:
    #     _, lat, long = idx2sat[i][:-4].split('_')
    #     print(f"lat: {lat}, long: {long}")

print("Saving...")
with open("dataset/vigor_gps_dict_cross.pkl", "wb") as f:
    pickle.dump(near_neighbors, f)

# -----------------------------------------------------------------------------#
# Same Area                                                                  #
# -----------------------------------------------------------------------------#

SAME_AREA = True

dataset = VigorDatasetTrain(data_folder="/data/dataset/VIGOR",
                            same_area=SAME_AREA)

df_sat = dataset.df_sat
df_ground = dataset.df_ground
print(f"length of df_ground : {len(df_ground)}")

idx2sat = dataset.idx2sat
train_sat_ids = np.sort(df_ground["sat"].unique())

print("Length Train Ids Same Area:", len(train_sat_ids))

gps_coords = {}
gps_coords_list = []

for idx in train_sat_ids:
    _, lat, long = idx2sat[idx][:-4].split('_')

    gps_coords[idx] = (float(lat), float(long))
    gps_coords_list.append((float(lat), float(long)))

print("Length of gps coords : " + str(len(gps_coords_list)))
print("Calculation...")

dist = DistanceMetric.get_metric('haversine')
dm = dist.pairwise(gps_coords_list, gps_coords_list)
print("Distance Matrix:", dm.shape)

dm_torch = torch.from_numpy(dm)
dm_torch = dm_torch.fill_diagonal_(dm.max())

values, ids = torch.topk(dm_torch, k=TOP_K, dim=1, largest=False)

values_near = values.numpy()
ids_near = ids.numpy()

near_neighbors = dict()

# for i, idx in enumerate(train_sat_ids):
#     near_neighbors[idx] = train_sat_ids[ids_near[i]].tolist()

for i, idx in enumerate(train_sat_ids):
    selected_ids = []
    _, lat_main, long_main = idx2sat[idx][:-4].split('_')
    count = 0
    for id in ids_near[i]:
        _, lat, long = idx2sat[train_sat_ids[id]][:-4].split('_')
        # print(f"lat_main: {lat_main}, long_main: {long_main}, lat: {lat}, long: {long}")
        dis = geodistance(lat_main, long_main, lat, long)
        if dis > 100 and count <= BATCH_SIZE:
            # print(f"add, dis = {dis}")
            selected_ids.append(id)
            count += 1
            if count == BATCH_SIZE:
                break
        # else:
        #     print("not add")
    near_neighbors[idx] = train_sat_ids[selected_ids].tolist()
    # print(f"lat_main: {lat_main}, long_main: {long_main}")

print("Saving...")
with open("dataset/vigor_gps_dict_same.pkl", "wb") as f:
    pickle.dump(near_neighbors, f)
