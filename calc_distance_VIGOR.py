import numpy as np
import torch
from sklearn.metrics import DistanceMetric
import pickle
# from vigor import VigorDatasetTrain
from datasets_with_sampling import VIGORDataset
from util import geodistance

TOP_K = 256
BATCH_SIZE = 10
dataset_root = '/data/dataset/VIGOR'
# -----------------------------------------------------------------------------#
# Cross Area                                                                   #
# -----------------------------------------------------------------------------#

def calc_cross_area():
    dataset = VIGORDataset(dataset_root, split="crossarea", train=True, pos_only=True,
                         transform=None, ori_noise=0,
                         random_orientation=None)

    # df_sat = dataset.df_sat
    df_ground = dataset.df_ground
    print(f"length of df_ground : {len(df_ground)}")

    idx2sat = dataset.idx2sat
    train_sat_ids = np.sort(df_ground["sat"].unique()) # one dim

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
        # print(f"main sample: {idx2sat[idx]}")
        for id in ids_near[i]:
            _, lat, long = idx2sat[train_sat_ids[id]][:-4].split('_')
            # print(f"lat_main: {lat_main}, long_main: {long_main}, lat: {lat}, long: {long}")
            dis = geodistance(lat_main, long_main, lat, long)
            if dis > 100 and count < BATCH_SIZE:
                add = True
                for j in selected_ids:
                    _, lat1, long1 = idx2sat[train_sat_ids[j]][:-4].split('_')
                    dis1 = geodistance(lat, long, lat1, long1)
                    if dis1 < 100:
                        add = False
                        break
                if add:
                    selected_ids.append(id)
                    count += 1
                if count == BATCH_SIZE:
                    break

        near_neighbors[idx] = train_sat_ids[selected_ids].tolist()
        # print(f"lat_main: {lat_main}, long_main: {long_main}")
        # for i in near_neighbors[idx]:
        #     _, lat, long = idx2sat[i][:-4].split('_')
        #     # print(f"lat: {lat}, long: {long}")


    # for i, idx in enumerate(train_sat_ids):
    #     near_neighbors[idx] = train_sat_ids[ids_near[i]].tolist()

    print("Saving...")
    with open("dataset/vigor_gps_dict_cross_debug4.pkl", "wb") as f:
        pickle.dump(near_neighbors, f)

# -----------------------------------------------------------------------------#
# Same Area                                                                  #
# -----------------------------------------------------------------------------#

def calc_same_area():

    dataset = VIGORDataset(dataset_root, split="samearea", train=True, pos_only=True,
                         transform=None, ori_noise=0,
                         random_orientation=None)


    # df_sat = dataset.df_sat
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
            if dis > 100 and count < BATCH_SIZE:
                add = True
                for j in selected_ids:
                    _, lat1, long1 = idx2sat[train_sat_ids[j]][:-4].split('_')
                    dis1 = geodistance(lat, long, lat1, long1)
                    if dis1 < 100:
                        add = False
                        break
                if add:
                    selected_ids.append(id)
                    count += 1
                if count == BATCH_SIZE:
                    break

        near_neighbors[idx] = train_sat_ids[selected_ids].tolist()
        # print(f"lat_main: {lat_main}, long_main: {long_main}")


    print("Saving...")
    with open("dataset/vigor_gps_dict_same_debug4.pkl", "wb") as f:
        pickle.dump(near_neighbors, f)

if __name__ == '__main__':
    calc_same_area()