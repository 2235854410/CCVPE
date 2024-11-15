import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets_with_sampling import VIGORDataset
from torchvision import transforms

from util import geodistance, print_colored

# from datasets import VIGORDataset

# 创建 DataLoader（假设 batch size 为 4，可以根据需求调整）
batch_size = 4
dataset_root = '/data/dataset/VIGOR'
transform_grd = transforms.Compose([
    transforms.Resize([320, 640]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])

transform_sat = transforms.Compose([
    # resize
    transforms.Resize([512, 512]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])
def unnormalize(img, mean, std):
    img = img.clone()  # 防止直接修改原数据
    for i in range(3):  # 对每个通道逆标准化
        img[i] = img[i] * std[i] + mean[i]
    return img
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
area = "samearea"
if area == 'crossarea':
    GPS_DICT_PATH = "/home/test/code/CCVPE/dataset/vigor_gps_dict_cross_debug4.pkl"
else:
    GPS_DICT_PATH = "/home/test/code/CCVPE/dataset/vigor_gps_dict_same_debug4.pkl"


with open(GPS_DICT_PATH, "rb") as f:
    sim_dict = pickle.load(f)

vigor = VIGORDataset(dataset_root, split=area, train=True, pos_only=True,
                     transform=(transform_grd, transform_sat), ori_noise=0,
                     random_orientation=None, test=True)

dataset_length = int(vigor.__len__())
index_list = np.arange(vigor.__len__())
train_indices = index_list[0: int(len(index_list) * 0.8)]
val_indices = index_list[int(len(index_list) * 0.8):]
training_set = Subset(vigor, train_indices)
val_set = Subset(vigor, val_indices)
train_dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=False, num_workers=12)
val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=12)

vigor.shuffle(sim_dict)
# 取出一个 batch
data_iter = iter(train_dataloader)

for j in range(10):

    grd, sat, gt, gt_with_ori, gt_orientation, city, _, delta0, delta1, sat_path, grd_path = next(data_iter)
    # grd, sat, gt, gt_with_ori, gt_orientation, city, _, sat_path, grd_path  = next(data_iter)
    # 可视化 batch 数据
    fig, axes = plt.subplots(batch_size, 2, figsize=(10, 5 * batch_size))
    coordinates = []

    for i in range(batch_size):
        # 卫星图
        ax_sat = axes[i, 0]
        sat_img = unnormalize(sat[i], mean, std).permute(1, 2, 0).numpy()  # 将卫星图还原并转为 numpy 格式

        coord_distribution = gt[i].squeeze().numpy()  # 得到 [512, 512] 的高斯分布
        y_coord, x_coord = divmod(coord_distribution.argmax(), coord_distribution.shape[1])  # 最大值位置

        _, lat, long = sat_path[i][:-4].split('_')
        coordinates.append([lat, long])

        print(sat_path[i])
        print(grd_path[i])
        print(f"lat: {lat}, long: {long}")
        print(f"delta y: {delta0[i]}, x :{delta1[i]}")
        print(f"y_coord: {y_coord}, x_coord: {x_coord}")

        # 显示卫星图并叠加高斯平滑的坐标分布
        ax_sat.imshow(sat_img)
        ax_sat.scatter(x_coord, y_coord, color='red', marker='x', label="Ground Truth Point")
        ax_sat.legend()
        ax_sat.set_title("Satellite Image with Ground Truth Point")

        # 地面图
        ax_ground = axes[i, 1]
        ground_img = unnormalize(grd[i], mean, std).permute(1, 2, 0).numpy()  # 将地面图还原并转为 numpy 格式
        ax_ground.imshow(ground_img)
        ax_ground.set_title("Ground Image")
    distances = []
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            dist_ij = geodistance(coordinates[i][0], coordinates[i][1], coordinates[j][0], coordinates[j][1])
            distances.append(dist_ij)

    # 输出结果
    print_colored(distances)
    plt.tight_layout()
    plt.show()