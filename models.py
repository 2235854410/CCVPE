from torchvision import transforms
import torchvision.models as models
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.utils import make_grid
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import ImageFile
from torch.utils.tensorboard import SummaryWriter
import math

from util import print_colored

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.manual_seed(17)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"The device is: {}".format(device)
from efficientnet_pytorch.model import EfficientNet

class permute_channels(nn.Module):
    def __init__(self, B, C, H, W):
        super(permute_channels, self).__init__()
        self.B = B
        self.C = C
        self.H = H
        self.W = W
        
    def forward(self, x):
        return torch.permute(x, (self.B, self.C, self.H, self.W))

class normalization(nn.Module):
    def __init__(self, p, dim):
        super(normalization, self).__init__()
        self.p = p
        self.dim = dim
        
    def forward(self, x):
        return F.normalize(x, p=self.p, dim=self.dim)

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    )

class CVM_VIGOR(nn.Module):
    def __init__(self, device, circular_padding):
        super(CVM_VIGOR, self).__init__()
        self.device = device
        self.circular_padding = circular_padding
        
        self.grd_efficientnet = EfficientNet.from_pretrained('efficientnet-b0', self.circular_padding)

        self.grd_feature_to_descriptor1 = nn.Sequential(
                                    nn.Conv2d(1280, 64, 1),
                                    permute_channels(0, 2, 3, 1),
                                    nn.Conv2d(10, 1, 1),
                                    nn.Flatten(start_dim=1)
                                    )
        
        self.grd_feature_to_descriptor2 = nn.Sequential(
                                    nn.Conv2d(1280, 32, 1),
                                    permute_channels(0, 2, 3, 1),
                                    nn.Conv2d(10, 1, 1),
                                    nn.Flatten(start_dim=1)
                                    )
        
        self.grd_feature_to_descriptor3 = nn.Sequential(
                                    nn.Conv2d(1280, 16, 1),
                                    permute_channels(0, 2, 3, 1),
                                    nn.Conv2d(10, 1, 1),
                                    nn.Flatten(start_dim=1)
                                    )
        
        self.grd_feature_to_descriptor4 = nn.Sequential(
                                    nn.Conv2d(1280, 8, 1),
                                    permute_channels(0, 2, 3, 1),
                                    nn.Conv2d(10, 1, 1),
                                    nn.Flatten(start_dim=1)
                                    )
        
        self.grd_feature_to_descriptor5 = nn.Sequential(
                                    nn.Conv2d(1280, 4, 1),
                                    permute_channels(0, 2, 3, 1),
                                    nn.Conv2d(10, 1, 1),
                                    nn.Flatten(start_dim=1)
                                    )
        
        self.grd_feature_to_descriptor6 = nn.Sequential(
                                    nn.Conv2d(1280, 2, 1),
                                    permute_channels(0, 2, 3, 1),
                                    nn.Conv2d(10, 1, 1),
                                    nn.Flatten(start_dim=1)
                                    )
        
        self.sat_efficientnet = EfficientNet.from_pretrained('efficientnet-b0', circular=False)
        
        
        self.sat_feature_to_descriptors = nn.Sequential(nn.Flatten(start_dim=1),
                                                        nn.Linear(1280*2*2, 1280)
                                                       )
        
        self.sat_normalization = normalization(2, 1)
        
        # loc
        self.deconv6 = nn.ConvTranspose2d(1281, 1024, 2, 2)
        self.conv6 = double_conv(1344, 640)
                                    
        self.deconv5 = nn.ConvTranspose2d(641, 320, 2, 2)
        self.conv5 = double_conv(432, 320)
        
        self.deconv4 = nn.ConvTranspose2d(321, 160, 2, 2)
        self.conv4 = double_conv(200, 160)
        
        self.deconv3 = nn.ConvTranspose2d(161, 80, 2, 2)
        self.conv3 = double_conv(104, 80)
        
        self.deconv2 = nn.ConvTranspose2d(81, 40, 2, 2)
        self.conv2 = double_conv(56, 40)
        
        self.deconv1 = nn.ConvTranspose2d(41, 16, 2, 2)
        self.conv1 = nn.Sequential(nn.Conv2d(16, 16, 3, stride=1, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(16, 1, 3, stride=1, padding=1))
        
        # ori
        self.deconv6_ori = nn.ConvTranspose2d(1300, 1024, 2, 2)
        self.conv6_ori = double_conv(1344, 640)
                                    
        self.deconv5_ori = nn.ConvTranspose2d(640, 256, 2, 2)
        self.conv5_ori = double_conv(368, 256)
        
        self.deconv4_ori = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv4_ori = double_conv(168, 128)
        
        self.deconv3_ori = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv3_ori = double_conv(88, 64)
        
        self.deconv2_ori = nn.ConvTranspose2d(64, 32, 2, 2)
        self.conv2_ori = double_conv(48, 32)
        
        self.deconv1_ori = nn.ConvTranspose2d(32, 16, 2, 2)
        self.conv1_ori = nn.Sequential(nn.Conv2d(16, 16, 3, stride=1, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(16, 2, 3, stride=1, padding=1))
        
    def forward(self, grd, sat):
        grd_feature_volume = self.grd_efficientnet.extract_features(grd) # shape=[8, 1280, 10, 20] 1280 是64*20
        grd_descriptor1 = self.grd_feature_to_descriptor1(grd_feature_volume) # length 1280 shape=[8, 1280]
        grd_descriptor2 = self.grd_feature_to_descriptor2(grd_feature_volume) # length 640 shape=[8, 640]
        grd_descriptor3 = self.grd_feature_to_descriptor3(grd_feature_volume) # length 320
        grd_descriptor4 = self.grd_feature_to_descriptor4(grd_feature_volume) # length 160
        grd_descriptor5 = self.grd_feature_to_descriptor5(grd_feature_volume) # length 80
        grd_descriptor6 = self.grd_feature_to_descriptor6(grd_feature_volume) # length 40
        
        grd_descriptor_map1 = grd_descriptor1.unsqueeze(2).unsqueeze(3).repeat(1, 1, 8, 8)
        grd_descriptor_map2 = grd_descriptor2.unsqueeze(2).unsqueeze(3).repeat(1, 1, 16, 16)
        grd_descriptor_map3 = grd_descriptor3.unsqueeze(2).unsqueeze(3).repeat(1, 1, 32, 32)
        grd_descriptor_map4 = grd_descriptor4.unsqueeze(2).unsqueeze(3).repeat(1, 1, 64, 64)
        grd_descriptor_map5 = grd_descriptor5.unsqueeze(2).unsqueeze(3).repeat(1, 1, 128, 128)
        grd_descriptor_map6 = grd_descriptor6.unsqueeze(2).unsqueeze(3).repeat(1, 1, 256, 256)
      
        sat_feature_volume, multiscale_sat = self.sat_efficientnet.extract_features_multiscale(sat)
        # len(multiscale_sat)=16, sat_feature_volume.shape = [8, 1280, 16, 16]
        sat_feature_block0 = multiscale_sat[0] # [16, 256, 256]
        sat_feature_block2 = multiscale_sat[2] #[24, 128, 128]
        sat_feature_block4 = multiscale_sat[4] # [40, 64, 64]
        sat_feature_block10 = multiscale_sat[10] # [112, 32, 32]
        sat_feature_block15 = multiscale_sat[15] # [320, 16, 16]

        sat_row_chunks = torch.stack(list(torch.chunk(sat_feature_volume, 8, 2)),
                                     dim=-1)  # dimension 4 is the number of row chunks (splitted in height dimension)
        for i, sat_row_chunk in enumerate(torch.unbind(sat_row_chunks, dim=-1), 0):
            sat_chunks = torch.stack(list(torch.chunk(sat_row_chunk, 8, 3)),
                                     dim=-1)  # dimension 5 is the number of vertical chunks (splitted in width dimension)
            for j, sat_chunk in enumerate(torch.unbind(sat_chunks, dim=-1), 0):
                # sat_chunk.shape=[8, 1280, 2, 2]
                if j == 0:
                    # sat_descriptor_row.shape=[8, 1280, 1, 1]
                    sat_descriptor_row = self.sat_feature_to_descriptors(sat_chunk).unsqueeze(2).unsqueeze(3)
                else:
                    sat_descriptor_row = torch.cat(
                        (sat_descriptor_row, self.sat_feature_to_descriptors(sat_chunk).unsqueeze(2).unsqueeze(3)), 3)
            if i == 0:
                sat_descriptor_map = sat_descriptor_row
            else:
                sat_descriptor_map = torch.cat((sat_descriptor_map, sat_descriptor_row), 2)

        # matching bottleneck
        # sat_descriptor_map.shape=[8, 1280, 8, 8]
        grd_des_len = grd_descriptor1.size()[1]
        sat_des_len = sat_descriptor_map.size()[1]
        # grd_map_norm.shape=[8, 1, 8, 8]
        grd_map_norm = torch.norm(grd_descriptor_map1, p='fro', dim=1, keepdim=True)
        sat_map_norm = torch.norm(sat_descriptor_map, p='fro', dim=1, keepdim=True)
        if torch.isnan(sat_descriptor_map).any():
            print_colored("NaN detected in sat_descriptor_map!")
            print(f"sat_descriptor_map: {sat_descriptor_map}")

        B, C, H, W = grd_descriptor_map1.shape
        score_maps1 = torch.zeros(B, B, 1, H, W).cuda()
        for i in range(B):
            for j in range(B):
                # 计算第 i 个 grd 和第 j 个 sat 描述符的相似度图，得到 score_map[i, j]
                matching = torch.sum(grd_descriptor_map1[i] * sat_descriptor_map[j], dim=0)
                norm = sat_map_norm[i] * grd_map_norm[j]
                if (norm == 0).all():
                    print_colored("norm is zero!")
                    print(f"grd_feature_volume: {grd_feature_volume}")
                    print(f"sat_feature_volume: {sat_feature_volume}")
                # matching shape=[8, 8], matching2.shape=[1, 8, 8]
                score_maps1[i, j, 0] = matching / (norm + 1e-8)

        # for i in range(20):
        #     sat_descriptor_map_rolled = torch.roll(sat_descriptor_map, shifts=-i*64, dims=1)
        #     sat_descriptor_map_window = sat_descriptor_map_rolled[:,:grd_des_len, :, :]
        #     sat_map_norm = torch.norm(sat_descriptor_map_window, p='fro', dim=1, keepdim=True)
        #     # grd_descriptor_map1.shape=[8, 1280, 8, 8], sat_descriptor_map_window.shape=[8, 1280, 8, 8]
        #     matching_score = torch.sum((grd_descriptor_map1*sat_descriptor_map_window), dim=1, keepdim=True) / (sat_map_norm * grd_map_norm) # cosine similarity
        #     if i == 0:
        #         matching_score_stacked = matching_score
        #     else:
        #         matching_score_stacked = torch.cat([matching_score_stacked, matching_score], dim=1)

        # matching_score_stacked.shape=[8, 20, 8, 8]
        # matching_score_max.shape = [8, 1, 8, 8]
        # matching_score_max, _ = torch.max(matching_score_stacked, dim=1, keepdim=True)
        # matching_score_max2 = score_maps1[torch.arange(B), torch.arange(B)] #[8, 1, 8, 8]

        # loc
        x = torch.cat([score_maps1[torch.arange(B), torch.arange(B)], self.sat_normalization(sat_descriptor_map)],
                      dim=1)  # [8, 1281, 8,8]

        x = self.deconv6(x)  # [8, 1024, 16, 16]
        x = torch.cat([x, sat_feature_block15], dim=1)
        x = self.conv6(x)  # [8, 640, 16, 16]

        # matching 16*16
        grd_des_len = grd_descriptor2.size()[1]  # 640
        sat_des_len = x.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map2, p='fro', dim=1, keepdim=True)
        sat_map_norm = torch.norm(x, p='fro', dim=1, keepdim=True)
        if torch.isnan(x).any():
            print_colored("NaN detected in sat_descriptor_map2!")
            print(f"sat_descriptor_map2: {x}")

        B, C, H, W = grd_descriptor_map2.shape
        score_maps2 = torch.zeros(B, B, 1, H, W).cuda()
        for i in range(B):
            for j in range(B):
                # 计算第 i 个 grd 和第 j 个 sat 描述符的相似度图，得到 score_map[i, j]
                norm = sat_map_norm[i] * grd_map_norm[j]
                if (norm == 0).all():
                    print_colored("norm is zero!")
                score_maps2[i, j] = torch.sum(grd_descriptor_map2[i] * x[j], dim=0) / (norm + 1e-8)

        # for i in range(20):
        #     sat_descriptor_map_rolled = torch.roll(x, shifts=-i*32, dims=1)
        #     sat_descriptor_map_window = sat_descriptor_map_rolled[:,:grd_des_len, :, :]
        #     sat_map_norm = torch.norm(sat_descriptor_map_window, p='fro', dim=1, keepdim=True)
        #
        #     matching_score = torch.sum((grd_descriptor_map2*sat_descriptor_map_window), dim=1, keepdim=True) / (sat_map_norm * grd_map_norm) # cosine similarity
        #     if i == 0:
        #         matching_score_stacked2 = matching_score
        #     else:
        #         matching_score_stacked2 = torch.cat([matching_score_stacked2, matching_score], dim=1)
        # matching_score_max, _ = torch.max(matching_score_stacked2, dim=1, keepdim=True)
        #
        x = torch.cat([score_maps2[torch.arange(B), torch.arange(B)], self.sat_normalization(x)], dim=1)
        x = self.deconv5(x)
        x = torch.cat([x, sat_feature_block10], dim=1)
        x = self.conv5(x)

        # matching 32*32
        grd_des_len = grd_descriptor3.size()[1]  # 320
        sat_des_len = x.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map3, p='fro', dim=1, keepdim=True)
        sat_map_norm = torch.norm(x, p='fro', dim=1, keepdim=True)
        if torch.isnan(x).any():
            print_colored("NaN detected in sat_descriptor_map3!")
            print(f"sat_descriptor_map3: {x}")

        B, C, H, W = grd_descriptor_map3.shape
        score_maps3 = torch.zeros(B, B, 1, H, W).cuda()
        for i in range(B):
            for j in range(B):
                # 计算第 i 个 grd 和第 j 个 sat 描述符的相似度图，得到 score_map[i, j]
                norm = sat_map_norm[i] * grd_map_norm[j]
                if (norm == 0).all():
                    print_colored("norm is zero!")
                score_maps3[i, j] = torch.sum(grd_descriptor_map3[i] * x[j], dim=0) / (norm + 1e-8)

        # for i in range(20):
        #     sat_descriptor_map_rolled = torch.roll(x, shifts=-i*16, dims=1)
        #     sat_descriptor_map_window = sat_descriptor_map_rolled[:,:grd_des_len, :, :]
        #     sat_map_norm = torch.norm(sat_descriptor_map_window, p='fro', dim=1, keepdim=True)
        #
        #     matching_score = torch.sum((grd_descriptor_map3*sat_descriptor_map_window), dim=1, keepdim=True) / (sat_map_norm * grd_map_norm) # cosine similarity
        #     if i == 0:
        #         matching_score_stacked3 = matching_score
        #     else:
        #         matching_score_stacked3 = torch.cat([matching_score_stacked3, matching_score], dim=1)
        # matching_score_max, _ = torch.max(matching_score_stacked3, dim=1, keepdim=True)
        #
        x = torch.cat([score_maps3[torch.arange(B), torch.arange(B)], self.sat_normalization(x)], dim=1)
        x = self.deconv4(x)
        x = torch.cat([x, sat_feature_block4], dim=1)
        x = self.conv4(x)

        # matching 64*64
        grd_des_len = grd_descriptor4.size()[1]  # 160
        sat_des_len = x.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map4, p='fro', dim=1, keepdim=True)
        sat_map_norm = torch.norm(x, p='fro', dim=1, keepdim=True)
        if torch.isnan(x).any():
            print_colored("NaN detected in sat_descriptor_map4!")
            print(f"sat_descriptor_map4: {x}")

        B, C, H, W = grd_descriptor_map4.shape
        score_maps4 = torch.zeros(B, B, 1, H, W).cuda()
        for i in range(B):
            for j in range(B):
                norm = sat_map_norm[i] * grd_map_norm[j]
                if (norm == 0).all():
                    print_colored("norm is zero!")
                # 计算第 i 个 grd 和第 j 个 sat 描述符的相似度图，得到 score_map[i, j]
                score_maps4[i, j] = torch.sum(grd_descriptor_map4[i] * x[j], dim=0) / (norm + 1e-8)

        # for i in range(20):
        #     sat_descriptor_map_rolled = torch.roll(x, shifts=-i*8, dims=1)
        #     sat_descriptor_map_window = sat_descriptor_map_rolled[:,:grd_des_len, :, :]
        #     sat_map_norm = torch.norm(sat_descriptor_map_window, p='fro', dim=1, keepdim=True)
        #
        #     matching_score = torch.sum((grd_descriptor_map4*sat_descriptor_map_window), dim=1, keepdim=True) / (sat_map_norm * grd_map_norm) # cosine similarity
        #     if i == 0:
        #         matching_score_stacked4 = matching_score
        #     else:
        #         matching_score_stacked4 = torch.cat([matching_score_stacked4, matching_score], dim=1)
        # matching_score_max, _ = torch.max(matching_score_stacked4, dim=1, keepdim=True)
        #
        x = torch.cat([score_maps4[torch.arange(B), torch.arange(B)], self.sat_normalization(x)], dim=1)
        x = self.deconv3(x)
        x = torch.cat([x, sat_feature_block2], dim=1)
        x = self.conv3(x)

        # matching 128*128
        grd_des_len = grd_descriptor5.size()[1]  # 80
        sat_des_len = x.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map5, p='fro', dim=1, keepdim=True)
        sat_map_norm = torch.norm(x, p='fro', dim=1, keepdim=True)
        if torch.isnan(x).any():
            print_colored("NaN detected in sat_descriptor_map4!")
            print(f"sat_descriptor_map4: {x}")

        B, C, H, W = grd_descriptor_map5.shape
        score_maps5 = torch.zeros(B, B, 1, H, W).cuda()
        for i in range(B):
            for j in range(B):
                # 计算第 i 个 grd 和第 j 个 sat 描述符的相似度图，得到 score_map[i, j]
                norm = sat_map_norm[i] * grd_map_norm[j]
                if (norm == 0).all():
                    print_colored("norm is zero!")
                score_maps5[i, j] = torch.sum(grd_descriptor_map5[i] * x[j], dim=0) / (norm + 1e-8)

        # for i in range(20):
        #     sat_descriptor_map_rolled = torch.roll(x, shifts=-i*4, dims=1)
        #     sat_descriptor_map_window = sat_descriptor_map_rolled[:,:grd_des_len, :, :]
        #     sat_map_norm = torch.norm(sat_descriptor_map_window, p='fro', dim=1, keepdim=True)
        #
        #     matching_score = torch.sum((grd_descriptor_map5*sat_descriptor_map_window), dim=1, keepdim=True) / (sat_map_norm * grd_map_norm) # cosine similarity
        #     if i == 0:
        #         matching_score_stacked5 = matching_score
        #     else:
        #         matching_score_stacked5 = torch.cat([matching_score_stacked5, matching_score], dim=1)
        # matching_score_max, _ = torch.max(matching_score_stacked5, dim=1, keepdim=True)
        #
        x = torch.cat([score_maps5[torch.arange(B), torch.arange(B)], self.sat_normalization(x)], dim=1)
        x = self.deconv2(x)
        x = torch.cat([x, sat_feature_block0], dim=1)
        x = self.conv2(x)

        # matching 256*256
        grd_des_len = grd_descriptor6.size()[1]  # 40
        sat_des_len = x.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map6, p='fro', dim=1, keepdim=True)
        sat_map_norm = torch.norm(x, p='fro', dim=1, keepdim=True)
        if torch.isnan(x).any():
            print_colored("NaN detected in sat_descriptor_map4!")
            print(f"sat_descriptor_map4: {x}")

        B, C, H, W = grd_descriptor_map6.shape
        score_maps6 = torch.zeros(B, B, 1, H, W).cuda()
        for i in range(B):
            for j in range(B):
                norm = sat_map_norm[i] * grd_map_norm[j]
                if (norm == 0).all():
                    print_colored("norm is zero!")
                score_maps6[i, j] = torch.sum(grd_descriptor_map6[i] * x[j], dim=0) / (norm + 1e-8)

        # for i in range(20):
        #     sat_descriptor_map_rolled = torch.roll(x, shifts=-i*2, dims=1)
        #     sat_descriptor_map_window = sat_descriptor_map_rolled[:,:grd_des_len, :, :]
        #     sat_map_norm = torch.norm(sat_descriptor_map_window, p='fro', dim=1, keepdim=True)
        #
        #     matching_score = torch.sum((grd_descriptor_map6*sat_descriptor_map_window), dim=1, keepdim=True) / (sat_map_norm * grd_map_norm) # cosine similarity
        #     if i == 0:
        #         matching_score_stacked6 = matching_score
        #     else:
        #         matching_score_stacked6 = torch.cat([matching_score_stacked6, matching_score], dim=1)
        # matching_score_max, _ = torch.max(matching_score_stacked6, dim=1, keepdim=True)
        x = torch.cat([score_maps6[torch.arange(B), torch.arange(B)], self.sat_normalization(x)], dim=1)
        x = self.deconv1(x)
        x = self.conv1(x)

        logits_flattened = torch.flatten(x, start_dim=1)
        heatmap = torch.reshape(nn.Softmax(dim=-1)(logits_flattened), x.size())
        #
        # # ori
        # x_ori = torch.cat([score_maps1[torch.arange(B), torch.arange(B)], self.sat_normalization(sat_descriptor_map)], dim=1) #[8, 1300, 8, 8]
        # x_ori = self.deconv6_ori(x_ori)#[8, 1024, 16, 16]
        # x_ori = torch.cat([x_ori, sat_feature_block15], dim=1)#[8, 1344, 16, 16]
        # x_ori = self.conv6_ori(x_ori)# [8, 640, 16, 16]
        # x_ori = self.deconv5_ori(x_ori)#[8, 256, 32, 32]
        # x_ori = torch.cat([x_ori, sat_feature_block10], dim=1)#[8, 368, 32, 32]
        # x_ori = self.conv5_ori(x_ori)#[8, 256, 32, 32]
        # x_ori = self.deconv4_ori(x_ori)#[8, 128, 64, 64]
        # x_ori = torch.cat([x_ori, sat_feature_block4], dim=1)#[8, 168, 64, 64]
        # x_ori = self.conv4_ori(x_ori)#[8, 128, 64, 64]
        # x_ori = self.deconv3_ori(x_ori)#[8, 64, 128, 128]
        # x_ori = torch.cat([x_ori, sat_feature_block2], dim=1)#[8, 88, 128, 128]
        # x_ori = self.conv3_ori(x_ori)#[8, 64, 128, 128]
        # x_ori = self.deconv2_ori(x_ori)#[8, 32, 256, 256]
        # x_ori = torch.cat([x_ori, sat_feature_block0], dim=1)#[8, 48, 256, 256]
        # x_ori = self.conv2_ori(x_ori)#[8, 32, 256, 256]
        # x_ori = self.deconv1_ori(x_ori)#[8, 16, 512, 512]
        # x_ori = self.conv1_ori(x_ori)#[8, 2, 512, 512]
        # x_ori = nn.functional.normalize(x_ori, p=2, dim=1)#[8, 2, 512, 512] channel0=cosine, channel1=sine

        # return logits_flattened, heatmap, x_ori, matching_score_stacked, matching_score_stacked2, matching_score_stacked3, matching_score_stacked4, matching_score_stacked5, matching_score_stacked6, \
        # score_maps1, score_maps2, score_maps3, score_maps4, score_maps5, score_maps6
        return logits_flattened, heatmap, \
            score_maps1, score_maps2, score_maps3, score_maps4, score_maps5, score_maps6


class CVM_VIGOR_no_negative(nn.Module):
    def __init__(self, device, circular_padding):
        super(CVM_VIGOR_no_negative, self).__init__()
        self.device = device
        self.circular_padding = circular_padding

        self.grd_efficientnet = EfficientNet.from_pretrained('efficientnet-b0', self.circular_padding)

        self.grd_feature_to_descriptor1 = nn.Sequential(
            nn.Conv2d(1280, 64, 1),
            permute_channels(0, 2, 3, 1),
            nn.Conv2d(10, 1, 1),
            nn.Flatten(start_dim=1)
        )

        self.grd_feature_to_descriptor2 = nn.Sequential(
            nn.Conv2d(1280, 32, 1),
            permute_channels(0, 2, 3, 1),
            nn.Conv2d(10, 1, 1),
            nn.Flatten(start_dim=1)
        )

        self.grd_feature_to_descriptor3 = nn.Sequential(
            nn.Conv2d(1280, 16, 1),
            permute_channels(0, 2, 3, 1),
            nn.Conv2d(10, 1, 1),
            nn.Flatten(start_dim=1)
        )

        self.grd_feature_to_descriptor4 = nn.Sequential(
            nn.Conv2d(1280, 8, 1),
            permute_channels(0, 2, 3, 1),
            nn.Conv2d(10, 1, 1),
            nn.Flatten(start_dim=1)
        )

        self.grd_feature_to_descriptor5 = nn.Sequential(
            nn.Conv2d(1280, 4, 1),
            permute_channels(0, 2, 3, 1),
            nn.Conv2d(10, 1, 1),
            nn.Flatten(start_dim=1)
        )

        self.grd_feature_to_descriptor6 = nn.Sequential(
            nn.Conv2d(1280, 2, 1),
            permute_channels(0, 2, 3, 1),
            nn.Conv2d(10, 1, 1),
            nn.Flatten(start_dim=1)
        )

        self.sat_efficientnet = EfficientNet.from_pretrained('efficientnet-b0', circular=False)

        self.sat_feature_to_descriptors = nn.Sequential(nn.Flatten(start_dim=1),
                                                        nn.Linear(1280 * 2 * 2, 1280)
                                                        )

        self.sat_normalization = normalization(2, 1)

        # loc
        self.deconv6 = nn.ConvTranspose2d(1281, 1024, 2, 2)
        self.conv6 = double_conv(1344, 640)

        self.deconv5 = nn.ConvTranspose2d(641, 320, 2, 2)
        self.conv5 = double_conv(432, 320)

        self.deconv4 = nn.ConvTranspose2d(321, 160, 2, 2)
        self.conv4 = double_conv(200, 160)

        self.deconv3 = nn.ConvTranspose2d(161, 80, 2, 2)
        self.conv3 = double_conv(104, 80)

        self.deconv2 = nn.ConvTranspose2d(81, 40, 2, 2)
        self.conv2 = double_conv(56, 40)

        self.deconv1 = nn.ConvTranspose2d(41, 16, 2, 2)
        self.conv1 = nn.Sequential(nn.Conv2d(16, 16, 3, stride=1, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(16, 1, 3, stride=1, padding=1))

        # ori
        self.deconv6_ori = nn.ConvTranspose2d(1300, 1024, 2, 2)
        self.conv6_ori = double_conv(1344, 640)

        self.deconv5_ori = nn.ConvTranspose2d(640, 256, 2, 2)
        self.conv5_ori = double_conv(368, 256)

        self.deconv4_ori = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv4_ori = double_conv(168, 128)

        self.deconv3_ori = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv3_ori = double_conv(88, 64)

        self.deconv2_ori = nn.ConvTranspose2d(64, 32, 2, 2)
        self.conv2_ori = double_conv(48, 32)

        self.deconv1_ori = nn.ConvTranspose2d(32, 16, 2, 2)
        self.conv1_ori = nn.Sequential(nn.Conv2d(16, 16, 3, stride=1, padding=1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(16, 2, 3, stride=1, padding=1))

    def forward(self, grd, sat):
        grd_feature_volume = self.grd_efficientnet.extract_features(grd)  # shape=[8, 1280, 10, 20] 1280 是64*20
        grd_descriptor1 = self.grd_feature_to_descriptor1(grd_feature_volume)  # length 1280 shape=[8, 1280]
        grd_descriptor2 = self.grd_feature_to_descriptor2(grd_feature_volume)  # length 640 shape=[8, 640]
        grd_descriptor3 = self.grd_feature_to_descriptor3(grd_feature_volume)  # length 320
        grd_descriptor4 = self.grd_feature_to_descriptor4(grd_feature_volume)  # length 160
        grd_descriptor5 = self.grd_feature_to_descriptor5(grd_feature_volume)  # length 80
        grd_descriptor6 = self.grd_feature_to_descriptor6(grd_feature_volume)  # length 40

        grd_descriptor_map1 = grd_descriptor1.unsqueeze(2).unsqueeze(3).repeat(1, 1, 8, 8)
        grd_descriptor_map2 = grd_descriptor2.unsqueeze(2).unsqueeze(3).repeat(1, 1, 16, 16)
        grd_descriptor_map3 = grd_descriptor3.unsqueeze(2).unsqueeze(3).repeat(1, 1, 32, 32)
        grd_descriptor_map4 = grd_descriptor4.unsqueeze(2).unsqueeze(3).repeat(1, 1, 64, 64)
        grd_descriptor_map5 = grd_descriptor5.unsqueeze(2).unsqueeze(3).repeat(1, 1, 128, 128)
        grd_descriptor_map6 = grd_descriptor6.unsqueeze(2).unsqueeze(3).repeat(1, 1, 256, 256)

        sat_feature_volume, multiscale_sat = self.sat_efficientnet.extract_features_multiscale(sat)
        # len(multiscale_sat)=16, sat_feature_volume.shape = [8, 1280, 16, 16]
        sat_feature_block0 = multiscale_sat[0]  # [16, 256, 256]
        sat_feature_block2 = multiscale_sat[2]  # [24, 128, 128]
        sat_feature_block4 = multiscale_sat[4]  # [40, 64, 64]
        sat_feature_block10 = multiscale_sat[10]  # [112, 32, 32]
        sat_feature_block15 = multiscale_sat[15]  # [320, 16, 16]

        sat_row_chunks = torch.stack(list(torch.chunk(sat_feature_volume, 8, 2)),
                                     dim=-1)  # dimension 4 is the number of row chunks (splitted in height dimension)
        for i, sat_row_chunk in enumerate(torch.unbind(sat_row_chunks, dim=-1), 0):
            sat_chunks = torch.stack(list(torch.chunk(sat_row_chunk, 8, 3)),
                                     dim=-1)  # dimension 5 is the number of vertical chunks (splitted in width dimension)
            for j, sat_chunk in enumerate(torch.unbind(sat_chunks, dim=-1), 0):
                # sat_chunk.shape=[8, 1280, 2, 2]
                if j == 0:
                    # sat_descriptor_row.shape=[8, 1280, 1, 1]
                    sat_descriptor_row = self.sat_feature_to_descriptors(sat_chunk).unsqueeze(2).unsqueeze(3)
                else:
                    sat_descriptor_row = torch.cat(
                        (sat_descriptor_row, self.sat_feature_to_descriptors(sat_chunk).unsqueeze(2).unsqueeze(3)), 3)
            if i == 0:
                sat_descriptor_map = sat_descriptor_row
            else:
                sat_descriptor_map = torch.cat((sat_descriptor_map, sat_descriptor_row), 2)

        # matching bottleneck
        # sat_descriptor_map.shape=[8, 1280, 8, 8]
        grd_des_len = grd_descriptor1.size()[1]
        sat_des_len = sat_descriptor_map.size()[1]
        # grd_map_norm.shape=[8, 1, 8, 8]
        grd_map_norm = torch.norm(grd_descriptor_map1, p='fro', dim=1, keepdim=True)
        sat_map_norm = torch.norm(sat_descriptor_map, p='fro', dim=1, keepdim=True)
        if torch.isnan(sat_descriptor_map).any():
            print_colored("NaN detected in sat_descriptor_map!")
            print(f"sat_descriptor_map: {sat_descriptor_map}")

        B, C, H, W = grd_descriptor_map1.shape

        score_maps1 = torch.sum((grd_descriptor_map1 * sat_descriptor_map), dim=1, keepdim=True) / (
                    sat_map_norm * grd_map_norm)  # cosine similarity

        # score_maps1 = torch.zeros(B, B, 1, H, W).cuda()
        # for i in range(B):
        #     for j in range(B):
        #         # 计算第 i 个 grd 和第 j 个 sat 描述符的相似度图，得到 score_map[i, j]
        #         matching = torch.sum(grd_descriptor_map1[i] * sat_descriptor_map[j], dim=0)
        #         norm = sat_map_norm[i] * grd_map_norm[j]
        #         if (norm == 0).all():
        #             print_colored("norm is zero!")
        #             print(f"grd_feature_volume: {grd_feature_volume}")
        #             print(f"sat_feature_volume: {sat_feature_volume}")
        #         # matching shape=[8, 8], matching2.shape=[1, 8, 8]
        #         score_maps1[i, j, 0] = matching / (norm + 1e-8)

        # for i in range(20):
        #     sat_descriptor_map_rolled = torch.roll(sat_descriptor_map, shifts=-i*64, dims=1)
        #     sat_descriptor_map_window = sat_descriptor_map_rolled[:,:grd_des_len, :, :]
        #     sat_map_norm = torch.norm(sat_descriptor_map_window, p='fro', dim=1, keepdim=True)
        #     # grd_descriptor_map1.shape=[8, 1280, 8, 8], sat_descriptor_map_window.shape=[8, 1280, 8, 8]
        #     matching_score = torch.sum((grd_descriptor_map1*sat_descriptor_map_window), dim=1, keepdim=True) / (sat_map_norm * grd_map_norm) # cosine similarity
        #     if i == 0:
        #         matching_score_stacked = matching_score
        #     else:
        #         matching_score_stacked = torch.cat([matching_score_stacked, matching_score], dim=1)

        # matching_score_stacked.shape=[8, 20, 8, 8]
        # matching_score_max.shape = [8, 1, 8, 8]
        # matching_score_max, _ = torch.max(matching_score_stacked, dim=1, keepdim=True)
        # matching_score_max2 = score_maps1[torch.arange(B), torch.arange(B)] #[8, 1, 8, 8]

        # loc
        x = torch.cat([score_maps1, self.sat_normalization(sat_descriptor_map)],
                      dim=1)  # [8, 1281, 8,8]

        x = self.deconv6(x)  # [8, 1024, 16, 16]
        x = torch.cat([x, sat_feature_block15], dim=1)
        x = self.conv6(x)  # [8, 640, 16, 16]

        # matching 16*16
        grd_des_len = grd_descriptor2.size()[1]  # 640
        sat_des_len = x.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map2, p='fro', dim=1, keepdim=True)
        sat_map_norm = torch.norm(x, p='fro', dim=1, keepdim=True)
        if torch.isnan(x).any():
            print_colored("NaN detected in sat_descriptor_map2!")
            print(f"sat_descriptor_map2: {x}")

        B, C, H, W = grd_descriptor_map2.shape
        score_maps2 = torch.sum((grd_descriptor_map2 * x), dim=1, keepdim=True) / (
                sat_map_norm * grd_map_norm)  # cosine similarity

        # score_maps2 = torch.zeros(B, B, 1, H, W).cuda()
        # for i in range(B):
        #     for j in range(B):
        #         # 计算第 i 个 grd 和第 j 个 sat 描述符的相似度图，得到 score_map[i, j]
        #         norm = sat_map_norm[i] * grd_map_norm[j]
        #         if (norm == 0).all():
        #             print_colored("norm is zero!")
        #         score_maps2[i, j] = torch.sum(grd_descriptor_map2[i] * x[j], dim=0) / (norm + 1e-8)

        # for i in range(20):
        #     sat_descriptor_map_rolled = torch.roll(x, shifts=-i*32, dims=1)
        #     sat_descriptor_map_window = sat_descriptor_map_rolled[:,:grd_des_len, :, :]
        #     sat_map_norm = torch.norm(sat_descriptor_map_window, p='fro', dim=1, keepdim=True)
        #
        #     matching_score = torch.sum((grd_descriptor_map2*sat_descriptor_map_window), dim=1, keepdim=True) / (sat_map_norm * grd_map_norm) # cosine similarity
        #     if i == 0:
        #         matching_score_stacked2 = matching_score
        #     else:
        #         matching_score_stacked2 = torch.cat([matching_score_stacked2, matching_score], dim=1)
        # matching_score_max, _ = torch.max(matching_score_stacked2, dim=1, keepdim=True)
        #
        x = torch.cat([score_maps2, self.sat_normalization(x)], dim=1)
        x = self.deconv5(x)
        x = torch.cat([x, sat_feature_block10], dim=1)
        x = self.conv5(x)

        # matching 32*32
        grd_des_len = grd_descriptor3.size()[1]  # 320
        sat_des_len = x.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map3, p='fro', dim=1, keepdim=True)
        sat_map_norm = torch.norm(x, p='fro', dim=1, keepdim=True)
        if torch.isnan(x).any():
            print_colored("NaN detected in sat_descriptor_map3!")
            print(f"sat_descriptor_map3: {x}")

        B, C, H, W = grd_descriptor_map3.shape
        score_maps3 = torch.sum((grd_descriptor_map3 * x), dim=1, keepdim=True) / (
                sat_map_norm * grd_map_norm)  # cosine similarity
        # score_maps3 = torch.zeros(B, B, 1, H, W).cuda()
        # for i in range(B):
        #     for j in range(B):
        #         # 计算第 i 个 grd 和第 j 个 sat 描述符的相似度图，得到 score_map[i, j]
        #         norm = sat_map_norm[i] * grd_map_norm[j]
        #         if (norm == 0).all():
        #             print_colored("norm is zero!")
        #         score_maps3[i, j] = torch.sum(grd_descriptor_map3[i] * x[j], dim=0) / (norm + 1e-8)

        # for i in range(20):
        #     sat_descriptor_map_rolled = torch.roll(x, shifts=-i*16, dims=1)
        #     sat_descriptor_map_window = sat_descriptor_map_rolled[:,:grd_des_len, :, :]
        #     sat_map_norm = torch.norm(sat_descriptor_map_window, p='fro', dim=1, keepdim=True)
        #
        #     matching_score = torch.sum((grd_descriptor_map3*sat_descriptor_map_window), dim=1, keepdim=True) / (sat_map_norm * grd_map_norm) # cosine similarity
        #     if i == 0:
        #         matching_score_stacked3 = matching_score
        #     else:
        #         matching_score_stacked3 = torch.cat([matching_score_stacked3, matching_score], dim=1)
        # matching_score_max, _ = torch.max(matching_score_stacked3, dim=1, keepdim=True)
        #
        x = torch.cat([score_maps3, self.sat_normalization(x)], dim=1)
        x = self.deconv4(x)
        x = torch.cat([x, sat_feature_block4], dim=1)
        x = self.conv4(x)

        # matching 64*64
        grd_des_len = grd_descriptor4.size()[1]  # 160
        sat_des_len = x.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map4, p='fro', dim=1, keepdim=True)
        sat_map_norm = torch.norm(x, p='fro', dim=1, keepdim=True)
        if torch.isnan(x).any():
            print_colored("NaN detected in sat_descriptor_map4!")
            print(f"sat_descriptor_map4: {x}")

        B, C, H, W = grd_descriptor_map4.shape
        score_maps4 = torch.sum((grd_descriptor_map4 * x), dim=1, keepdim=True) / (
                sat_map_norm * grd_map_norm)  # cosine similarity
        # score_maps4 = torch.zeros(B, B, 1, H, W).cuda()
        # for i in range(B):
        #     for j in range(B):
        #         norm = sat_map_norm[i] * grd_map_norm[j]
        #         if (norm == 0).all():
        #             print_colored("norm is zero!")
        #         # 计算第 i 个 grd 和第 j 个 sat 描述符的相似度图，得到 score_map[i, j]
        #         score_maps4[i, j] = torch.sum(grd_descriptor_map4[i] * x[j], dim=0) / (norm + 1e-8)

        # for i in range(20):
        #     sat_descriptor_map_rolled = torch.roll(x, shifts=-i*8, dims=1)
        #     sat_descriptor_map_window = sat_descriptor_map_rolled[:,:grd_des_len, :, :]
        #     sat_map_norm = torch.norm(sat_descriptor_map_window, p='fro', dim=1, keepdim=True)
        #
        #     matching_score = torch.sum((grd_descriptor_map4*sat_descriptor_map_window), dim=1, keepdim=True) / (sat_map_norm * grd_map_norm) # cosine similarity
        #     if i == 0:
        #         matching_score_stacked4 = matching_score
        #     else:
        #         matching_score_stacked4 = torch.cat([matching_score_stacked4, matching_score], dim=1)
        # matching_score_max, _ = torch.max(matching_score_stacked4, dim=1, keepdim=True)
        #
        x = torch.cat([score_maps4, self.sat_normalization(x)], dim=1)
        x = self.deconv3(x)
        x = torch.cat([x, sat_feature_block2], dim=1)
        x = self.conv3(x)

        # matching 128*128
        grd_des_len = grd_descriptor5.size()[1]  # 80
        sat_des_len = x.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map5, p='fro', dim=1, keepdim=True)
        sat_map_norm = torch.norm(x, p='fro', dim=1, keepdim=True)
        if torch.isnan(x).any():
            print_colored("NaN detected in sat_descriptor_map4!")
            print(f"sat_descriptor_map4: {x}")

        B, C, H, W = grd_descriptor_map5.shape
        score_maps5 = torch.sum((grd_descriptor_map5 * x), dim=1, keepdim=True) / (
                sat_map_norm * grd_map_norm)  # cosine similarity
        # score_maps5 = torch.zeros(B, B, 1, H, W).cuda()
        # for i in range(B):
        #     for j in range(B):
        #         # 计算第 i 个 grd 和第 j 个 sat 描述符的相似度图，得到 score_map[i, j]
        #         norm = sat_map_norm[i] * grd_map_norm[j]
        #         if (norm == 0).all():
        #             print_colored("norm is zero!")
        #         score_maps5[i, j] = torch.sum(grd_descriptor_map5[i] * x[j], dim=0) / (norm + 1e-8)

        # for i in range(20):
        #     sat_descriptor_map_rolled = torch.roll(x, shifts=-i*4, dims=1)
        #     sat_descriptor_map_window = sat_descriptor_map_rolled[:,:grd_des_len, :, :]
        #     sat_map_norm = torch.norm(sat_descriptor_map_window, p='fro', dim=1, keepdim=True)
        #
        #     matching_score = torch.sum((grd_descriptor_map5*sat_descriptor_map_window), dim=1, keepdim=True) / (sat_map_norm * grd_map_norm) # cosine similarity
        #     if i == 0:
        #         matching_score_stacked5 = matching_score
        #     else:
        #         matching_score_stacked5 = torch.cat([matching_score_stacked5, matching_score], dim=1)
        # matching_score_max, _ = torch.max(matching_score_stacked5, dim=1, keepdim=True)
        #
        x = torch.cat([score_maps5, self.sat_normalization(x)], dim=1)
        x = self.deconv2(x)
        x = torch.cat([x, sat_feature_block0], dim=1)
        x = self.conv2(x)

        # matching 256*256
        grd_des_len = grd_descriptor6.size()[1]  # 40
        sat_des_len = x.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map6, p='fro', dim=1, keepdim=True)
        sat_map_norm = torch.norm(x, p='fro', dim=1, keepdim=True)
        if torch.isnan(x).any():
            print_colored("NaN detected in sat_descriptor_map4!")
            print(f"sat_descriptor_map4: {x}")


        score_maps6 = torch.sum((grd_descriptor_map6 * x), dim=1, keepdim=True) / (
                sat_map_norm * grd_map_norm)  # cosine similarity
        B, C, H, W = grd_descriptor_map6.shape
        # score_maps6 = torch.zeros(B, B, 1, H, W).cuda()
        # for i in range(B):
        #     for j in range(B):
        #         norm = sat_map_norm[i] * grd_map_norm[j]
        #         if (norm == 0).all():
        #             print_colored("norm is zero!")
        #         score_maps6[i, j] = torch.sum(grd_descriptor_map6[i] * x[j], dim=0) / (norm + 1e-8)

        # for i in range(20):
        #     sat_descriptor_map_rolled = torch.roll(x, shifts=-i*2, dims=1)
        #     sat_descriptor_map_window = sat_descriptor_map_rolled[:,:grd_des_len, :, :]
        #     sat_map_norm = torch.norm(sat_descriptor_map_window, p='fro', dim=1, keepdim=True)
        #
        #     matching_score = torch.sum((grd_descriptor_map6*sat_descriptor_map_window), dim=1, keepdim=True) / (sat_map_norm * grd_map_norm) # cosine similarity
        #     if i == 0:
        #         matching_score_stacked6 = matching_score
        #     else:
        #         matching_score_stacked6 = torch.cat([matching_score_stacked6, matching_score], dim=1)
        # matching_score_max, _ = torch.max(matching_score_stacked6, dim=1, keepdim=True)
        x = torch.cat([score_maps6, self.sat_normalization(x)], dim=1)
        x = self.deconv1(x)
        x = self.conv1(x)

        logits_flattened = torch.flatten(x, start_dim=1)
        heatmap = torch.reshape(nn.Softmax(dim=-1)(logits_flattened), x.size())
        #
        # # ori
        # x_ori = torch.cat([score_maps1[torch.arange(B), torch.arange(B)], self.sat_normalization(sat_descriptor_map)], dim=1) #[8, 1300, 8, 8]
        # x_ori = self.deconv6_ori(x_ori)#[8, 1024, 16, 16]
        # x_ori = torch.cat([x_ori, sat_feature_block15], dim=1)#[8, 1344, 16, 16]
        # x_ori = self.conv6_ori(x_ori)# [8, 640, 16, 16]
        # x_ori = self.deconv5_ori(x_ori)#[8, 256, 32, 32]
        # x_ori = torch.cat([x_ori, sat_feature_block10], dim=1)#[8, 368, 32, 32]
        # x_ori = self.conv5_ori(x_ori)#[8, 256, 32, 32]
        # x_ori = self.deconv4_ori(x_ori)#[8, 128, 64, 64]
        # x_ori = torch.cat([x_ori, sat_feature_block4], dim=1)#[8, 168, 64, 64]
        # x_ori = self.conv4_ori(x_ori)#[8, 128, 64, 64]
        # x_ori = self.deconv3_ori(x_ori)#[8, 64, 128, 128]
        # x_ori = torch.cat([x_ori, sat_feature_block2], dim=1)#[8, 88, 128, 128]
        # x_ori = self.conv3_ori(x_ori)#[8, 64, 128, 128]
        # x_ori = self.deconv2_ori(x_ori)#[8, 32, 256, 256]
        # x_ori = torch.cat([x_ori, sat_feature_block0], dim=1)#[8, 48, 256, 256]
        # x_ori = self.conv2_ori(x_ori)#[8, 32, 256, 256]
        # x_ori = self.deconv1_ori(x_ori)#[8, 16, 512, 512]
        # x_ori = self.conv1_ori(x_ori)#[8, 2, 512, 512]
        # x_ori = nn.functional.normalize(x_ori, p=2, dim=1)#[8, 2, 512, 512] channel0=cosine, channel1=sine

        # return logits_flattened, heatmap, x_ori, matching_score_stacked, matching_score_stacked2, matching_score_stacked3, matching_score_stacked4, matching_score_stacked5, matching_score_stacked6, \
        # score_maps1, score_maps2, score_maps3, score_maps4, score_maps5, score_maps6
        return logits_flattened, heatmap, \
            score_maps1, score_maps2, score_maps3, score_maps4, score_maps5, score_maps6


class CVM_VIGOR_ori_prior(nn.Module):
    def __init__(self, device, ori_noise, circular_padding=True):
        super(CVM_VIGOR_ori_prior, self).__init__()
        self.device = device
        self.circular_padding = circular_padding
        self.ori_noise = ori_noise
        
        self.grd_efficientnet = EfficientNet.from_pretrained('efficientnet-b0', self.circular_padding)

        self.grd_feature_to_descriptor1 = nn.Sequential(
                                    nn.Conv2d(1280, 64, 1),
                                    permute_channels(0, 2, 3, 1),
                                    nn.Conv2d(10, 1, 1),
                                    nn.Flatten(start_dim=1)
                                    )
        
        self.grd_feature_to_descriptor2 = nn.Sequential(
                                    nn.Conv2d(1280, 32, 1),
                                    permute_channels(0, 2, 3, 1),
                                    nn.Conv2d(10, 1, 1),
                                    nn.Flatten(start_dim=1)
                                    )
        
        self.grd_feature_to_descriptor3 = nn.Sequential(
                                    nn.Conv2d(1280, 16, 1),
                                    permute_channels(0, 2, 3, 1),
                                    nn.Conv2d(10, 1, 1),
                                    nn.Flatten(start_dim=1)
                                    )
        
        self.grd_feature_to_descriptor4 = nn.Sequential(
                                    nn.Conv2d(1280, 8, 1),
                                    permute_channels(0, 2, 3, 1),
                                    nn.Conv2d(10, 1, 1),
                                    nn.Flatten(start_dim=1)
                                    )
        
        self.grd_feature_to_descriptor5 = nn.Sequential(
                                    nn.Conv2d(1280, 4, 1),
                                    permute_channels(0, 2, 3, 1),
                                    nn.Conv2d(10, 1, 1),
                                    nn.Flatten(start_dim=1)
                                    )
        
        self.grd_feature_to_descriptor6 = nn.Sequential(
                                    nn.Conv2d(1280, 2, 1),
                                    permute_channels(0, 2, 3, 1),
                                    nn.Conv2d(10, 1, 1),
                                    nn.Flatten(start_dim=1)
                                    )
        
        self.sat_efficientnet = EfficientNet.from_pretrained('efficientnet-b0', False)
        
        
        self.sat_feature_to_descriptors = nn.Sequential(nn.Flatten(start_dim=1),
                                                        nn.Linear(1280*2*2, 1280)
                                                       )
        
        self.sat_normalization = normalization(2, 1)
        
        # loc
        self.deconv6 = nn.ConvTranspose2d(1281, 1024, 2, 2)
        self.conv6 = double_conv(1344, 640)
                                    
        self.deconv5 = nn.ConvTranspose2d(641, 320, 2, 2)
        self.conv5 = double_conv(432, 320)
        
        self.deconv4 = nn.ConvTranspose2d(321, 160, 2, 2)
        self.conv4 = double_conv(200, 160)
        
        self.deconv3 = nn.ConvTranspose2d(161, 80, 2, 2)
        self.conv3 = double_conv(104, 80)
        
        self.deconv2 = nn.ConvTranspose2d(81, 40, 2, 2)
        self.conv2 = double_conv(56, 40)
        
        self.deconv1 = nn.ConvTranspose2d(41, 16, 2, 2)
        self.conv1 = nn.Sequential(nn.Conv2d(16, 16, 3, stride=1, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(16, 1, 3, stride=1, padding=1))
        
        # ori
        self.deconv6_ori = nn.ConvTranspose2d(1300, 1024, 2, 2)
        self.conv6_ori = double_conv(1344, 640)
                                    
        self.deconv5_ori = nn.ConvTranspose2d(640, 256, 2, 2)
        self.conv5_ori = double_conv(368, 256)
        
        self.deconv4_ori = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv4_ori = double_conv(168, 128)
        
        self.deconv3_ori = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv3_ori = double_conv(88, 64)
        
        self.deconv2_ori = nn.ConvTranspose2d(64, 32, 2, 2)
        self.conv2_ori = double_conv(48, 32)
        
        self.deconv1_ori = nn.ConvTranspose2d(32, 16, 2, 2)
        self.conv1_ori = nn.Sequential(nn.Conv2d(16, 16, 3, stride=1, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(16, 2, 3, stride=1, padding=1))
        
    def forward(self, grd, sat):
        grd_feature_volume = self.grd_efficientnet.extract_features(grd)
        grd_descriptor1 = self.grd_feature_to_descriptor1(grd_feature_volume) # length 1280
        grd_descriptor2 = self.grd_feature_to_descriptor2(grd_feature_volume) # length 640
        grd_descriptor3 = self.grd_feature_to_descriptor3(grd_feature_volume) # length 320
        grd_descriptor4 = self.grd_feature_to_descriptor4(grd_feature_volume) # length 160
        grd_descriptor5 = self.grd_feature_to_descriptor5(grd_feature_volume) # length 80
        grd_descriptor6 = self.grd_feature_to_descriptor6(grd_feature_volume) # length 40
        
        grd_descriptor_map1 = grd_descriptor1.unsqueeze(2).unsqueeze(3).repeat(1, 1, 8, 8)
        grd_descriptor_map2 = grd_descriptor2.unsqueeze(2).unsqueeze(3).repeat(1, 1, 16, 16)
        grd_descriptor_map3 = grd_descriptor3.unsqueeze(2).unsqueeze(3).repeat(1, 1, 32, 32)
        grd_descriptor_map4 = grd_descriptor4.unsqueeze(2).unsqueeze(3).repeat(1, 1, 64, 64)
        grd_descriptor_map5 = grd_descriptor5.unsqueeze(2).unsqueeze(3).repeat(1, 1, 128, 128)
        grd_descriptor_map6 = grd_descriptor6.unsqueeze(2).unsqueeze(3).repeat(1, 1, 256, 256)
      
        sat_feature_volume, multiscale_sat = self.sat_efficientnet.extract_features_multiscale(sat)
        sat_feature_block0 = multiscale_sat[0] # [16, 256, 256]
        sat_feature_block2 = multiscale_sat[2] #[24, 128, 128]
        sat_feature_block4 = multiscale_sat[4] # [40, 64, 64]
        sat_feature_block10 = multiscale_sat[10] # [112, 32, 32]
        sat_feature_block15 = multiscale_sat[15] # [320, 16, 16]
        
        sat_row_chunks = torch.stack(list(torch.chunk(sat_feature_volume, 8, 2)), dim=-1) # dimension 4 is the number of row chunks (splitted in height dimension)
        for i, sat_row_chunk in enumerate(torch.unbind(sat_row_chunks, dim=-1), 0):
            sat_chunks = torch.stack(list(torch.chunk(sat_row_chunk, 8, 3)), dim=-1) # dimension 5 is the number of vertical chunks (splitted in width dimension)
            for j, sat_chunk in enumerate(torch.unbind(sat_chunks, dim=-1), 0):
                if j == 0:
                    sat_descriptor_row = self.sat_feature_to_descriptors(sat_chunk).unsqueeze(2).unsqueeze(3)
                else:
                    sat_descriptor_row = torch.cat((sat_descriptor_row, self.sat_feature_to_descriptors(sat_chunk).unsqueeze(2).unsqueeze(3)), 3)
            if i == 0:
                sat_descriptor_map = sat_descriptor_row
            else:
                sat_descriptor_map = torch.cat((sat_descriptor_map, sat_descriptor_row), 2)
        
        # matching bottleneck
        grd_des_len = grd_descriptor1.size()[1]
        sat_des_len = sat_descriptor_map.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map1, p='fro', dim=1, keepdim=True)
        
        for i in range(-int(self.ori_noise/18), int(self.ori_noise/18)+1):
            sat_descriptor_map_rolled = torch.roll(sat_descriptor_map, shifts=-i*64, dims=1)
            sat_descriptor_map_window = sat_descriptor_map_rolled[:,:grd_des_len, :, :]
            sat_map_norm = torch.norm(sat_descriptor_map_window, p='fro', dim=1, keepdim=True)

            matching_score = torch.sum((grd_descriptor_map1*sat_descriptor_map_window), dim=1, keepdim=True) / (sat_map_norm * grd_map_norm) # cosine similarity
            if i == -int(self.ori_noise/18):
                matching_score_stacked = matching_score
            else:
                matching_score_stacked = torch.cat([matching_score_stacked, matching_score], dim=1)
        matching_score_max, _ = torch.max(matching_score_stacked, dim=1, keepdim=True)
        
        # this part generates a full 20-channel matching score volume for orientation decoder
        for i in range(20):
            sat_descriptor_map_rolled = torch.roll(sat_descriptor_map, shifts=-i*64, dims=1)
            sat_descriptor_map_window = sat_descriptor_map_rolled[:,:grd_des_len, :, :]
            sat_map_norm = torch.norm(sat_descriptor_map_window, p='fro', dim=1, keepdim=True)

            matching_score = torch.sum((grd_descriptor_map1*sat_descriptor_map_window), dim=1, keepdim=True) / (sat_map_norm * grd_map_norm) # cosine similarity
            if i == 0:
                matching_score_stacked = matching_score
            else:
                matching_score_stacked = torch.cat([matching_score_stacked, matching_score], dim=1)
                
        # loc
        x = torch.cat([matching_score_max, self.sat_normalization(sat_descriptor_map)], dim=1)

        x = self.deconv6(x)
        x = torch.cat([x, sat_feature_block15], dim=1)
        x = self.conv6(x)
                
        # matching 16*16
        grd_des_len = grd_descriptor2.size()[1] # 640
        sat_des_len = x.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map2, p='fro', dim=1, keepdim=True)
        
        for i in range(-int(self.ori_noise/18), int(self.ori_noise/18)+1):
            sat_descriptor_map_rolled = torch.roll(x, shifts=-i*32, dims=1)
            sat_descriptor_map_window = sat_descriptor_map_rolled[:,:grd_des_len, :, :]
            sat_map_norm = torch.norm(sat_descriptor_map_window, p='fro', dim=1, keepdim=True)

            matching_score = torch.sum((grd_descriptor_map2*sat_descriptor_map_window), dim=1, keepdim=True) / (sat_map_norm * grd_map_norm) # cosine similarity
            if i == -int(self.ori_noise/18):
                matching_score_stacked2 = matching_score
            else:
                matching_score_stacked2 = torch.cat([matching_score_stacked2, matching_score], dim=1)
        matching_score_max, _ = torch.max(matching_score_stacked2, dim=1, keepdim=True)
        
        x = torch.cat([matching_score_max, self.sat_normalization(x)], dim=1)
        x = self.deconv5(x)
        x = torch.cat([x, sat_feature_block10], dim=1)
        x = self.conv5(x)
        
        # matching 32*32
        grd_des_len = grd_descriptor3.size()[1] # 320
        sat_des_len = x.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map3, p='fro', dim=1, keepdim=True)
        
        for i in range(-int(self.ori_noise/18), int(self.ori_noise/18)+1):
            sat_descriptor_map_rolled = torch.roll(x, shifts=-i*16, dims=1)
            sat_descriptor_map_window = sat_descriptor_map_rolled[:,:grd_des_len, :, :]
            sat_map_norm = torch.norm(sat_descriptor_map_window, p='fro', dim=1, keepdim=True)

            matching_score = torch.sum((grd_descriptor_map3*sat_descriptor_map_window), dim=1, keepdim=True) / (sat_map_norm * grd_map_norm) # cosine similarity
            if i == -int(self.ori_noise/18):
                matching_score_stacked3 = matching_score
            else:
                matching_score_stacked3 = torch.cat([matching_score_stacked3, matching_score], dim=1)
        matching_score_max, _ = torch.max(matching_score_stacked3, dim=1, keepdim=True)
        
        x = torch.cat([matching_score_max, self.sat_normalization(x)], dim=1)
        x = self.deconv4(x)
        x = torch.cat([x, sat_feature_block4], dim=1)
        x = self.conv4(x)
        
        # matching 64*64
        grd_des_len = grd_descriptor4.size()[1] # 160
        sat_des_len = x.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map4, p='fro', dim=1, keepdim=True)
        
        for i in range(-int(self.ori_noise/18), int(self.ori_noise/18)+1):
            sat_descriptor_map_rolled = torch.roll(x, shifts=-i*8, dims=1)
            sat_descriptor_map_window = sat_descriptor_map_rolled[:,:grd_des_len, :, :]
            sat_map_norm = torch.norm(sat_descriptor_map_window, p='fro', dim=1, keepdim=True)

            matching_score = torch.sum((grd_descriptor_map4*sat_descriptor_map_window), dim=1, keepdim=True) / (sat_map_norm * grd_map_norm) # cosine similarity
            if i == -int(self.ori_noise/18):
                matching_score_stacked4 = matching_score
            else:
                matching_score_stacked4 = torch.cat([matching_score_stacked4, matching_score], dim=1)
        matching_score_max, _ = torch.max(matching_score_stacked4, dim=1, keepdim=True)
        
        x = torch.cat([matching_score_max, self.sat_normalization(x)], dim=1)
        x = self.deconv3(x)
        x = torch.cat([x, sat_feature_block2], dim=1)
        x = self.conv3(x)
        
        # matching 128*128
        grd_des_len = grd_descriptor5.size()[1] # 80
        sat_des_len = x.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map5, p='fro', dim=1, keepdim=True)
        
        for i in range(-int(self.ori_noise/18), int(self.ori_noise/18)+1):
            sat_descriptor_map_rolled = torch.roll(x, shifts=-i*4, dims=1)
            sat_descriptor_map_window = sat_descriptor_map_rolled[:,:grd_des_len, :, :]
            sat_map_norm = torch.norm(sat_descriptor_map_window, p='fro', dim=1, keepdim=True)

            matching_score = torch.sum((grd_descriptor_map5*sat_descriptor_map_window), dim=1, keepdim=True) / (sat_map_norm * grd_map_norm) # cosine similarity
            if i == -int(self.ori_noise/18):
                matching_score_stacked5 = matching_score
            else:
                matching_score_stacked5 = torch.cat([matching_score_stacked5, matching_score], dim=1)
        matching_score_max, _ = torch.max(matching_score_stacked5, dim=1, keepdim=True)
        
        x = torch.cat([matching_score_max, self.sat_normalization(x)], dim=1)
        x = self.deconv2(x)
        x = torch.cat([x, sat_feature_block0], dim=1)
        x = self.conv2(x)
        
        # matching 256*256
        grd_des_len = grd_descriptor6.size()[1] # 40
        sat_des_len = x.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map6, p='fro', dim=1, keepdim=True)
        
        for i in range(-int(self.ori_noise/18), int(self.ori_noise/18)+1):
            sat_descriptor_map_rolled = torch.roll(x, shifts=-i*2, dims=1)
            sat_descriptor_map_window = sat_descriptor_map_rolled[:,:grd_des_len, :, :]
            sat_map_norm = torch.norm(sat_descriptor_map_window, p='fro', dim=1, keepdim=True)

            matching_score = torch.sum((grd_descriptor_map6*sat_descriptor_map_window), dim=1, keepdim=True) / (sat_map_norm * grd_map_norm) # cosine similarity
            if i == -int(self.ori_noise/18):
                matching_score_stacked6 = matching_score
            else:
                matching_score_stacked6 = torch.cat([matching_score_stacked6, matching_score], dim=1)
        matching_score_max, _ = torch.max(matching_score_stacked6, dim=1, keepdim=True)
        x = torch.cat([matching_score_max, self.sat_normalization(x)], dim=1)
        x = self.deconv1(x)
        x = self.conv1(x)
        
        logits_flattened = torch.flatten(x, start_dim=1)
        heatmap = torch.reshape(nn.Softmax(dim=-1)(logits_flattened), x.size())
        
        # ori
        x_ori = torch.cat([matching_score_stacked, self.sat_normalization(sat_descriptor_map)], dim=1)
        x_ori = self.deconv6_ori(x_ori)
        x_ori = torch.cat([x_ori, sat_feature_block15], dim=1)
        x_ori = self.conv6_ori(x_ori)
        x_ori = self.deconv5_ori(x_ori)
        x_ori = torch.cat([x_ori, sat_feature_block10], dim=1)
        x_ori = self.conv5_ori(x_ori)
        x_ori = self.deconv4_ori(x_ori)
        x_ori = torch.cat([x_ori, sat_feature_block4], dim=1)
        x_ori = self.conv4_ori(x_ori)
        x_ori = self.deconv3_ori(x_ori)
        x_ori = torch.cat([x_ori, sat_feature_block2], dim=1)
        x_ori = self.conv3_ori(x_ori)
        x_ori = self.deconv2_ori(x_ori)
        x_ori = torch.cat([x_ori, sat_feature_block0], dim=1)
        x_ori = self.conv2_ori(x_ori)
        x_ori = self.deconv1_ori(x_ori)
        x_ori = self.conv1_ori(x_ori)
        x_ori = nn.functional.normalize(x_ori, p=2, dim=1)
        
        return logits_flattened, heatmap, x_ori, matching_score_stacked, matching_score_stacked2, matching_score_stacked3, matching_score_stacked4, matching_score_stacked5, matching_score_stacked6
    
    
class CVM_KITTI(nn.Module):
    def __init__(self, device):
        super(CVM_KITTI, self).__init__()
        self.device = device
        
        self.grd_efficientnet = EfficientNet.from_pretrained('efficientnet-b0', False) # no horizontal circular padding

        self.grd_feature_to_descriptor1 = nn.Sequential(
                                    nn.Conv2d(1280, 16, 1),
                                    permute_channels(0, 2, 3, 1),
                                    nn.Conv2d(8, 1, 1),
                                    nn.Flatten(start_dim=1)
                                    )
        
        self.grd_feature_to_descriptor2 = nn.Sequential(
                                    nn.Conv2d(1280, 8, 1),
                                    permute_channels(0, 2, 3, 1),
                                    nn.Conv2d(8, 1, 1),
                                    nn.Flatten(start_dim=1)
                                    )
        
        self.grd_feature_to_descriptor3 = nn.Sequential(
                                    nn.Conv2d(1280, 4, 1),
                                    permute_channels(0, 2, 3, 1),
                                    nn.Conv2d(8, 1, 1),
                                    nn.Flatten(start_dim=1)
                                    )
        self.grd_feature_to_descriptor4 = nn.Sequential(
                                    nn.Conv2d(1280, 2, 1),
                                    permute_channels(0, 2, 3, 1),
                                    nn.Conv2d(8, 1, 1),
                                    nn.Flatten(start_dim=1)
                                    )
        self.grd_feature_to_descriptor5 = nn.Sequential(
                                    nn.Conv2d(1280, 1, 1),
                                    permute_channels(0, 2, 3, 1),
                                    nn.Conv2d(8, 1, 1),
                                    nn.Flatten(start_dim=1)
                                    )
        self.grd_feature_to_descriptor6 = nn.Sequential(
                                    nn.Conv2d(1280, 1, 1),
                                    permute_channels(0, 2, 3, 1),
                                    nn.Conv2d(8, 1, 1),
                                    nn.Flatten(start_dim=1)
                                    )
        
        self.sat_efficientnet = EfficientNet.from_pretrained('efficientnet-b0', False)
        
        
        self.sat_feature_to_descriptors = nn.Sequential(nn.Flatten(start_dim=1),
                                                        nn.Linear(1280*2*2, 2048)
                                                       )
        self.sat_normalization = normalization(2, 1)
        
        # loc
        self.deconv6 = nn.ConvTranspose2d(2048+1, 1024, 2, 2)
        self.conv6 = double_conv(1344, 512)
                                    
        self.deconv5 = nn.ConvTranspose2d(512+1, 256, 2, 2)
        self.conv5 = double_conv(368, 256)
        
        self.deconv4 = nn.ConvTranspose2d(256+1, 128, 2, 2)
        self.conv4 = double_conv(168, 128)
        
        self.deconv3 = nn.ConvTranspose2d(128+1, 64, 2, 2)
        self.conv3 = double_conv(88, 128)
        
        self.deconv2 = nn.ConvTranspose2d(128+1, 32, 2, 2)
        self.conv2 = double_conv(48, 32)
        
        self.deconv1 = nn.ConvTranspose2d(32+1, 16, 2, 2)
        self.conv1 = nn.Sequential(nn.Conv2d(16, 16, 3, stride=1, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(16, 1, 3, stride=1, padding=1))
        
        # ori
        self.deconv6_ori = nn.ConvTranspose2d(2048+16, 1024, 2, 2)
        self.conv6_ori = double_conv(1344, 512)
                                    
        self.deconv5_ori = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv5_ori = double_conv(368, 256)
        
        self.deconv4_ori = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv4_ori = double_conv(168, 128)
        
        self.deconv3_ori = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv3_ori = double_conv(88, 64)
        
        self.deconv2_ori = nn.ConvTranspose2d(64, 32, 2, 2)
        self.conv2_ori = double_conv(48, 32)
        
        self.deconv1_ori = nn.ConvTranspose2d(32, 16, 2, 2)
        self.conv1_ori = nn.Sequential(nn.Conv2d(16, 16, 3, stride=1, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(16, 2, 3, stride=1, padding=1))
        
        
    def forward(self, grd, sat):
        grd_feature_volume = self.grd_efficientnet.extract_features(grd) # [1280, 8, 32]
        grd_descriptor1 = self.grd_feature_to_descriptor1(grd_feature_volume) # length 512
        grd_descriptor2 = self.grd_feature_to_descriptor2(grd_feature_volume) # length 256
        grd_descriptor3 = self.grd_feature_to_descriptor3(grd_feature_volume) # length 128
        grd_descriptor4 = self.grd_feature_to_descriptor4(grd_feature_volume) # length 64
        grd_descriptor5 = self.grd_feature_to_descriptor5(grd_feature_volume) # length 32
        grd_descriptor6 = self.grd_feature_to_descriptor6(grd_feature_volume) # length 32
        
        grd_descriptor_map1 = grd_descriptor1.unsqueeze(2).unsqueeze(3).repeat(1, 1, 8, 8)
        grd_descriptor_map2 = grd_descriptor2.unsqueeze(2).unsqueeze(3).repeat(1, 1, 16, 16)
        grd_descriptor_map3 = grd_descriptor3.unsqueeze(2).unsqueeze(3).repeat(1, 1, 32, 32)
        grd_descriptor_map4 = grd_descriptor4.unsqueeze(2).unsqueeze(3).repeat(1, 1, 64, 64)
        grd_descriptor_map5 = grd_descriptor5.unsqueeze(2).unsqueeze(3).repeat(1, 1, 128, 128)
        grd_descriptor_map6 = grd_descriptor6.unsqueeze(2).unsqueeze(3).repeat(1, 1, 256, 256)
        
        sat_feature_volume, multiscale_sat = self.sat_efficientnet.extract_features_multiscale(sat)
        sat_feature_block0 = multiscale_sat[0] # [16, 256, 256]
        sat_feature_block2 = multiscale_sat[2] #[24, 128, 128]
        sat_feature_block4 = multiscale_sat[4] # [40, 64, 64]
        sat_feature_block10 = multiscale_sat[10] # [112, 32, 32]
        sat_feature_block15 = multiscale_sat[15] # [320, 16, 16]
        
        sat_row_chunks = torch.stack(list(torch.chunk(sat_feature_volume, 8, 2)), dim=-1) # dimension 4 is the number of row chunks (splitted in height dimension)
        for i, sat_row_chunk in enumerate(torch.unbind(sat_row_chunks, dim=-1), 0):
            sat_chunks = torch.stack(list(torch.chunk(sat_row_chunk, 8, 3)), dim=-1) # dimension 5 is the number of vertical chunks (splitted in width dimension)
            for j, sat_chunk in enumerate(torch.unbind(sat_chunks, dim=-1), 0):
                if j == 0:
                    sat_descriptor_row = self.sat_feature_to_descriptors(sat_chunk).unsqueeze(2).unsqueeze(3)
                else:
                    sat_descriptor_row = torch.cat((sat_descriptor_row, self.sat_feature_to_descriptors(sat_chunk).unsqueeze(2).unsqueeze(3)), 3)
            if i == 0:
                sat_descriptor_map = sat_descriptor_row
            else:
                sat_descriptor_map = torch.cat((sat_descriptor_map, sat_descriptor_row), 2)
                
        # matching 8*8
        grd_des_len = grd_descriptor1.size()[1]
        sat_des_len = sat_descriptor_map.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map1, p='fro', dim=1, keepdim=True)
        
        for i in range(16):
            sat_descriptor_map_rolled = torch.roll(sat_descriptor_map, shifts=-i*128, dims=1)
            sat_descriptor_map_window = sat_descriptor_map_rolled[:,:grd_des_len, :, :]
            sat_map_norm = torch.norm(sat_descriptor_map_window, p='fro', dim=1, keepdim=True)

            matching_score = torch.sum((grd_descriptor_map1*sat_descriptor_map_window), dim=1, keepdim=True) / (sat_map_norm * grd_map_norm) # cosine similarity
            if i == 0:
                matching_score_stacked = matching_score
            else:
                matching_score_stacked = torch.cat([matching_score_stacked, matching_score], dim=1)
        
        matching_score_max, _ = torch.max(matching_score_stacked, dim=1, keepdim=True)
        # loc
        x = torch.cat([matching_score_max, self.sat_normalization(sat_descriptor_map)], dim=1)

        x = self.deconv6(x)
        x = torch.cat([x, sat_feature_block15], dim=1)
        x = self.conv6(x)
        
        # matching 16*16
        grd_des_len = grd_descriptor2.size()[1]
        sat_des_len = x.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map2, p='fro', dim=1, keepdim=True)
        
        for i in range(16):
            sat_descriptor_map_rolled = torch.roll(x, shifts=-i*64, dims=1)
            sat_descriptor_map_window = sat_descriptor_map_rolled[:,:grd_des_len, :, :]
            sat_map_norm = torch.norm(sat_descriptor_map_window, p='fro', dim=1, keepdim=True)

            matching_score = torch.sum((grd_descriptor_map2*sat_descriptor_map_window), dim=1, keepdim=True) / (sat_map_norm * grd_map_norm) # cosine similarity
            if i == 0:
                matching_score_stacked2 = matching_score
            else:
                matching_score_stacked2 = torch.cat([matching_score_stacked2, matching_score], dim=1)
        
        matching_score_max, _ = torch.max(matching_score_stacked2, dim=1, keepdim=True)
        
        x = torch.cat([matching_score_max, self.sat_normalization(x)], dim=1)
        x = self.deconv5(x)
        x = torch.cat([x, sat_feature_block10], dim=1)
        x = self.conv5(x)
        
        # matching 32*32
        grd_des_len = grd_descriptor3.size()[1]
        sat_des_len = x.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map3, p='fro', dim=1, keepdim=True)
        
        for i in range(16):
            sat_descriptor_map_rolled = torch.roll(x, shifts=-i*32, dims=1)
            sat_descriptor_map_window = sat_descriptor_map_rolled[:,:grd_des_len, :, :]
            sat_map_norm = torch.norm(sat_descriptor_map_window, p='fro', dim=1, keepdim=True)

            matching_score = torch.sum((grd_descriptor_map3*sat_descriptor_map_window), dim=1, keepdim=True) / (sat_map_norm * grd_map_norm) # cosine similarity
            if i == 0:
                matching_score_stacked3 = matching_score
            else:
                matching_score_stacked3 = torch.cat([matching_score_stacked3, matching_score], dim=1)
        
        matching_score_max, _ = torch.max(matching_score_stacked3, dim=1, keepdim=True)
        
        x = torch.cat([matching_score_max, self.sat_normalization(x)], dim=1)
        x = self.deconv4(x)
        x = torch.cat([x, sat_feature_block4], dim=1)
        x = self.conv4(x)
        
        # matching 64*64
        grd_des_len = grd_descriptor4.size()[1]
        sat_des_len = x.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map4, p='fro', dim=1, keepdim=True)
        
        for i in range(16):
            sat_descriptor_map_rolled = torch.roll(x, shifts=-i*16, dims=1)
            sat_descriptor_map_window = sat_descriptor_map_rolled[:,:grd_des_len, :, :]
            sat_map_norm = torch.norm(sat_descriptor_map_window, p='fro', dim=1, keepdim=True)

            matching_score = torch.sum((grd_descriptor_map4*sat_descriptor_map_window), dim=1, keepdim=True) / (sat_map_norm * grd_map_norm) # cosine similarity
            if i == 0:
                matching_score_stacked4 = matching_score
            else:
                matching_score_stacked4 = torch.cat([matching_score_stacked4, matching_score], dim=1)
        
        matching_score_max, _ = torch.max(matching_score_stacked4, dim=1, keepdim=True)
        
        x = torch.cat([matching_score_max, self.sat_normalization(x)], dim=1)
        x = self.deconv3(x)
        x = torch.cat([x, sat_feature_block2], dim=1)
        x = self.conv3(x)
        
        # matching 128*128
        grd_des_len = grd_descriptor5.size()[1]
        sat_des_len = x.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map5, p='fro', dim=1, keepdim=True)
        
        for i in range(16):
            sat_descriptor_map_rolled = torch.roll(x, shifts=-i*8, dims=1)
            sat_descriptor_map_window = sat_descriptor_map_rolled[:,:grd_des_len, :, :]
            sat_map_norm = torch.norm(sat_descriptor_map_window, p='fro', dim=1, keepdim=True)

            matching_score = torch.sum((grd_descriptor_map5*sat_descriptor_map_window), dim=1, keepdim=True) / (sat_map_norm * grd_map_norm) # cosine similarity
            if i == 0:
                matching_score_stacked5 = matching_score
            else:
                matching_score_stacked5 = torch.cat([matching_score_stacked5, matching_score], dim=1)
        
        matching_score_max, _ = torch.max(matching_score_stacked5, dim=1, keepdim=True)
        
        x = torch.cat([matching_score_max, self.sat_normalization(x)], dim=1)
        x = self.deconv2(x)
        x = torch.cat([x, sat_feature_block0], dim=1)
        x = self.conv2(x)
        
        # matching 256*256
        grd_des_len = grd_descriptor6.size()[1]
        sat_des_len = x.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map6, p='fro', dim=1, keepdim=True)
        
        for i in range(16):
            sat_descriptor_map_rolled = torch.roll(x, shifts=-i*8, dims=1)
            sat_descriptor_map_window = sat_descriptor_map_rolled[:,:grd_des_len, :, :]
            sat_map_norm = torch.norm(sat_descriptor_map_window, p='fro', dim=1, keepdim=True)

            matching_score = torch.sum((grd_descriptor_map6*sat_descriptor_map_window), dim=1, keepdim=True) / (sat_map_norm * grd_map_norm) # cosine similarity
            if i == 0:
                matching_score_stacked6 = matching_score
            else:
                matching_score_stacked6 = torch.cat([matching_score_stacked6, matching_score], dim=1)
        
        matching_score_max, _ = torch.max(matching_score_stacked6, dim=1, keepdim=True)
        
        x = torch.cat([matching_score_max, self.sat_normalization(x)], dim=1)
        x = self.deconv1(x)
        x = self.conv1(x)
        
        logits_flattened = torch.flatten(x, start_dim=1)
        heatmap = torch.reshape(nn.Softmax(dim=-1)(logits_flattened), x.size())
        
        # ori
        x_ori = torch.cat([matching_score_stacked, self.sat_normalization(sat_descriptor_map)], dim=1)
        x_ori = self.deconv6_ori(x_ori)
        x_ori = torch.cat([x_ori, sat_feature_block15], dim=1)
        x_ori = self.conv6_ori(x_ori)
        x_ori = self.deconv5_ori(x_ori)
        x_ori = torch.cat([x_ori, sat_feature_block10], dim=1)
        x_ori = self.conv5_ori(x_ori)
        x_ori = self.deconv4_ori(x_ori)
        x_ori = torch.cat([x_ori, sat_feature_block4], dim=1)
        x_ori = self.conv4_ori(x_ori)
        x_ori = self.deconv3_ori(x_ori)
        x_ori = torch.cat([x_ori, sat_feature_block2], dim=1)
        x_ori = self.conv3_ori(x_ori)
        x_ori = self.deconv2_ori(x_ori)
        x_ori = torch.cat([x_ori, sat_feature_block0], dim=1)
        x_ori = self.conv2_ori(x_ori)
        x_ori = self.deconv1_ori(x_ori)
        x_ori = self.conv1_ori(x_ori)
        x_ori = nn.functional.normalize(x_ori, p=2, dim=1)
        
        return logits_flattened, heatmap, x_ori, matching_score_stacked, matching_score_stacked2, matching_score_stacked3, matching_score_stacked4, matching_score_stacked5, matching_score_stacked6

    
    
class CVM_OxfordRobotCar(nn.Module):
    def __init__(self, device):
        super(CVM_OxfordRobotCar, self).__init__()
        self.device = device
        
        self.grd_efficientnet = EfficientNet.from_pretrained('efficientnet-b0', False) # no horizontal circular padding

        self.grd_feature_to_descriptor1 = nn.Sequential(
                                    nn.Conv2d(1280, 32, 1),
                                    permute_channels(0, 2, 3, 1),
                                    nn.Conv2d(4, 1, 1),
                                    nn.Flatten(start_dim=1)
                                    )
        
        self.grd_feature_to_descriptor2 = nn.Sequential(
                                    nn.Conv2d(1280, 16, 1),
                                    permute_channels(0, 2, 3, 1),
                                    nn.Conv2d(4, 1, 1),
                                    nn.Flatten(start_dim=1)
                                    )
        
        self.grd_feature_to_descriptor3 = nn.Sequential(
                                    nn.Conv2d(1280, 8, 1),
                                    permute_channels(0, 2, 3, 1),
                                    nn.Conv2d(4, 1, 1),
                                    nn.Flatten(start_dim=1)
                                    )
        self.grd_feature_to_descriptor4 = nn.Sequential(
                                    nn.Conv2d(1280, 4, 1),
                                    permute_channels(0, 2, 3, 1),
                                    nn.Conv2d(4, 1, 1),
                                    nn.Flatten(start_dim=1)
                                    )
        self.grd_feature_to_descriptor5 = nn.Sequential(
                                    nn.Conv2d(1280, 2, 1),
                                    permute_channels(0, 2, 3, 1),
                                    nn.Conv2d(4, 1, 1),
                                    nn.Flatten(start_dim=1)
                                    )
        self.grd_feature_to_descriptor6 = nn.Sequential(
                                    nn.Conv2d(1280, 1, 1),
                                    permute_channels(0, 2, 3, 1),
                                    nn.Conv2d(4, 1, 1),
                                    nn.Flatten(start_dim=1)
                                    )
        
        self.sat_efficientnet = EfficientNet.from_pretrained('efficientnet-b0', False)
        
        
        self.sat_feature_to_descriptors = nn.Sequential(nn.Flatten(start_dim=1),
                                                        nn.Linear(1280*2*2, 1280)
                                                       )
        self.sat_normalization = normalization(2, 1)
        
        # loc
        self.deconv6 = nn.ConvTranspose2d(1281, 1024, 2, 2)
        self.conv6 = double_conv(1344, 640)
                                    
        self.deconv5 = nn.ConvTranspose2d(641, 320, 2, 2)
        self.conv5 = double_conv(432, 320)
        
        self.deconv4 = nn.ConvTranspose2d(321, 160, 2, 2)
        self.conv4 = double_conv(200, 160)
        
        self.deconv3 = nn.ConvTranspose2d(161, 80, 2, 2)
        self.conv3 = double_conv(104, 80)
        
        self.deconv2 = nn.ConvTranspose2d(81, 40, 2, 2)
        self.conv2 = double_conv(56, 40)
        
        self.deconv1 = nn.ConvTranspose2d(41, 16, 2, 2)
        self.conv1 = nn.Sequential(nn.Conv2d(16, 16, 3, stride=1, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(16, 1, 3, stride=1, padding=1))
        
        # ori
        self.deconv6_ori = nn.ConvTranspose2d(1300, 1024, 2, 2)
        self.conv6_ori = double_conv(1344, 640)
                                    
        self.deconv5_ori = nn.ConvTranspose2d(640, 256, 2, 2)
        self.conv5_ori = double_conv(368, 256)
        
        self.deconv4_ori = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv4_ori = double_conv(168, 128)
        
        self.deconv3_ori = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv3_ori = double_conv(88, 64)
        
        self.deconv2_ori = nn.ConvTranspose2d(64, 32, 2, 2)
        self.conv2_ori = double_conv(48, 32)
        
        self.deconv1_ori = nn.ConvTranspose2d(32, 16, 2, 2)
        self.conv1_ori = nn.Sequential(nn.Conv2d(16, 16, 3, stride=1, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(16, 2, 3, stride=1, padding=1))
        
        
    def forward(self, grd, sat):
        grd_feature_volume = self.grd_efficientnet.extract_features(grd)
        grd_descriptor1 = self.grd_feature_to_descriptor1(grd_feature_volume) # length 224
        grd_descriptor2 = self.grd_feature_to_descriptor2(grd_feature_volume) # length 112
        grd_descriptor3 = self.grd_feature_to_descriptor3(grd_feature_volume) # length 56
        grd_descriptor4 = self.grd_feature_to_descriptor4(grd_feature_volume) # length 28
        grd_descriptor5 = self.grd_feature_to_descriptor5(grd_feature_volume) # length 14
        grd_descriptor6 = self.grd_feature_to_descriptor6(grd_feature_volume) # length 7
        
        grd_descriptor_map1 = grd_descriptor1.unsqueeze(2).unsqueeze(3).repeat(1, 1, 8, 8)
        grd_descriptor_map2 = grd_descriptor2.unsqueeze(2).unsqueeze(3).repeat(1, 1, 16, 16)
        grd_descriptor_map3 = grd_descriptor3.unsqueeze(2).unsqueeze(3).repeat(1, 1, 32, 32)
        grd_descriptor_map4 = grd_descriptor4.unsqueeze(2).unsqueeze(3).repeat(1, 1, 64, 64)
        grd_descriptor_map5 = grd_descriptor5.unsqueeze(2).unsqueeze(3).repeat(1, 1, 128, 128)
        grd_descriptor_map6 = grd_descriptor6.unsqueeze(2).unsqueeze(3).repeat(1, 1, 256, 256)
       
        sat_feature_volume, multiscale_sat = self.sat_efficientnet.extract_features_multiscale(sat)
        sat_feature_block0 = multiscale_sat[0] # [16, 256, 256]
        sat_feature_block2 = multiscale_sat[2] #[24, 128, 128]
        sat_feature_block4 = multiscale_sat[4] # [40, 64, 64]
        sat_feature_block10 = multiscale_sat[10] # [112, 32, 32]
        sat_feature_block15 = multiscale_sat[15] # [320, 16, 16]
        
        sat_row_chunks = torch.stack(list(torch.chunk(sat_feature_volume, 8, 2)), dim=-1) # dimension 4 is the number of row chunks (splitted in height dimension)
        for i, sat_row_chunk in enumerate(torch.unbind(sat_row_chunks, dim=-1), 0):
            sat_chunks = torch.stack(list(torch.chunk(sat_row_chunk, 8, 3)), dim=-1) # dimension 5 is the number of vertical chunks (splitted in width dimension)
            for j, sat_chunk in enumerate(torch.unbind(sat_chunks, dim=-1), 0):
                if j == 0:
                    sat_descriptor_row = self.sat_feature_to_descriptors(sat_chunk).unsqueeze(2).unsqueeze(3)
                else:
                    sat_descriptor_row = torch.cat((sat_descriptor_row, self.sat_feature_to_descriptors(sat_chunk).unsqueeze(2).unsqueeze(3)), 3)
            if i == 0:
                sat_descriptor_map = sat_descriptor_row
            else:
                sat_descriptor_map = torch.cat((sat_descriptor_map, sat_descriptor_row), 2)
        
        # matching bottleneck
        grd_des_len = grd_descriptor1.size()[1]
        sat_des_len = sat_descriptor_map.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map1, p='fro', dim=1, keepdim=True)
        
        for i in range(20):
            sat_descriptor_map_rolled = torch.roll(sat_descriptor_map, shifts=-i*64, dims=1)
            sat_descriptor_map_window = sat_descriptor_map_rolled[:, int(sat_des_len/2-grd_des_len/2):int(sat_des_len/2+grd_des_len/2), :, :]
            sat_map_norm = torch.norm(sat_descriptor_map_window, p='fro', dim=1, keepdim=True)

            matching_score = torch.sum((grd_descriptor_map1*sat_descriptor_map_window), dim=1, keepdim=True) / (sat_map_norm * grd_map_norm) # cosine similarity
            if i == 0:
                matching_score_stacked = matching_score
            else:
                matching_score_stacked = torch.cat([matching_score_stacked, matching_score], dim=1)
        
        matching_score_max, _ = torch.max(matching_score_stacked, dim=1, keepdim=True)
       
        # loc
        x = torch.cat([matching_score_max, self.sat_normalization(sat_descriptor_map)], dim=1)

        x = self.deconv6(x)
        x = torch.cat([x, sat_feature_block15], dim=1)
        x = self.conv6(x)
                
        # matching 16*16
        grd_des_len = grd_descriptor2.size()[1] 
        sat_des_len = x.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map2, p='fro', dim=1, keepdim=True)
        
        for i in range(20):
            sat_descriptor_map_rolled = torch.roll(x, shifts=-i*32, dims=1)
            sat_descriptor_map_window = sat_descriptor_map_rolled[:, int(sat_des_len/2-grd_des_len/2):int(sat_des_len/2+grd_des_len/2), :, :]
            sat_map_norm = torch.norm(sat_descriptor_map_window, p='fro', dim=1, keepdim=True)

            matching_score = torch.sum((grd_descriptor_map2*sat_descriptor_map_window), dim=1, keepdim=True) / (sat_map_norm * grd_map_norm) # cosine similarity
            if i == 0:
                matching_score_stacked2 = matching_score
            else:
                matching_score_stacked2 = torch.cat([matching_score_stacked2, matching_score], dim=1)
        matching_score_max, _ = torch.max(matching_score_stacked2, dim=1, keepdim=True)
        
        x = torch.cat([matching_score_max, self.sat_normalization(x)], dim=1)
        x = self.deconv5(x)
        x = torch.cat([x, sat_feature_block10], dim=1)
        x = self.conv5(x)
        
        # matching 32*32
        grd_des_len = grd_descriptor3.size()[1] 
        sat_des_len = x.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map3, p='fro', dim=1, keepdim=True)
        
        for i in range(20):
            sat_descriptor_map_rolled = torch.roll(x, shifts=-i*16, dims=1)
            sat_descriptor_map_window = sat_descriptor_map_rolled[:, int(sat_des_len/2-grd_des_len/2):int(sat_des_len/2+grd_des_len/2), :, :]
            sat_map_norm = torch.norm(sat_descriptor_map_window, p='fro', dim=1, keepdim=True)

            matching_score = torch.sum((grd_descriptor_map3*sat_descriptor_map_window), dim=1, keepdim=True) / (sat_map_norm * grd_map_norm) # cosine similarity
            if i == 0:
                matching_score_stacked3 = matching_score
            else:
                matching_score_stacked3 = torch.cat([matching_score_stacked3, matching_score], dim=1)
        matching_score_max, _ = torch.max(matching_score_stacked3, dim=1, keepdim=True)
        
        x = torch.cat([matching_score_max, self.sat_normalization(x)], dim=1)
        x = self.deconv4(x)
        x = torch.cat([x, sat_feature_block4], dim=1)
        x = self.conv4(x)
        
        # matching 64*64
        grd_des_len = grd_descriptor4.size()[1] 
        sat_des_len = x.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map4, p='fro', dim=1, keepdim=True)
        
        for i in range(20):
            sat_descriptor_map_rolled = torch.roll(x, shifts=-i*8, dims=1)
            sat_descriptor_map_window = sat_descriptor_map_rolled[:, int(sat_des_len/2-grd_des_len/2):int(sat_des_len/2+grd_des_len/2), :, :]
            sat_map_norm = torch.norm(sat_descriptor_map_window, p='fro', dim=1, keepdim=True)

            matching_score = torch.sum((grd_descriptor_map4*sat_descriptor_map_window), dim=1, keepdim=True) / (sat_map_norm * grd_map_norm) # cosine similarity
            if i == 0:
                matching_score_stacked4 = matching_score
            else:
                matching_score_stacked4 = torch.cat([matching_score_stacked4, matching_score], dim=1)
        matching_score_max, _ = torch.max(matching_score_stacked4, dim=1, keepdim=True)
        
        x = torch.cat([matching_score_max, self.sat_normalization(x)], dim=1)
        x = self.deconv3(x)
        x = torch.cat([x, sat_feature_block2], dim=1)
        x = self.conv3(x)
        
        # matching 128*128
        grd_des_len = grd_descriptor5.size()[1] 
        sat_des_len = x.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map5, p='fro', dim=1, keepdim=True)
        
        for i in range(20):
            sat_descriptor_map_rolled = torch.roll(x, shifts=-i*4, dims=1)
            sat_descriptor_map_window = sat_descriptor_map_rolled[:, int(sat_des_len/2-grd_des_len/2):int(sat_des_len/2+grd_des_len/2), :, :]
            sat_map_norm = torch.norm(sat_descriptor_map_window, p='fro', dim=1, keepdim=True)

            matching_score = torch.sum((grd_descriptor_map5*sat_descriptor_map_window), dim=1, keepdim=True) / (sat_map_norm * grd_map_norm) # cosine similarity
            if i == 0:
                matching_score_stacked5 = matching_score
            else:
                matching_score_stacked5 = torch.cat([matching_score_stacked5, matching_score], dim=1)
        matching_score_max, _ = torch.max(matching_score_stacked5, dim=1, keepdim=True)
        
        x = torch.cat([matching_score_max, self.sat_normalization(x)], dim=1)
        x = self.deconv2(x)
        x = torch.cat([x, sat_feature_block0], dim=1)
        x = self.conv2(x)
        
        # matching 256*256
        grd_des_len = grd_descriptor6.size()[1] 
        sat_des_len = x.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map6, p='fro', dim=1, keepdim=True)
        
        for i in range(20):
            sat_descriptor_map_rolled = torch.roll(x, shifts=-i*2, dims=1)
            sat_descriptor_map_window = sat_descriptor_map_rolled[:, int(sat_des_len/2-grd_des_len/2):int(sat_des_len/2+grd_des_len/2), :, :]
            sat_map_norm = torch.norm(sat_descriptor_map_window, p='fro', dim=1, keepdim=True)

            matching_score = torch.sum((grd_descriptor_map6*sat_descriptor_map_window), dim=1, keepdim=True) / (sat_map_norm * grd_map_norm) # cosine similarity
            if i == 0:
                matching_score_stacked6 = matching_score
            else:
                matching_score_stacked6 = torch.cat([matching_score_stacked6, matching_score], dim=1)
        matching_score_max, _ = torch.max(matching_score_stacked6, dim=1, keepdim=True)
        x = torch.cat([matching_score_max, self.sat_normalization(x)], dim=1)
        x = self.deconv1(x)
        x = self.conv1(x)
        
        logits_flattened = torch.flatten(x, start_dim=1)
        heatmap = torch.reshape(nn.Softmax(dim=-1)(logits_flattened), x.size())
        
        # ori
        x_ori = torch.cat([matching_score_stacked, self.sat_normalization(sat_descriptor_map)], dim=1)
        x_ori = self.deconv6_ori(x_ori)
        x_ori = torch.cat([x_ori, sat_feature_block15], dim=1)
        x_ori = self.conv6_ori(x_ori)
        x_ori = self.deconv5_ori(x_ori)
        x_ori = torch.cat([x_ori, sat_feature_block10], dim=1)
        x_ori = self.conv5_ori(x_ori)
        x_ori = self.deconv4_ori(x_ori)
        x_ori = torch.cat([x_ori, sat_feature_block4], dim=1)
        x_ori = self.conv4_ori(x_ori)
        x_ori = self.deconv3_ori(x_ori)
        x_ori = torch.cat([x_ori, sat_feature_block2], dim=1)
        x_ori = self.conv3_ori(x_ori)
        x_ori = self.deconv2_ori(x_ori)
        x_ori = torch.cat([x_ori, sat_feature_block0], dim=1)
        x_ori = self.conv2_ori(x_ori)
        x_ori = self.deconv1_ori(x_ori)
        x_ori = self.conv1_ori(x_ori)
        x_ori = nn.functional.normalize(x_ori, p=2, dim=1)
        
        return logits_flattened, heatmap, x_ori, matching_score_stacked, matching_score_stacked2, matching_score_stacked3, matching_score_stacked4, matching_score_stacked5, matching_score_stacked6