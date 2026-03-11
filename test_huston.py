# 真实数据 无参考
import os
import time

import cv2
import hdf5storage
import numpy as np
import torch
import hdf5storage as h5
from utils import create_F, fspecial
from train_dataloader import *
from torch import nn
from tqdm import tqdm
import time
import pandas as pd
import torch.utils.data as data
from model.CLSNet_Houston import CLSNet as allnet

def reconstruction_huston(net2, R, HSI_LR, MSI_HR, downsample_factor,training_size, stride):
    index_matrix = torch.zeros((HSI_LR.shape[1], MSI_HR.shape[2], MSI_HR.shape[3])).cuda()
    abundance_t = torch.zeros((HSI_LR.shape[1], MSI_HR.shape[2], MSI_HR.shape[3])).cuda()
    a = []
    for j in range(0, MSI_HR.shape[2] - training_size + 1, stride):
        a.append(j)
    a.append(MSI_HR.shape[2] - training_size)
    b = []
    for j in range(0, MSI_HR.shape[3] - training_size + 1, stride):
        b.append(j)
    b.append(MSI_HR.shape[3] - training_size)
    for j in a:
        for k in b:
            temp_hrms = MSI_HR[:, :, j:j + training_size, k:k + training_size]
            temp_lrhs = HSI_LR[:, :, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                        int(k / downsample_factor):int((k + training_size) / downsample_factor)]
            with torch.no_grad():
                out = net2(temp_lrhs,temp_hrms)   # ssgt
                assert torch.isnan(out).sum() == 0

                HSI = out.squeeze()
                HSI = torch.clamp(HSI, 0, 1)
                abundance_t[:, j:j + training_size, k:k + training_size] = abundance_t[:, j:j + training_size,
                                                                           k:k + training_size] + HSI
                index_matrix[:, j:j + training_size, k:k + training_size] = 1 + index_matrix[:, j:j + training_size,
                                                                                k:k + training_size]

    HSI_recon = abundance_t / index_matrix
    assert torch.isnan(HSI_recon).sum() == 0
    return HSI_recon

data_name = 'huston'
HSI_ori = h5.loadmat(r"F:\data\Huston.mat")
HSI_ori = np.float32(HSI_ori["HSI"])
HSI_ori = HSI_ori / HSI_ori.max()


PSF = fspecial('gaussian', 9, 3)
HSI_ori = np.transpose(HSI_ori, (2, 0, 1))
grouped_channels = np.array_split(HSI_ori, 4, axis=0)
result_channels = [np.mean(group, axis=0) for group in grouped_channels]
MSI = np.stack(result_channels, axis=0)

R = create_F()
downsample_factor=8
C, W, H = HSI_ori.shape
dw = W // downsample_factor
dh = H // downsample_factor
HSI_ori = HSI_ori[:, :dw * downsample_factor, :dh * downsample_factor]
MSI = MSI[:, :dw * downsample_factor, :dh * downsample_factor]

HSI_D1 = Gaussian_downsample(HSI_ori, PSF, downsample_factor)
MSI_D1 = Gaussian_downsample(MSI, PSF, downsample_factor)

C, W, H = HSI_D1.shape
dw = W // downsample_factor
dh = H // downsample_factor
HSI_D11 = HSI_D1[:, :dw * downsample_factor, :dh * downsample_factor]

C, W, H = MSI_D1.shape
dw = W // downsample_factor
dh = H // downsample_factor
MSI_D2 = MSI_D1[:, :dw * downsample_factor, :dh * downsample_factor]
HSI_D2 = Gaussian_downsample(HSI_D11, PSF, downsample_factor)

test_HRMSI0 = MSI
test_HSI0 = HSI_D1
model_path = r"\hsi_fuse\fusion_code\fusion_code_2/train_save_Comparison/Our/9/huston_pkl/200EPOCH_PSNR_best.pkl"
net2=allnet().cuda()
checkpoint = torch.load(model_path)  # 加载断点
net2.load_state_dict(checkpoint)
save_path = r"F:\\paper\\HUSTON"

downsample_factor=8
training_size=32
stride=8

def mkdir(path):

    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("测试保存文件夹为：{}".format(path))
    else:
        print('已存在{}'.format(path))
mkdir(save_path)


with torch.no_grad():

    test_HRMSI = torch.unsqueeze(torch.Tensor(test_HRMSI0), 0)
    test_LRHSI = torch.unsqueeze(torch.Tensor(test_HSI0), 0)
    time1= time.time()
    prediction = reconstruction_huston(net2, R, test_LRHSI.cuda(), test_HRMSI.cuda(),
                                              downsample_factor, training_size, stride)
    Fuse = prediction.cpu().detach().numpy()
    time2 = time.time()
    print(time2 - time1)

faker_hyper = np.transpose(Fuse, (1, 2, 0))
test_data_path = os.path.join(save_path + "huston_200")
hdf5storage.savemat(test_data_path, {'fak': faker_hyper}, format='7.3')
