import os
import hdf5storage as h5
from torch.utils import data
from utils import *
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import scipy.io as scio
import torch
from torch.utils.data import Dataset
import pickle
import tqdm
from tqdm.contrib import tzip
import h5py
import hdf5storage as h5
from scipy.io import loadmat
from numpy.lib.stride_tricks import as_strided

class RealDATAProcess3(Dataset):
    def __init__(self, LR,msi,HR, training_size, stride, downsample_factor):
        """
        :param path:
        :param R: 光谱响应矩阵
        :param training_size:
        :param stride:
        :param downsample_factor:
        :param PSF: 高斯模糊核
        :param num: cave=32个场景
        """
        train_hrhs = []
        train_lrhs = []
        train_hrms = []

        HSI_LR = LR
        MSI = msi
        HRHSI=HR
        for j in range(0, HRHSI.shape[1] - training_size + 1, stride):
            for k in range(0, HRHSI.shape[2] - training_size + 1, stride):
                # if (j+training_size)>800 and k<400:
                #     pass
                # else:
                temp_hrhs = HRHSI[:, j:j + training_size, k:k + training_size]
                temp_hrms = MSI[:, j:j + training_size, k:k + training_size]

                temp_lrhs = HSI_LR[:, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                            int(k / downsample_factor):int((k + training_size) / downsample_factor)]
                # print(temp_hrhs.shape,temp_lrhs.shape,temp_hrms.shape)
                temp_hrhs=temp_hrhs.astype(np.float32)
                temp_lrhs=temp_lrhs.astype(np.float32)
                temp_hrms = temp_hrms.astype(np.float32)
                train_hrhs.append(temp_hrhs)
                train_lrhs.append(temp_lrhs)
                train_hrms.append(temp_hrms)

        train_hrhs = torch.Tensor(np.array(train_hrhs))
        train_lrhs = torch.Tensor(np.array(train_lrhs))
        train_hrms = torch.Tensor(np.array(train_hrms))

        # print(train_hrhs.shape, train_hrms.shape)
        self.train_hrhs_all = train_hrhs
        self.train_lrhs_all = train_lrhs
        self.train_hrms_all = train_hrms

    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_lrhs = self.train_lrhs_all[index, :, :, :]
        train_hrms = self.train_hrms_all[index, :, :, :]
        # print(train_hrhs.shape, train_hrms.sha
        return train_hrhs, train_hrms, train_lrhs

    def __len__(self):
        return self.train_hrhs_all.shape[0]


class RealDATAProcess2(Dataset):
    def __init__(self, hsi,msi, training_size, stride, downsample_factor, PSF):
        """
        :param path:
        :param R: 光谱响应矩阵
        :param training_size:
        :param stride:
        :param downsample_factor:
        :param PSF: 高斯模糊核
        :param num: cave=32个场景
        """
        train_hrhs = []
        train_lrhs = []
        train_hrms = []

        # hwc-chw
        HRHSI = np.transpose(hsi, (2, 0, 1))
        msi = np.transpose(msi, (2, 0, 1))

        HSI_LR = Gaussian_downsample(HRHSI, PSF, downsample_factor)
        MSI = Gaussian_downsample(msi, PSF, downsample_factor)

        for j in range(0, HRHSI.shape[1] - training_size + 1, stride):
            for k in range(0, HRHSI.shape[2] - training_size + 1, stride):
                # if (j+training_size)>800 and k<400:
                #     pass
                # else:
                temp_hrhs = HRHSI[:, j:j + training_size, k:k + training_size]
                temp_hrms = MSI[:, j:j + training_size, k:k + training_size]

                temp_lrhs = HSI_LR[:, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                            int(k / downsample_factor):int((k + training_size) / downsample_factor)]
                # print(temp_hrhs.shape,temp_lrhs.shape,temp_hrms.shape)
                temp_hrhs=temp_hrhs.astype(np.float32)
                temp_lrhs=temp_lrhs.astype(np.float32)
                temp_hrms = temp_hrms.astype(np.float32)
                train_hrhs.append(temp_hrhs)
                train_lrhs.append(temp_lrhs)
                train_hrms.append(temp_hrms)

        train_hrhs = torch.Tensor(np.array(train_hrhs))
        train_lrhs = torch.Tensor(np.array(train_lrhs))
        train_hrms = torch.Tensor(np.array(train_hrms))

        # print(train_hrhs.shape, train_hrms.shape)
        self.train_hrhs_all = train_hrhs
        self.train_lrhs_all = train_lrhs
        self.train_hrms_all = train_hrms

    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_lrhs = self.train_lrhs_all[index, :, :, :]
        train_hrms = self.train_hrms_all[index, :, :, :]
        # print(train_hrhs.shape, train_hrms.shape,train_lrhs.shape)
        return train_hrhs, train_hrms, train_lrhs

    def __len__(self):
        return self.train_hrhs_all.shape[0]

class RealDATAProcess(Dataset):
    def __init__(self, hsi,msi, training_size, stride, downsample_factor, PSF):
        """
        :param path:
        :param R: 光谱响应矩阵
        :param training_size:
        :param stride:
        :param downsample_factor:
        :param PSF: 高斯模糊核
        :param num: cave=32个场景
        """
        train_hrhs = []
        train_lrhs = []
        train_hrms = []

        HRHSI=hsi
        # hwc-chw
        HSI_LR = Gaussian_downsample(hsi, PSF, downsample_factor)
        MSI = Gaussian_downsample(msi, PSF, downsample_factor)

        for j in range(0, HRHSI.shape[1] - training_size + 1, stride):
            for k in range(0, HRHSI.shape[2] - training_size + 1, stride):
                temp_hrhs = HRHSI[:, j:j + training_size, k:k + training_size]
                temp_hrms = MSI[:, j:j + training_size, k:k + training_size]

                temp_lrhs = HSI_LR[:, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                            int(k / downsample_factor):int((k + training_size) / downsample_factor)]
                # print(temp_hrhs.shape,temp_lrhs.shape,temp_hrms.shape)
                temp_hrhs=temp_hrhs.astype(np.float32)
                temp_lrhs=temp_lrhs.astype(np.float32)
                temp_hrms = temp_hrms.astype(np.float32)
                train_hrhs.append(temp_hrhs)
                train_lrhs.append(temp_lrhs)
                train_hrms.append(temp_hrms)

        train_hrhs = torch.Tensor(np.array(train_hrhs))
        train_lrhs = torch.Tensor(np.array(train_lrhs))
        train_hrms = torch.Tensor(np.array(train_hrms))

        # print(train_hrhs.shape, train_hrms.shape)
        self.train_hrhs_all = train_hrhs
        self.train_lrhs_all = train_lrhs
        self.train_hrms_all = train_hrms

    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_lrhs = self.train_lrhs_all[index, :, :, :]
        train_hrms = self.train_hrms_all[index, :, :, :]
        # print(train_hrhs.shape, train_hrms.shape,train_lrhs.shape)
        return train_hrhs, train_hrms, train_lrhs

    def __len__(self):
        return self.train_hrhs_all.shape[0]



class CAVEHSIDATAprocess(Dataset):
    def __init__(self, path, R, training_size, stride, downsample_factor, PSF, num):
        """
        :param path:
        :param R: 光谱响应矩阵
        :param training_size:
        :param stride:
        :param downsample_factor:
        :param PSF: 高斯模糊核
        :param num: cave=32个场景
        """
        imglist = os.listdir(path)
        train_hrhs = []
        train_hrms = []
        train_lrhs = []

        for i in range(num):
            img = h5.loadmat(path + imglist[i])
            img1 = img["b"]

            HRHSI = np.transpose(img1, (2, 0, 1))
            # hwc-chw
            HSI_LR = Gaussian_downsample(HRHSI, PSF, downsample_factor)
            MSI = np.tensordot(R, HRHSI, axes=([1], [0]))
            for j in range(0, HRHSI.shape[1] - training_size + 1, stride):
                for k in range(0, HRHSI.shape[2] - training_size + 1, stride):
                    temp_hrhs = HRHSI[:, j:j + training_size, k:k + training_size]
                    temp_hrms = MSI[:, j:j + training_size, k:k + training_size]
                    temp_lrhs = HSI_LR[:, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                                int(k / downsample_factor):int((k + training_size) / downsample_factor)]
                    # print(temp_hrhs.shape,temp_lrhs.shape,temp_hrms.shape)
                    temp_hrhs=temp_hrhs.astype(np.float32)
                    temp_hrms=temp_hrms.astype(np.float32)
                    temp_lrhs=temp_lrhs.astype(np.float32)
                    train_hrhs.append(temp_hrhs)
                    train_hrms.append(temp_hrms)
                    train_lrhs.append(temp_lrhs)
        train_hrhs = torch.Tensor(np.array(train_hrhs))
        train_hrms = torch.Tensor(np.array(train_hrms))
        train_lrhs = torch.Tensor(np.array(train_lrhs))

        # print(train_hrhs.shape, train_hrms.shape)
        self.train_hrhs_all = train_hrhs
        self.train_hrms_all = train_hrms
        self.train_lrhs_all = train_lrhs

    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_hrms = self.train_hrms_all[index, :, :, :]
        train_lrhs = self.train_lrhs_all[index, :, :, :]
        # print(train_hrhs.shape, train_hrms.shape,train_lrhs.shape)
        return train_hrhs, train_hrms, train_lrhs

    def __len__(self):
        return self.train_hrhs_all.shape[0]


class HarvardHSIDATAprocess(Dataset):
    def __init__(self, path, R, training_size, stride, downsample_factor, PSF, num):
        """
        :param path:
        :param R: 光谱响应矩阵
        :param training_size:
        :param stride:
        :param downsample_factor:
        :param PSF: 高斯模糊核
        :param num: cave=32个场景
        """
        imglist = os.listdir(path)
        train_hrhs = []
        train_hrms = []
        train_lrhs = []

        for i in range(num):
            img = h5.loadmat(path + imglist[i])
            img1 = img["ref"]
            img1 = img1 / img1.max()

            HRHSI = np.transpose(img1, (2, 0, 1))
            # hwc-chw
            HSI_LR = Gaussian_downsample(HRHSI, PSF, downsample_factor)
            MSI = np.tensordot(R, HRHSI, axes=([1], [0]))
            for j in range(0, HRHSI.shape[1] - training_size + 1, stride):
                for k in range(0, HRHSI.shape[2] - training_size + 1, stride):
                    temp_hrhs = HRHSI[:, j:j + training_size, k:k + training_size]
                    temp_hrms = MSI[:, j:j + training_size, k:k + training_size]
                    temp_lrhs = HSI_LR[:, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                                int(k / downsample_factor):int((k + training_size) / downsample_factor)]
                    # print(temp_hrhs.shape,temp_lrhs.shape,temp_hrms.shape)
                    temp_hrhs=temp_hrhs.astype(np.float32)
                    temp_hrms=temp_hrms.astype(np.float32)
                    temp_lrhs=temp_lrhs.astype(np.float32)
                    train_hrhs.append(temp_hrhs)
                    train_hrms.append(temp_hrms)
                    train_lrhs.append(temp_lrhs)
        train_hrhs = torch.Tensor(np.array(train_hrhs))
        train_hrms = torch.Tensor(np.array(train_hrms))
        train_lrhs = torch.Tensor(np.array(train_lrhs))

        # print(train_hrhs.shape, train_hrms.shape)
        self.train_hrhs_all = train_hrhs
        self.train_hrms_all = train_hrms
        self.train_lrhs_all = train_lrhs

    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_hrms = self.train_hrms_all[index, :, :, :]
        train_lrhs = self.train_lrhs_all[index, :, :, :]
        # print(train_hrhs.shape, train_hrms.shape,train_lrhs.shape)
        return train_hrhs, train_hrms, train_lrhs

    def __len__(self):
        return self.train_hrhs_all.shape[0]
    

class KAISTHSIDATAprocess(Dataset):
    def __init__(self, path, R, training_size, stride, downsample_factor, PSF, num):
        """
        :param path:
        :param R: 光谱响应矩阵
        :param training_size:
        :param stride:
        :param downsample_factor:
        :param PSF: 高斯模糊核
        :param num: cave=32个场景
        """
        imglist = os.listdir(path)
        train_hrhs = []
        train_hrms = []
        train_lrhs = []

        for i in range(num):
            img = h5.loadmat(path + imglist[i])
            img1 = img["HSI"]
            img1 = img1 / img1.max()

            HRHSI = np.transpose(img1, (2, 0, 1))
            # hwc-chw
            HSI_LR = Gaussian_downsample(HRHSI, PSF, downsample_factor)
            MSI = np.tensordot(R, HRHSI, axes=([1], [0]))
            for j in range(0, HRHSI.shape[1] - training_size + 1, stride):
                for k in range(0, HRHSI.shape[2] - training_size + 1, stride):
                    temp_hrhs = HRHSI[:, j:j + training_size, k:k + training_size]
                    temp_hrms = MSI[:, j:j + training_size, k:k + training_size]
                    temp_lrhs = HSI_LR[:, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                                int(k / downsample_factor):int((k + training_size) / downsample_factor)]
                    # print(temp_hrhs.shape,temp_lrhs.shape,temp_hrms.shape)
                    temp_hrhs=temp_hrhs.astype(np.float32)
                    temp_hrms=temp_hrms.astype(np.float32)
                    temp_lrhs=temp_lrhs.astype(np.float32)
                    train_hrhs.append(temp_hrhs)
                    train_hrms.append(temp_hrms)
                    train_lrhs.append(temp_lrhs)
        train_hrhs = torch.Tensor(np.array(train_hrhs))
        train_hrms = torch.Tensor(np.array(train_hrms))
        train_lrhs = torch.Tensor(np.array(train_lrhs))

        # print(train_hrhs.shape, train_hrms.shape)
        self.train_hrhs_all = train_hrhs
        self.train_hrms_all = train_hrms
        self.train_lrhs_all = train_lrhs

    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_hrms = self.train_hrms_all[index, :, :, :]
        train_lrhs = self.train_lrhs_all[index, :, :, :]
        # print(train_hrhs.shape, train_hrms.shape,train_lrhs.shape)
        return train_hrhs, train_hrms, train_lrhs

    def __len__(self):
        return self.train_hrhs_all.shape[0]
    





class MakeSimulateDataset(Dataset):
    def __init__(self, args, type="train"):
        cache_path = os.path.join(args.cache_path, type + "_cache.pkl")
        PSF = fspecial('gaussian', 8, 3)
        R = create_F()

        if not os.path.exists(cache_path):
            lrhs_list, hrms_list, hrhs_list = [], [], []
            base_path = os.path.join(args.data_path, args.dataset, type)
            print("Cache file not found. Generating from:", base_path)
            img_list = os.listdir(base_path)
 
            for img_name in tqdm.tqdm(img_list):
                # ====== 加载 HRHSI ======
                if args.dataset == "CAVE":
                    hrhs_full = loadmat(os.path.join(base_path, img_name))["b"]
                    HRHSI = hrhs_full.transpose(2,0,1)  # (C,H,W)
                elif args.dataset == "ICVL":
                    hrhs_full = h5py.File(os.path.join(base_path, img_name))["rad"][:]
                    # HRHSI = np.rot90(hrhs_full.transpose(2,1,0))
                    HRHSI = HRHSI.transpose(2,0,1)
                    HRHSI = HRHSI / HRHSI.max()
                elif args.dataset == "HARVARD":
                    hrhs_full = h5.loadmat(os.path.join(base_path, img_name))["ref"][:]
                    HRHSI = hrhs_full.transpose(2,0,1)
                    HRHSI = HRHSI / HRHSI.max()
                elif args.dataset == "KAIST":
                    hrhs_full = h5.loadmat(os.path.join(base_path, img_name))["HSI"][:]
                    HRHSI = hrhs_full.transpose(2,0,1)
                    HRHSI = HRHSI / HRHSI.max()

                # ====== 构造 LRHSI 和 HRMS ======
                HSI_LR = Gaussian_downsample(HRHSI, PSF, args.spatial_ratio)  # (C,h,w)
                MSI = np.tensordot(R, HRHSI, axes=([1],[0]))                  # (M,H,W)

                if type == "train":
                    for j in range(0, HRHSI.shape[1] - args.train_size + 1, args.stride):
                        for k in range(0, HRHSI.shape[2] - args.train_size + 1, args.stride):
                            temp_lrhs = HSI_LR[:, int(j/args.spatial_ratio):int((j+args.train_size)/args.spatial_ratio),
                                                  int(k/args.spatial_ratio):int((k+args.train_size)/args.spatial_ratio)]
                            temp_hrms = MSI[:, j:j+args.train_size, k:k+args.train_size]
                            temp_hrhs = HRHSI[:, j:j+args.train_size, k:k+args.train_size]

                            lrhs_list.append(temp_lrhs.astype(np.float32))
                            hrms_list.append(temp_hrms.astype(np.float32))
                            hrhs_list.append(temp_hrhs.astype(np.float32))

                elif type == "test":
                    lrhs_list.append(HSI_LR.astype(np.float32))
                    hrms_list.append(MSI.astype(np.float32))
                    hrhs_list.append(HRHSI.astype(np.float32))

            # ====== 保存缓存 ======
            with open(cache_path, "wb") as f:
                pickle.dump([lrhs_list, hrms_list, hrhs_list], f)

        print("Load data from cache file:", cache_path)
        with open(cache_path, "rb") as f:
                 lrhs_list, hrms_list, hrhs_list = pickle.load(f)

        self.lrhs = torch.tensor(np.array(lrhs_list))
        self.hrms = torch.tensor(np.array(hrms_list))
        self.hrhs = torch.tensor(np.array(hrhs_list))
        
    def __len__(self):
        return self.hrhs.shape[0]

    def __getitem__(self, index):
        return self.lrhs[index], self.hrms[index], self.hrhs[index]


# 名称	尺寸 (C,H,W)	说明
# HSI_ori	(150, 1100, 1144)	原始 HR-HSI
# MSI_ori	(4, 2200, 2288)	原始 HR-MSI
# HSI_D1	(150, 550, 572)	一次下采样 HSI
# MSI_D1	(4, 1100, 1144)	一次下采样 MSI
# HSI_D11	(150, 550, 572)	裁剪后，仍是 HSI_D1 尺寸
# MSI_D11	(4, 1100, 1144)	裁剪后，仍是 MSI_D1 尺寸
# HSI_D2	(150, 275, 286)	二次下采样 HSI
# MSI_D2	(4, 550, 572)	二次下采样 MSI


class MakeSimulateDataset_GF5(Dataset):
    def __init__(self, args, type="train"):
        """
        必需的 args 字段：
          - data_path: 包含 reg_msi.npy, reg_pan.npy, R.npy, C.npy 的目录
          - cache_path: 缓存目录
          - train_size, stride, spatial_ratio
          - （可选）rebuild_cache: True 时强制重建缓存
        """
        cache_path = os.path.join(args.cache_path, type + "_cache.pkl")

        if not os.path.exists(cache_path):
            lrhs_list, hrms_list, hrhs_list = [], [], []
            base_path = os.path.join(args.data_path, args.dataset)
            print("Cache file not found. Generating from:", base_path) 

            # ====== 载入 GF5 原始数据 ======
            HSI_ori = np.load(os.path.join(base_path, "reg_msi.npy"))   # (H,W,C)
            MSI_ori = np.load(os.path.join(base_path, "reg_pan.npy"))   # (H,W,C)
            R = np.load(os.path.join(base_path, "R.npy"))               # (Cmsi, Chsi)
            Cpsf = np.load(os.path.join(base_path, "C.npy"))            # PSF核
            PSF = Cpsf

            # 维度统一到 (C,H,W)
            HSI_ori = HSI_ori.transpose(2, 0, 1).astype(np.float32)
            MSI_ori = MSI_ori.transpose(2, 0, 1).astype(np.float32)
            R = R.transpose(1, 0).astype(np.float32)  # (M, C)

            # 归一化 R 每一行
            row_sum = R.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1.0
            R = R / row_sum

            # 让尺寸能被 spatial_ratio 整除
            C, H, W = HSI_ori.shape
            C1, H1, W1 = MSI_ori.shape
            h_div = H // args.spatial_ratio
            w_div = W // args.spatial_ratio
            h1_div = H1 // args.spatial_ratio
            w1_div = W1 // args.spatial_ratio
            HSI_ori = HSI_ori[:, :h_div*args.spatial_ratio, :w_div*args.spatial_ratio]
            MSI_ori = MSI_ori[:, :h1_div*args.spatial_ratio, :w1_div*args.spatial_ratio]

            # 一次下采样（与测试输入对齐）
            HSI_D1 = Gaussian_downsample(HSI_ori, PSF, args.spatial_ratio).astype(np.float32)
            MSI_D1 = Gaussian_downsample(MSI_ori, PSF, args.spatial_ratio).astype(np.float32)

            # 训练还需要再做一次配准到更低分辨率（保证 LRHSI 与 HR 对应）
            if type == "train":
                # 按一次下采样后的尺寸再裁到可整除
                C3, H3, W3 = HSI_D1.shape
                C4, H4, W4 = MSI_D1.shape
                h3_div = H3 // args.spatial_ratio
                w3_div = W3 // args.spatial_ratio
                h4_div = H4 // args.spatial_ratio
                w4_div = W4 // args.spatial_ratio
                HSI_D11 = HSI_D1[:, :h3_div*args.spatial_ratio, :w3_div*args.spatial_ratio]
                MSI_D11 = MSI_D1[:, :h4_div*args.spatial_ratio, :w4_div*args.spatial_ratio]

                # 再次下采样，得到训练用的 LRHSI / HRMSI 对齐
                HSI_D2  = Gaussian_downsample(HSI_D11, PSF, args.spatial_ratio).astype(np.float32)
                MSI_D2  = Gaussian_downsample(MSI_D11, PSF, args.spatial_ratio).astype(np.float32)

                # 按 HSI_D11（HR 网格）切 patch；LR 从 HSI_D2 对应区域取
                for j in range(0, HSI_D11.shape[1] - args.train_size + 1, args.stride):
                    for k in range(0, HSI_D11.shape[2] - args.train_size + 1, args.stride):
                        temp_hrhs = HSI_D11[:, j:j+args.train_size, k:k+args.train_size]
                        temp_hrms = MSI_D2[:, j:j+args.train_size, k:k+args.train_size]
                        temp_lrhs = HSI_D2[:, 
                                           j//args.spatial_ratio:(j+args.train_size)//args.spatial_ratio,
                                           k//args.spatial_ratio:(k+args.train_size)//args.spatial_ratio]

                        lrhs_list.append(temp_lrhs.astype(np.float32))
                        hrms_list.append(temp_hrms.astype(np.float32))
                        hrhs_list.append(temp_hrhs.astype(np.float32))

            elif type == "test":
                # —— 这三者必须与你的测试代码一致 ——
                # LRHSI = 一次下采样的 HSI_D1
                # HRMSI = 一次下采样的 MSI_D1
                # HRHSI = 原始分辨率的 HSI_ori（用于评价/val_loss）
                lrhs_list.append(HSI_D1.astype(np.float32))
                hrms_list.append(MSI_D1.astype(np.float32))
                hrhs_list.append(HSI_ori.astype(np.float32))

            elif type == "test_real":
                lrhs_list.append(HSI_ori.astype(np.float32))
                hrms_list.append(MSI_ori.astype(np.float32))
                # 用零张量占位，保持结构一致
                hrhs_list.append(np.zeros_like(HSI_ori, dtype=np.float32))

            with open(cache_path, "wb") as f:
                pickle.dump([lrhs_list, hrms_list, hrhs_list], f)

        # ====== 读取缓存 ======
        print("Load data from cache file:", cache_path)
        with open(cache_path, "rb") as f:
            lrhs_list, hrms_list, hrhs_list = pickle.load(f)

        # 注意：测试时 HR 和 LR 空间尺寸不同，所以测试请 batch_size=1
        self.lrhs = torch.tensor(np.array(lrhs_list))
        self.hrms = torch.tensor(np.array(hrms_list))
        self.hrhs = torch.tensor(np.array(hrhs_list))

    def __len__(self):
        return self.hrhs.shape[0]

    def __getitem__(self, index):
        return self.lrhs[index], self.hrms[index], self.hrhs[index]

