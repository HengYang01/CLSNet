import torch
import os
os.environ['FOR_IGNORE_EXCEPTIONS'] = '1'

from scipy import signal
from numpy import *
from torch import nn
import numpy as np
import torch.nn.functional as F

def fspecial(func_name, kernel_size, sigma):
    if func_name == 'gaussian':
        m = n = (kernel_size - 1.) / 2.
        y, x = ogrid[-m:m + 1, -n:n + 1]
        h = exp(-(x * x + y * y) / (2. * sigma * sigma))
        h[h < finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

def Gaussian_downsample(x, psf, s):
    if x.ndim == 2:
        x = np.expand_dims(x, axis=0)
    y = np.zeros((x.shape[0], int(x.shape[1] / s), int(x.shape[2] / s)))
    for i in range(x.shape[0]):
        x1 = x[i, :, :]
        x2 = signal.convolve2d(x1, psf, boundary='symm', mode='same')
        y[i, :, :] = x2[0::s, 0::s]
    return y


def create_F():
    F = np.array(
        [[2.0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
         [1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 11, 16, 19, 21, 20, 18, 16, 14, 11, 7, 5, 3, 2, 2, 1, 1, 2, 2, 2, 2, 2],
         [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16, 9, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    for band in range(3):
        div = np.sum(F[band][:])
        for i in range(31):
            F[band][i] = F[band][i] / div
    return F


def warm_lr_scheduler(optimizer, init_lr1, init_lr2,min, warm_iter, iteraion, lr_decay_iter, max_iter, power):
    if iteraion % lr_decay_iter or iteraion > max_iter:
        return optimizer
    if iteraion < warm_iter:
        lr = init_lr1 + iteraion / warm_iter * (init_lr2 - init_lr1)
    else:
        lr = init_lr2 * (1 - (iteraion - warm_iter) / (max_iter - warm_iter)) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if lr<min:
        lr=min
    return lr

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


loss_func = nn.L1Loss(reduction='mean').cuda()


def reconstruction_fast(net2, R, HSI_LR, MSI, downsample_factor, training_size, stride):
    # 初始化
    index_matrix = torch.zeros((R.shape[1], MSI.shape[2], MSI.shape[3]), device=MSI.device)
    abundance_t = torch.zeros((R.shape[1], MSI.shape[2], MSI.shape[3]), device=MSI.device)

    # 预先生成滑动窗口起点索引，避免多次 append
    a = list(range(0, MSI.shape[2] - training_size + 1, stride))
    b = list(range(0, MSI.shape[3] - training_size + 1, stride))
    a.append(MSI.shape[2] - training_size)
    b.append(MSI.shape[3] - training_size)

    # 遍历 patch
    with torch.no_grad():
        for j in a:
            j_lr_start = j // downsample_factor
            j_lr_end   = (j + training_size) // downsample_factor
            for k in b:
                k_lr_start = k // downsample_factor
                k_lr_end   = (k + training_size) // downsample_factor

                # 提取 HRMS patch / LRHS patch
                temp_hrms = MSI[:, :, j:j + training_size, k:k + training_size]
                temp_lrhs = HSI_LR[:, :, j_lr_start:j_lr_end, k_lr_start:k_lr_end]

                # 网络推理
                out = net2(temp_lrhs, temp_hrms)
                assert torch.isnan(out).sum() == 0

                # 去掉多余维度，并限制范围 [0,1]
                HSI = torch.clamp(out.squeeze(), 0, 1)

                # 累加到结果矩阵
                abundance_t[:, j:j + training_size, k:k + training_size] += HSI
                index_matrix[:, j:j + training_size, k:k + training_size] += 1

    # 归一化（避免除零）
    HSI_recon = abundance_t / index_matrix
    assert torch.isnan(HSI_recon).sum() == 0

    return HSI_recon

def reconstruction(net2, HSI_LR, MSI, downsample_factor, training_size, stride, batch_size):
    """s
    net2: 融合网络，支持 batch 输入
    R: 光谱响应矩阵 (不变)
    HSI_LR, MSI, HRHSI: 输入张量
    downsample_factor: 下采样比例
    training_size: patch 尺寸
    stride: 步长
    batch_size: 每次 forward 的 patch 数量 (避免显存溢出)
    """
    C, H, W = HSI_LR.shape[1], MSI.shape[2], MSI.shape[3]

    index_matrix = torch.zeros((C, H, W), device=MSI.device)
    abundance_t = torch.zeros((C, H, W), device=MSI.device)

    # ===== 生成 patch 坐标 =====
    a = list(range(0, H - training_size + 1, stride))
    b = list(range(0, W - training_size + 1, stride))
    a.append(H - training_size)
    b.append(W - training_size)
    coords = [(j, k) for j in a for k in b]

    # ===== 动态分批处理 =====
    for start in range(0, len(coords), batch_size):
        batch_coords = coords[start:start+batch_size]

        lrhs_batch, hrms_batch = [], []
        for (j, k) in batch_coords:
            temp_hrms = MSI[:, :, j:j + training_size, k:k + training_size]
            temp_lrhs = HSI_LR[:, :,
                j // downsample_factor:(j + training_size) // downsample_factor,
                k // downsample_factor:(k + training_size) // downsample_factor]
            lrhs_batch.append(temp_lrhs)
            hrms_batch.append(temp_hrms)

        lrhs_batch = torch.cat(lrhs_batch, dim=0)  # (N, C, h, w)
        hrms_batch = torch.cat(hrms_batch, dim=0)  # (N, C, h, w)

        # ===== 一次 forward =====
        with torch.no_grad():
            out_batch = net2(lrhs_batch, hrms_batch)  # (N, C, h, w)
            assert torch.isnan(out_batch).sum() == 0
            out_batch = torch.clamp(out_batch, 0, 1)

        # ===== 拼回完整图像 =====
        for idx, (j, k) in enumerate(batch_coords):
            HSI = out_batch[idx]
            abundance_t[:, j:j + training_size, k:k + training_size] += HSI
            index_matrix[:, j:j + training_size, k:k + training_size] += 1

    HSI_recon = abundance_t / index_matrix
    assert torch.isnan(HSI_recon).sum() == 0
    return HSI_recon


def reconstruction_fg5(net2, R, HSI_LR, MSI_HR,HSI_HR, downsample_factor,training_size, stride,val_loss):
    index_matrix = torch.zeros((R.shape[1], MSI_HR.shape[2], MSI_HR.shape[3])).cuda()
    abundance_t = torch.zeros((R.shape[1], MSI_HR.shape[2], MSI_HR.shape[3])).cuda()
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
            temp_hrhs = HSI_HR[:, :, j:j + training_size, k:k + training_size]
            with torch.no_grad():
                # out = net2(temp_hrms,temp_lrhs)   # ssgt
                # out,ss1,ss2 = net2(temp_lrhs,temp_hrms)   # Fuformer
                out = net2(temp_lrhs,temp_hrms)  # hsrnet
                # out, out_spat, out_spec, edge_spat1, edge_spat2, edge_spec = net2(temp_lrhs, temp_hrms)   # ssrnet
                assert torch.isnan(out).sum() == 0

                loss_temp = loss_func(out, temp_hrhs.cuda())
                val_loss.update(loss_temp)
                HSI = out.squeeze()
                # 去掉维数为一的维度
                HSI = torch.clamp(HSI, 0, 1)
                abundance_t[:, j:j + training_size, k:k + training_size] = abundance_t[:, j:j + training_size,
                                                                           k:k + training_size] + HSI
                index_matrix[:, j:j + training_size, k:k + training_size] = 1 + index_matrix[:, j:j + training_size,
                                                                                k:k + training_size]

    HSI_recon = abundance_t / index_matrix
    assert torch.isnan(HSI_recon).sum() == 0
    return HSI_recon,val_loss

def reconstruction_huston(net2, R, HSI_LR, MSI_HR,HSI_HR, downsample_factor,training_size, stride,val_loss):
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
            temp_hrhs = HSI_HR[:, :, j:j + training_size, k:k + training_size]
            with torch.no_grad():
                # out = net2(temp_hrms,temp_lrhs)   # ssgt
                # out,ss1,ss2 = net2(temp_lrhs,temp_hrms)   # Fuformer
                out = net2(temp_lrhs,temp_hrms)  # hsrnet
                # out, out_spat, out_spec, edge_spat1, edge_spat2, edge_spec = net2(temp_lrhs, temp_hrms)   # ssrnet
                assert torch.isnan(out).sum() == 0

                loss_temp = loss_func(out, temp_hrhs.cuda())
                val_loss.update(loss_temp)
                HSI = out.squeeze()
                # 去掉维数为一的维度
                HSI = torch.clamp(HSI, 0, 1)
                abundance_t[:, j:j + training_size, k:k + training_size] = abundance_t[:, j:j + training_size,
                                                                           k:k + training_size] + HSI
                index_matrix[:, j:j + training_size, k:k + training_size] = 1 + index_matrix[:, j:j + training_size,
                                                                                k:k + training_size]

    HSI_recon = abundance_t / index_matrix
    assert torch.isnan(HSI_recon).sum() == 0
    return HSI_recon,val_loss