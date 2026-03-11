import os
import numpy as np
import torch
from hdf5storage import loadmat
from torch import nn

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


class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=255):
        # 对输入进行裁剪并缩放到数据范围
        Itrue = im_true.clamp(0., 1.) * data_range
        Ifake = im_fake.clamp(0., 1.) * data_range
        
        # 计算平方误差
        err = Itrue - Ifake
        err = torch.pow(err, 2)
        
        # 对于(C, H, W)格式，在空间维度H和W上计算均值（dim=1和dim=2）
        err = torch.mean(err, dim=1)  # 对H维度求均值
        err = torch.mean(err, dim=1)  # 对W维度求均值（此时维度已变为[C]）
        
        # 计算PSNR
        psnr = 10. * torch.log10((data_range ** 2) / err)
        psnr = torch.mean(psnr)  # 对所有通道求均值
        return psnr


class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs.clamp(0., 1.)*255- label.clamp(0., 1.)*255
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error.contiguous().view(-1)))
        return rmse

class Loss_SAM1(nn.Module):
    def __init__(self):
        super(Loss_SAM1, self).__init__()
        self.eps=2.2204e-16
    def forward(self,im1, im2):
        assert im1.shape == im2.shape
        H,W,C=im1.shape
        im1 = np.reshape(im1,( H*W,C))
        im2 = np.reshape(im2,(H*W,C))
        core=np.multiply(im1, im2)
        mole = np.sum(core, axis=1)
        im1_norm = np.sqrt(np.sum(np.square(im1), axis=1))
        im2_norm = np.sqrt(np.sum(np.square(im2), axis=1))
        deno = np.multiply(im1_norm, im2_norm)
        sam = np.rad2deg(np.arccos(((mole+self.eps)/(deno+self.eps)).clip(-1,1)))
        return np.mean(sam)

class Loss_SAM(nn.Module):
    def __init__(self):
        super(Loss_SAM, self).__init__()
        self.eps = 2.2204e-16  # 防止除零和数值不稳定
    
    def forward(self, im1, im2):
        # 确保输入张量形状相同
        assert im1.shape == im2.shape, "输入张量形状必须一致"
        C, H, W = im1.shape  # 对于(C, H, W)格式，提取通道、高度、宽度
        
        # 重塑为 [H*W, C]：将空间维度(H, W)展平，保留通道维度作为光谱向量
        # 先转置为(H, W, C)再展平，或直接展平空间维度
        im1_reshaped = im1.permute(1, 2, 0).contiguous().view(-1, C)  # 先转置再展平
        im2_reshaped = im2.permute(1, 2, 0).contiguous().view(-1, C)
        
        # 计算分子：两个光谱向量的点积（沿通道维度求和）
        core = torch.multiply(im1_reshaped, im2_reshaped)
        mole = torch.sum(core, dim=1)  # 沿通道维度(C)求和
        
        # 计算分母：两个光谱向量的模长乘积
        im1_norm = torch.sqrt(torch.sum(torch.square(im1_reshaped), dim=1))  # 每个空间位置的光谱模长
        im2_norm = torch.sqrt(torch.sum(torch.square(im2_reshaped), dim=1))
        deno = torch.multiply(im1_norm, im2_norm)
        
        # 计算光谱角并转换为角度（度）
        cos_theta = ((mole + self.eps) / (deno + self.eps)).clamp(-1.0, 1.0)  # 确保在有效范围内
        sam = torch.rad2deg(torch.arccos(cos_theta))  # 弧度转角度
        
        # 返回平均光谱角误差
        return torch.mean(sam)


if __name__ == '__main__':
    SAM=Loss_SAM()
    RMSE=Loss_RMSE()
    PSNR=Loss_PSNR()
    psnr_list=[]
    sam_list=[]
    sam=AverageMeter()
    rmse=AverageMeter()
    psnr=AverageMeter()
    path = 'D:\LYY\YJX_fusion\model_save\\fusion_model_v9_1\cavee_test/'
    imglist = os.listdir(path)

    for i in range(0, len(imglist)):
        img = loadmat(path + imglist[i])
        lable = img["rea"]
        recon = img["fak"]
        sam_temp=SAM(lable,recon)
        psnr_temp=PSNR(torch.Tensor(lable), torch.Tensor(recon))
        sam.update(sam_temp)
        rmse.update(RMSE(torch.Tensor(lable),torch.Tensor(recon)))
        psnr.update(psnr_temp)
        psnr_list.append(psnr_temp)
        sam_list.append(sam_temp)
    print(sam.avg)
    print(rmse.avg)
    print(psnr.avg)
    print(psnr_list)
    print(sam_list)