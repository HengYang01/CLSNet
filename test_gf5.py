import os
import torch
import torch.nn as nn
import argparse
import hdf5storage
import numpy as np
from torch.utils.data import DataLoader
import tqdm
from utils import *
from calculate_metrics import *
from train_dataloader import *
from model.CLSNet_Gf5 import CLSNet as allnet

def test(fusion_net: nn.Module, val_dataloader, args):
    print('===> Begin Testing!')

    SAM = Loss_SAM()
    RMSE = Loss_RMSE()
    PSNR = Loss_PSNR()

    psnr_avg, sam_avg, rmse_avg = 0., 0., 0.
    imglist = os.listdir(args.data_path)

    fusion_net.eval()
    with torch.no_grad():
        for cnt, (lrhs, hrms, hrhs) in enumerate(tqdm.tqdm(val_dataloader), 1):

            prediction = (reconstruction(
                fusion_net, lrhs.cuda(), hrms.cuda(),
                downsample_factor=args.spatial_ratio,
                training_size=args.test_size,
                stride=args.stride,
                batch_size=args.batch_size
            ).cuda()).detach()

            prediction_np = np.transpose(prediction.detach().cpu().numpy().squeeze(), (1, 2, 0))
            save_path = os.path.join(args.save_dir, "gf5_entire.mat")
            os.makedirs(args.save_dir, exist_ok=True)
            hdf5storage.savemat(save_path, {'fak': prediction_np,}, format='7.3')

    print("===> Test Results:")
    print(f"PSNR: {psnr_avg/cnt:.4f}, SAM: {sam_avg/cnt:.4f}, RMSE: {rmse_avg/cnt:.4f}")


def main(args):
    dir_idx = os.path.join("DataCache", str(args.dataset))
    if not os.path.exists(dir_idx):
        os.mkdir(dir_idx)
    args.cache_path = dir_idx

    # 数据集
    val_set = MakeSimulateDataset_GF5(args, "test_real")
    val_dataloader = DataLoader(dataset=val_set, batch_size=1, shuffle=False)
    fusion_net = allnet(48).cuda()
    backup_pth = os.path.join(args.model_path)
    print("==> Load checkpoint: {}".format(backup_pth))
    fusion_net.load_state_dict(torch.load(backup_pth)["fusion_net"], strict=False)

    test(fusion_net, val_dataloader, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Only')
    parser.add_argument('--dataset', type=str, default="GF5", help='Dataset to be loaded.')
    parser.add_argument('--test_size', type=int, default=64, help='Size of patch.')
    parser.add_argument('--stride', type=int, default=32, help='Stride for cropping.')
    parser.add_argument('--spatial_ratio', type=int, default=2, help='Spatial ratio between MS and PAN')
    parser.add_argument('--num_bands', type=int, default=150, help='Number of bands of MS image.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for testing.')
    parser.add_argument('--data_path', type=str, default="E:/data", help='Path of dataset.')
    parser.add_argument('--model_path', type=str, default="G:/hsi_fuse/fusion_code/fusion_code_2/GF5_Train_Results/SMGUNet/1/model/200.pth", help='Path to trained model checkpoint.')
    parser.add_argument('--save_dir', type=str, default="G:/paper1/Test_result/SMGUNet/gf5/", help='Dir to save test .mat results.')
    args = parser.parse_args()
    main(args)
