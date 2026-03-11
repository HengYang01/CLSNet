import os
import torch
import torch.nn as nn
import argparse
import hdf5storage
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *
from calculate_metrics import *
from train_dataloader import MakeSimulateDataset
from model.CLSNet import CLSNet as allnet

def test(fusion_net: nn.Module, val_dataloader, args):
    print('===> Begin Testing!')

    SAM = Loss_SAM()
    RMSE = Loss_RMSE()
    PSNR = Loss_PSNR()

    psnr_avg, sam_avg, rmse_avg = 0., 0., 0.
    imglist = os.listdir(args.data_path)

    fusion_net.eval()
    with torch.no_grad():
        for cnt, (lrhs, hrms, hrhs) in enumerate(tqdm(val_dataloader), 1):
            hrhs_gt = hrhs.cuda().squeeze(0).detach()

            prediction = (reconstruction(
                fusion_net, lrhs.cuda(), hrms.cuda(),
                downsample_factor=args.spatial_ratio,
                training_size=args.test_size,
                stride=args.stride,
                batch_size=args.batch_size
            ).cuda()).detach()
            psnr_val = PSNR(hrhs_gt, prediction).item()
            sam_val = SAM(hrhs_gt, prediction).item()
            rmse_val = RMSE(hrhs_gt, prediction).item() 

            psnr_avg += psnr_val
            sam_avg += sam_val
            rmse_avg += rmse_val
            # 单张图像结果
            print(f"[{cnt}/{len(val_dataloader)}] {imglist[cnt-1]} -> "
                  f"PSNR: {psnr_val:.4f}, SAM: {sam_val:.4f}, RMSE: {rmse_val:.4f}")

            prediction_np = np.transpose(prediction.detach().cpu().numpy().squeeze(), (1, 2, 0))
            hrhs_gt_np = np.transpose(hrhs_gt.detach().cpu().numpy().squeeze(), (1, 2, 0))

            save_path = os.path.join(args.save_dir, imglist[cnt-1])
            os.makedirs(args.save_dir, exist_ok=True)
            hdf5storage.savemat(save_path, {'fak': prediction_np,
                                            'rea': hrhs_gt_np}, format='7.3')

    print("===> Test Results:")
    print(f"PSNR: {psnr_avg/cnt:.4f}, SAM: {sam_avg/cnt:.4f}, RMSE: {rmse_avg/cnt:.4f}")

def main(args):
    dir_idx = os.path.join("DataCache", str(args.dataset))
    if not os.path.exists(dir_idx):
        os.mkdir(dir_idx)
    args.cache_path = dir_idx

    val_set = MakeSimulateDataset(args, "test")
    val_dataloader = DataLoader(dataset=val_set, batch_size=1, shuffle=False)
    fusion_net = allnet().cuda()#OUR
    backup_pth = os.path.join(args.model_path)
    print("==> Load checkpoint: {}".format(backup_pth))
    fusion_net.load_state_dict(torch.load(backup_pth)["fusion_net"], strict=False)
    test(fusion_net, val_dataloader, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Only')
    parser.add_argument('--dataset', type=str, default="KAIST", help='Dataset to be loaded.')
    parser.add_argument('--test_size', type=int, default=64, help='Size of patch.')
    parser.add_argument('--stride', type=int, default=32, help='Stride for cropping.')
    parser.add_argument('--spatial_ratio', type=int, default=8, help='Spatial ratio between MS and PAN')
    parser.add_argument('--num_bands', type=int, default=31, help='Number of bands of MS image.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for testing.')
    parser.add_argument('--data_path', type=str, default="E:/data/CAVE/test/", help='Path of dataset.')
    parser.add_argument('--model_path', type=str, required= True, help='Path to trained model checkpoint.')
    parser.add_argument('--save_dir', type=str, default="G:/paper1/Test result/CSGAV\CAVE/", help='Dir to save test .mat results.')
    args = parser.parse_args()
    main(args)
