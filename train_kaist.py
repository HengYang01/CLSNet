import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  #（代表仅使用第0，1号GPU）
import time
import os
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse
import json
import numpy
from tqdm import tqdm
from scipy.io import loadmat
from calculate_metrics import *
from train_dataloader import *
import time
from utils import create_F, fspecial
import math
import torch
from model.CLSNet import CLSNet as allnet

def train(fusion_net: nn.Module, optimizers,scheduler, train_dataloader, val_dataloader, args):
    print('===>Begin Training!')
    start_epoch = 0
    if args.resume != "":
        start_epoch = int(args.resume) if "best" not in args.resume else int(args.resume.split("_")[-1])

    R = create_F()
    t = time.time()
    # device = next(pandemosaic_net.parameters()).device

    best_epoch, best_psnr = 0, 0
    numpy.set_printoptions(precision=3, suppress=True)
    
    SAM = Loss_SAM()
    RMSE = Loss_RMSE()
    PSNR = Loss_PSNR()
    
    for epoch in range(start_epoch + 1, args.epochs + 1):
        # train
        fusion_net.train()
        start_time = time.time()
        loss_sum = 0
        loop = tqdm.tqdm(train_dataloader, total=len(train_dataloader))
        for step, (lrhs, hrms, hrhs) in enumerate(loop,1):
            output = fusion_net(lrhs.cuda(),hrms.cuda())
            loss = torch.nn.functional.l1_loss(output, hrhs.cuda())
            optimizers.zero_grad()
            loss.backward()
            optimizers.step()
            scheduler.step()
            loss_sum += loss.item()
            train_avg_loss = loss_sum / step
            loop.set_description(f'epoch: {epoch}, loss: {train_avg_loss:.8f}')

        # val
        if epoch==1 or args.epochs - epoch < 5:
            psnr_avg = 0.
            sam_avg = 0.
            rmse_avg = 0.
            val_avg_loss = 0.
            fusion_net.eval()
            with torch.no_grad():
                for cnt, (lrhs, hrms, hrhs) in enumerate(val_dataloader, 1):
                    
                    hrhs_for_Indicator = hrhs.cuda().squeeze(0).detach()
                    prediction = (reconstruction(fusion_net, lrhs.cuda(), hrms.cuda(), downsample_factor = args.spatial_ratio, training_size = args.train_size, stride = args.stride).cuda()).detach()
                    val_loss = loss_func(prediction, hrhs_for_Indicator)
                    val_avg_loss += val_loss.item()
                    psnr_avg += PSNR(hrhs_for_Indicator, prediction).item()
                    sam_avg += SAM(hrhs_for_Indicator, prediction).item()
                    rmse_avg += RMSE(hrhs_for_Indicator, prediction).item()

            psnr_avg /= cnt
            sam_avg /= cnt
            rmse_avg /= cnt
            val_avg_loss /= cnt
        if args.record is not False:
            record = []
            if epoch==1 or args.epochs - epoch < 5:
                if os.path.exists(args.record):
                    with open(args.record, "r") as f:
                        record = json.load(f)
                record.append({"epoch": epoch,
                                "train_loss": train_avg_loss,
                                "val_loss": val_avg_loss,
                                "psnr": psnr_avg,
                                "sam": sam_avg,
                                "rmse": rmse_avg,
                                "best_psnr": best_psnr,
                                "best_epoch": best_epoch,
                                "learning rate": optimizers.param_groups[0]["lr"],
                                })
                with open(args.record, "w") as f:
                    record = json.dump(record, f, indent=2)

                
                # save model with highest PSNR
                if psnr_avg > best_psnr:
                    best_psnr = psnr_avg
                    if best_epoch != 0:
                        os.remove(os.path.join(args.dir_model, "best_{}.pth".format(best_epoch)))
                    best_epoch = epoch
                    torch.save({"fusion_net": fusion_net.state_dict()}, os.path.join(args.dir_model, "best_{}.pth".format(epoch)))
                
                torch.save({"fusion_net": fusion_net.state_dict()}, os.path.join(args.dir_model, f"{epoch}.pth"))

                # log
                print("Epoch: ", epoch,
                    "train_loss: %.4f"%train_avg_loss,
                    "val_loss: %.4f"%val_avg_loss,
                    "time: %.2f"%((time.time() - start_time) / 60), "min",
                    "psnr: %.4f"%psnr_avg,
                    "sam: %.4f"%sam_avg,
                    "rmse: %.4f"%rmse_avg,
                    "learning rate: ", optimizers.param_groups[0]["lr"], "\n",
                    )

    print(f"Total time: {(time.time() - t) / 60} min")
    print("Best epoch: {}, Best PSNR: {}".format(best_epoch, best_psnr))

def main(args):
    dir_idx = os.path.join("DataCache", str(args.dataset))
    if not os.path.exists(dir_idx):
        os.makedirs(dir_idx, exist_ok=True)
    args.cache_path = dir_idx

    dir_train = os.path.join(str(args.dataset) + "_Train_Results", str(args.model_name), str(args.idx))
    os.makedirs(dir_train, exist_ok=True)

    dir_model = os.path.join(dir_train, "model")
    if not os.path.exists(dir_model):
        os.mkdir(dir_model)
    args.dir_model = dir_model

    if args.record is True:
        dir_record = os.path.join(dir_train, "record")
        if not os.path.exists(dir_record):
            os.makedirs(dir_record)
        args.dir_record = dir_record
        args.record = os.path.join(dir_record, "record.json")
        if args.resume == "" and os.path.exists(args.record):
            os.remove(args.record)

    maxiteration = math.ceil(((2704 - args.train_size) // args.stride + 1) * ((3376 - args.train_size) // args.stride + 1) * args.nums / args.batch_size) * args.epochs
    print("maxiteration:", maxiteration)

    train_set = MakeSimulateDataset(args, "train")
    train_dataloader = DataLoader(dataset=train_set, batch_size = args.batch_size, shuffle=True)
    val_set = MakeSimulateDataset(args, "test")
    val_dataloader = DataLoader(dataset=val_set, batch_size = 1, shuffle=False)

    fusion_net = allnet().cuda()

    if args.resume != "":
        backup_pth = os.path.join(dir_model, args.resume + ".pth")
        print("==> Load checkpoint: {}".format(backup_pth))
        fusion_net.load_state_dict(torch.load(backup_pth)["fusion_net"], strict=False)
    else:
        print('==> Train from scratch')
     

    for m in fusion_net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    optimizer = torch.optim.Adam(fusion_net.parameters() ,lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-8 )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, maxiteration, eta_min=1e-6)

    train(fusion_net, optimizer,scheduler, train_dataloader, val_dataloader, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--idx', type=int, default="1", help='Index to identify models.')
    parser.add_argument('--model_name', type=str, default="ours", help='Index to identify models.')
    parser.add_argument('--dataset', type=str, default="KAIST", help='Dataset to be loaded.')
    parser.add_argument('--train_size', type=int, default=64, help='Size of the training image in a batch.')
    parser.add_argument('--spatial_ratio', type=int, default=8, help='Ratio of spatial resolutions between MS and PAN')
    parser.add_argument('--num_bands', type=int, default=31, help='Number of bands of a MS image.')
    parser.add_argument('--stride', type=int, default=32, help='Stride when crop an original image into patches.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training dataset.')
    parser.add_argument('--epochs', type=int, default=1000, help='Total epochs to train the model.')
    parser.add_argument('--save_freq', type=int, default=50, help='Save the checkpoints of the model every [save_freq] epochs.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer.')
    parser.add_argument('--lr_decay', action="store_true", help='Determine if to decay the learning rate.')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate to train the model.')
    parser.add_argument('--device', type=str, default='0', help='Device to train the model.')
    parser.add_argument('--num_workers', type=int, default=1, help='Num_workers to train the model.')
    parser.add_argument('--resume', type=str, default='', help='Index of the model to be resumed, eg. 1000.')
    parser.add_argument('--data_path', type=str, default="E:/data", help='Path of the dataset.')
    parser.add_argument('--nums', type=int, default="20", help='Number of the images.')
    parser.add_argument('--record', type=bool, default=True, help='Whether to record the PSNR of each epoch.')

    args = parser.parse_args()
    main(args)