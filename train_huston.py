os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from scipy.io import loadmat
from calculate_metrics import Loss_SAM, Loss_RMSE, Loss_PSNR
from train_dataloader import *
from torch import nn
from tqdm import tqdm
import time
import pandas as pd
import torch.utils.data as data
from utils import create_F, fspecial
import math
import hdf5storage as h5
from model.CLSNet_Houston import CLSNet as allnet

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("训练文件夹为：{}".format(path))
    else:
        print('已存在{}'.format(path))


if __name__ == '__main__':
    root=os.getcwd()+"/train_save_Comparison"
    model_name='Our'
    mkdir(os.path.join(root,model_name))
    ori_list=os.listdir(os.path.join(root,model_name))
    current_list=[]
    for i in ori_list:
        if len(i)<=2:
            current_list.append(i)

    del ori_list

    current_list.append('0')
    int_list = [int(x) for x in current_list]
    train_value = max(int_list)+1
    model_name=os.path.join(model_name,str(train_value))

    training_size=32
    stride=8
    downsample_factor=8


    data_name='huston'
    HSI_ori = h5.loadmat(r"F:\data\Huston.mat")
    HSI_ori = np.float32(HSI_ori["HSI"])
    HSI_ori = HSI_ori/HSI_ori.max()

    R = create_F()
    PSF=fspecial('gaussian', 9, 3)
    HSI_ori = np.transpose(HSI_ori, (2, 0, 1))

    grouped_channels = np.array_split(HSI_ori, 4, axis=0)
    result_channels = [np.mean(group, axis=0) for group in grouped_channels]
    MSI = np.stack(result_channels, axis=0)

    C,W,H=HSI_ori.shape
    dw =  W//downsample_factor
    dh = H//downsample_factor
    HSI_ori = HSI_ori[:,:dw*downsample_factor,:dh*downsample_factor]
    MSI = MSI[:,:dw*downsample_factor,:dh*downsample_factor]

    HSI_D1 = Gaussian_downsample(HSI_ori, PSF, downsample_factor)
    MSI_D1= Gaussian_downsample(MSI, PSF, downsample_factor)

    C,W,H = HSI_D1.shape
    dw=W//downsample_factor
    dh=H//downsample_factor
    HSI_D11=HSI_D1[:,:dw*downsample_factor,:dh*downsample_factor]

    C,W,H = MSI_D1.shape
    dw=W//downsample_factor
    dh=H//downsample_factor
    MSI_D2=MSI_D1[:,:dw*downsample_factor,:dh*downsample_factor]
    HSI_D2 = Gaussian_downsample(HSI_D11, PSF, downsample_factor)
    # MSI_D2 = Gaussian_downsample(MSI_D11, PSF, downsample_factor)

    print("训练数据处理完成")
    # 训练参数
    loss_func = nn.L1Loss(reduction='mean').cuda()

    stride1 = 32
    LR = 6e-4
    EPOCH = 200
    weight_decay = 0    # 我的模型是1e-8
    BATCH_SIZE = 4
    psnr_optimal = 40
    rmse_optimal = 3

    test_epoch=50
    val_interval = 50           # 每隔val_interval epoch测试一次
    checkpoint_interval = 100

    # 创建方法名字的文件
    path=os.path.join(root,model_name)
    mkdir(path)  # 创建文件夹
    # 创建训练记录文件
    pkl_name=data_name+'_pkl'
    pkl_path=os.path.join(path,pkl_name)      # 模型保存路径
    os.makedirs(pkl_path)      # 创建文件夹
    # 创建excel
    df = pd.DataFrame(columns=['epoch', 'lr', 'train_loss','val_loss','val_rmse', 'val_psnr', 'val_sam'])  # 列名
    excel_name=data_name+'_record.csv'
    excel_path=os.path.join(path,excel_name)
    df.to_csv(excel_path, index=False)

    df2 = pd.DataFrame(columns=['epoch', 'lr', 'train_loss'])  # 列名
    excel_name2 = data_name + '_train_record.csv'
    excel_path2 = os.path.join(path, excel_name2)
    df2.to_csv(excel_path2, index=False)

    train_data = RealDATAProcess3(HSI_D2,MSI_D2,HSI_D11,training_size, stride, downsample_factor)
    train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    maxiteration = math.ceil(len(train_data) / BATCH_SIZE) * EPOCH
    print("maxiteration：", maxiteration)

    decay_power = 1.5
    init_lr2 = 1e-4
    init_lr1 = 1e-4 / 10
    min_lr=0
    warm_iter = math.floor(maxiteration / 40)

    cnn = allnet().cuda()
    # 模型初始化
    for m in cnn.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, betas=(0.9, 0.999),weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, maxiteration, eta_min=1e-6)

    start_epoch = 0
    # resume = True
    resume = False
    path_checkpoint = "checkpoints/500_epoch.pkl"  # 断点路径

    # start_step=0
    if resume:
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        cnn.load_state_dict(checkpoint['model_state_dict'])  # 加载模型可学习参数
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
        psnr_optimal = checkpoint['psnr_optimal']
        rmse_optimal = checkpoint['rmse_optimal']

    step=0   # warm_lr_scheduler要用
    for epoch in range(start_epoch+1, EPOCH+1):
        cnn.train()
        loss_all = []
        loop = tqdm(train_loader, total=len(train_loader))
        for a1, a2, a3 in loop:
            lr = optimizer.param_groups[0]['lr']
            step = step + 1
            output = cnn(a3.cuda(), a2.cuda())   # hsrnet
            loss = loss_func(output, a1.cuda())

            loss_temp = loss
            loss_all.append(np.array(loss_temp.detach().cpu().numpy()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            loop.set_description(f'Epoch [{epoch}/{EPOCH}]')
            loop.set_postfix({'loss': '{0:1.8f}'.format(np.mean(loss_all)), "lr": '{0:1.8f}'.format(lr)})

        train_list = [epoch, lr, np.mean(loss_all)]
        train_record = pd.DataFrame([train_list])
        train_record.to_csv(excel_path2, mode='a', header=False, index=False)  # mode设为a,就可以向csv文件追加数据了

        if ((epoch % val_interval == 0) and (epoch>=test_epoch) ) or epoch==1:
            cnn.eval()
            val_loss=AverageMeter()
            SAM = Loss_SAM()
            RMSE = Loss_RMSE()
            PSNR = Loss_PSNR()
            sam = AverageMeter()
            rmse = AverageMeter()
            psnr = AverageMeter()

            with torch.no_grad():
                test_HRHSI = torch.unsqueeze(torch.Tensor(HSI_ori),0)
                test_HRMSI =torch.unsqueeze(torch.Tensor(MSI),0)
                test_LRHSI=torch.unsqueeze(torch.Tensor(HSI_D1),0)

                prediction,val_loss = reconstruction_huston(cnn, R, test_LRHSI.cuda(), test_HRMSI.cuda(), test_HRHSI,downsample_factor, training_size, stride1,val_loss)
                sam.update(SAM(np.transpose(test_HRHSI.squeeze().cpu().detach().numpy(),(1, 2, 0)),np.transpose(prediction.squeeze().cpu().detach().numpy(),(1, 2, 0))))
                rmse.update(RMSE(test_HRHSI.squeeze().cpu().permute(1,2,0),prediction.squeeze().cpu().permute(1,2,0)))
                psnr.update(PSNR(test_HRHSI.squeeze().cpu().permute(1,2,0),prediction.squeeze().cpu().permute(1,2,0)))

            if  epoch == 1 or epoch==EPOCH:
                torch.save(cnn.state_dict(),pkl_path +'/'+ str(epoch) + 'EPOCH' + '_PSNR_best.pkl')

            if torch.abs(psnr_optimal-psnr.avg)<0.15 or psnr.avg>psnr_optimal:
                torch.save(cnn.state_dict(), pkl_path + '/' + str(epoch) + 'EPOCH' + '_PSNR_best.pkl')
            if psnr.avg > psnr_optimal:
                psnr_optimal = psnr.avg

            if torch.abs(rmse.avg-rmse_optimal)<0.15 or rmse.avg<rmse_optimal:
                torch.save(cnn.state_dict(),pkl_path +'/'+ str(epoch) + 'EPOCH' + '_RMSE_best.pkl')
            if rmse.avg < rmse_optimal:
                rmse_optimal = rmse.avg

            print("val  PSNR:",psnr.avg.cpu().detach().numpy(), "  RMSE:", rmse.avg.cpu().detach().numpy(), "  SAM:", sam.avg,"val loss:", val_loss.avg.cpu().detach().numpy())
            val_list = [epoch, lr,np.mean(loss_all),val_loss.avg.cpu().detach().numpy(),rmse.avg.cpu().detach().numpy(), psnr.avg.cpu().detach().numpy(), sam.avg]

            val_data = pd.DataFrame([val_list])
            val_data.to_csv(excel_path,mode='a', header=False, index=False)  # mode设为a,就可以向csv文件追加数据了
            time.sleep(0.1)

    C,W,H=HSI_ori.shape
    dw=W//downsample_factor
    dh=H//downsample_factor   #dh=21

    dh_t=10
    train_HSI=HSI_ori[:,:dw*downsample_factor,:dh_t*downsample_factor].astype(np.float32)
    train_MSI=MSI[:,:dw*downsample_factor*downsample_factor,:dh_t*downsample_factor*downsample_factor].astype(np.float32)


    test_HSI = HSI_ori[:, :dw * downsample_factor, dh_t * downsample_factor:dh * downsample_factor].astype(np.float32)
    test_MSI = MSI[:, :dw * downsample_factor * downsample_factor,
                dh_t * downsample_factor * downsample_factor:dh * downsample_factor * downsample_factor].astype(np.float32)