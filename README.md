# Train
## CAVE
python train_cave.py --dataset CAVE --epochs 1000 --train_size 64 --stride 32 --batch_size 32 --save_freq 50 --model_name ours --idx 1

## HARVARD
python train_harvard.py --dataset HARVARD --epochs 300 --train_size 64 --stride 32 --batch_size 16 --save_freq 50 --model_name ours --idx 1

## KAIST
python train_kaist.py --dataset KAIST --epochs 150 --train_size 64 --stride 64 --batch_size 16 --save_freq 50 --model_name ours --idx 1

## GF5
python train_gf5.py --dataset GF5 --epochs 200 --train_size 64 --stride 8 --batch_size 4 --save_freq 50 --model_name ours --idx 1


# Test
## CAVE
python test_cave.py --dataset CAVE --data_path E:/data/CAVE/test/ --model_path G:\hsi_fuse\fusion_code\fusion_code_2\CAVE_Train_Results\ours\1\model\1000.pth --save_dir G:/Test_result/ours/CAVE/

## HARVATD
python test_harvard.py --dataset HARVARD --data_path E:/data/HARVARD/test/ --model_path G:\hsi_fuse\fusion_code\fusion_code_2\HARVARD_Train_Results\ours\1\model\300.pth --save_dir G:/Test_result/ours/Harvard/

## KAIST
python test_kaist.py --dataset KAIST --stride 64 --data_path E:/data/KAIST/test/ --model_path G:\hsi_fuse\fusion_code\fusion_code_2\KAIST_Train_Results\ours\1\model\150.pth --save_dir G:/paper1/Test_result/ours/KAIST/

## GF5
python test_gf5.py --dataset GF5 --data_path E:/data/GF5/ --model_path G:/hsi_fuse/fusion_code/fusion_code_2/GF5_Train_Results/ours/1/model/200.pth --save_dir G:/paper1/Test_result/ours/gf5/