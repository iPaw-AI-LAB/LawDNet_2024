import sys
import cv2
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import yaml
import argparse
import datetime
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.cuda.amp import autocast as autocast
from models.Discriminator import Discriminator
from models.VGG19 import Vgg19
from models.LawDNet import LawDNet
from models.Syncnet import SyncNetPerception
from utils.training_utils import get_scheduler, update_learning_rate, GANLoss
from config.config import DINetTrainingOptions
from sync_batchnorm import convert_model
from torch.utils.data import DataLoader
from dataset.dataset_DINet_clip import DINetDataset
# from models.Gaussian_blur import Gaussian_bluring
from tensor_processing import SmoothSqMask
# from models.content_model import AudioContentModel, LipContentModel
from torch.nn.utils import clip_grad_norm_


# def setup(rank, world_size, master_addr='localhost', master_port='12355'):
#     """
#     Initialize the distributed environment.

#     Parameters:
#     - rank: The rank of the current process in the distributed setup.
#     - world_size: The total number of processes in the distributed setup.
#     - master_addr: The address of the master node. Default is 'localhost'.
#     - master_port: The port on which to listen or connect to the master node. Default is '12355'.
#     """
#     os.environ['MASTER_ADDR'] = master_addr
#     os.environ['MASTER_PORT'] = master_port
#     torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
#     torch.cuda.set_device(rank)

# 初始化和登录WandB
def init_wandb(name,rank):
    if rank == 0:
        wandb.login()
        run = wandb.init(project=name)

def load_experiment_config(config_module_path):
    """动态加载指定的配置文件"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("config_module", config_module_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.experiment_config

# 加载配置和设备设置
def load_config_and_device(args, rank):
    # import pdb; pdb.set_trace()
    '''加载配置和设置设备'''

    # 动态加载配置文件
    experiment_config = load_experiment_config(args.config_path)

    # 创建 opt 实例，这里避免 argparse 解析命令行参数

    opt = DINetTrainingOptions().parse_args()

    # 根据动态加载的配置更新 opt
    for key, value in experiment_config.items():
        if hasattr(opt, key):
            setattr(opt, key, value)

    # 假设 wandb 已经初始化
    if rank == 0:
        wandb.config.update(opt)  # 如果使用 wandb，可以这样更新配置

    random.seed(opt.seed + rank)
    np.random.seed(opt.seed+ rank)
    torch.cuda.manual_seed_all(opt.seed+rank)
    torch.manual_seed(opt.seed+rank)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 根据实验名称直接修改文件夹名字
    # 分割result_path，最多分割成两部分
    path_parts = opt.result_path.rsplit('/', 1)

    # 在倒数第一个/之前插入本次实验的名字
    opt.result_path = f'{path_parts[0]}/{args.name}/{path_parts[1]}'

    return opt

# Save configuration to a YAML file
def save_config_to_yaml(config, filename):
    with open(filename, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

# 加载训练数据
def load_training_data(opt, world_size, rank):
    train_data = DINetDataset(opt.train_data, opt.augment_num, opt.mouth_region_size)
    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True)
    training_data_loader = DataLoader(
        dataset=train_data,
        batch_size=opt.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory,
        drop_last=True
    )
    return training_data_loader, train_sampler


# 初始化网络
def init_networks(opt,rank):

    net_g = LawDNet(opt.source_channel, opt.ref_channel, opt.audio_channel, 
                    opt.warp_layer_num, opt.num_kpoints, opt.coarse_grid_size, rank).to(rank)
    net_dI = Discriminator(opt.source_channel, opt.D_block_expansion, opt.D_num_blocks, opt.D_max_features).to(rank)
    net_dV = Discriminator(opt.source_channel * 5, opt.D_block_expansion, opt.D_num_blocks, opt.D_max_features).to(rank)
    net_vgg = Vgg19().to(rank)
    net_lipsync = SyncNetPerception(opt.pretrained_syncnet_path).to(rank)

    # torch2.0 compile
    # net_g = torch.compile(net_g, mode="reduce-overhead")
    # net_dI = torch.compile(net_dI, mode="reduce-overhead")
    # net_dV = torch.compile(net_dV, mode="reduce-overhead")
    # net_vgg = torch.compile(net_vgg, mode="reduce-overhead").to(rank)
    # net_lipsync = torch.compile(net_lipsync, mode="reduce-overhead").to(rank)

    print("net_lipsync is loaded")

    # device_ids = [int(x) for x in opt.cuda_devices.split(',')]

    # 其它网络没有DDP
    net_g = convert_model(net_g)
    net_g = DDP(net_g, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    print("net_g is loaded with DDP")
    net_dI = DDP(net_dI, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    print("net_dI is loaded with DDP")
    net_dV = DDP(net_dV, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    print("net_dV is loaded with DDP")

    return net_g, net_dI, net_dV, net_vgg, net_lipsync

# 设置优化器
def setup_optimizers(net_g, net_dI, net_dV):
    optimizer_g = optim.AdamW(net_g.parameters(), lr=opt.lr_g)
    optimizer_dI = optim.AdamW(net_dI.parameters(), lr=opt.lr_dI)
    optimizer_dV = optim.AdamW(net_dV.parameters(), lr=opt.lr_dI)
    return optimizer_g, optimizer_dI, optimizer_dV


def load_pretrained_weights(net_g, 
                            net_dI, 
                            net_dV, 
                            optimizer_g, 
                            optimizer_dI, 
                            optimizer_dV,
                            scheduler_g,
                            scheduler_dI,
                            scheduler_dV,
                            opt, 
                            args):
    """
    Loads the pretrained weights into the models and optimizers if a valid path is provided and
    depending on the resume option, it either loads all weights or only the generator weights.

    Parameters:
    - net_g, net_dI, net_dV: models to load the weights into.
    - optimizer_g, optimizer_dI, optimizer_dV: optimizers to load the states into.
    - opt: options object that contains the path to the pretrained weights and the resume flag.
    - args: additional arguments, might be used to modify the path dynamically.
    
    Returns:
    - A boolean value indicating whether the weights were loaded successfully.
    """
    
    if opt.pretrained_frame_DINet_path:
        path_parts = opt.pretrained_frame_DINet_path.rsplit('/', 2)
        modified_path = f'{path_parts[0]}/{args.name}/{path_parts[1]}/{path_parts[2]}'
        try:
            print(f'Loading pretrained weights from: {modified_path}')
            checkpoint = torch.load(modified_path)
            
            # Always load state_dict for net_g for coarse to fine training
            net_g.load_state_dict(checkpoint['state_dict']['net_g'])
            
            if opt.resume:
                print('resume training, Loading all weights and optimizers')
                # If resuming, load state_dicts for net_dI and net_dV, and all optimizers
                net_dI.load_state_dict(checkpoint['state_dict']['net_dI'])
                net_dV.load_state_dict(checkpoint['state_dict']['net_dV'])
                optimizer_g.load_state_dict(checkpoint['optimizer']['net_g'])
                optimizer_dI.load_state_dict(checkpoint['optimizer']['net_dI'])
                optimizer_dV.load_state_dict(checkpoint['optimizer']['net_dV'])
                # Load scheduler states
                if 'scheduler' in checkpoint:
                    scheduler_g.load_state_dict(checkpoint['scheduler']['net_g'])
                    scheduler_dI.load_state_dict(checkpoint['scheduler']['net_dI'])
                    scheduler_dV.load_state_dict(checkpoint['scheduler']['net_dV'])

                # Optionally, load the epoch number to resume training correctly
                opt.start_epoch = checkpoint['epoch']
            print('Loading pretrained weights finished!')
            return True
        except Exception as e:
            sys.exit(1)
            print(f'Error loading pretrained weights: {e}')
            return False
    else:
        print("Path to pretrained weights is empty.")
        return False

    
# 设置损失函数
def setup_criterion():
    criterionGAN = GANLoss()
    criterionL1 = nn.L1Loss()
    criterionMSE = nn.MSELoss()
    # criterionCosine = nn.CosineEmbeddingLoss()
    return criterionGAN, criterionL1, criterionMSE

# 设置学习率调度器
def setup_schedulers(optimizer_g, optimizer_dI, optimizer_dV):
    net_g_scheduler = get_scheduler(optimizer_g, opt.non_decay, opt.decay)
    net_dI_scheduler = get_scheduler(optimizer_dI, opt.non_decay, opt.decay)
    net_dV_scheduler = get_scheduler(optimizer_dV, opt.non_decay, opt.decay)
    return net_g_scheduler, net_dI_scheduler, net_dV_scheduler

def log_to_wandb(source_clip, source_clip_mask, fake_out):
    # source_clip = source_clip.float()  # 将数据转换为全精度
    # fake_out = fake_out.float()        # 同上

    # 可视化原始source_clip
    images_source = [wandb.Image(source_clip[i].float().detach().cpu(), caption=f"Source Clip {i}") for i in range(source_clip.shape[0])]
    wandb.log({"Source Clips": images_source})

    # 可视化source_clip_mask
    images_source_mask = [wandb.Image(source_clip_mask[i].float().detach().cpu(), caption=f"Source Clip Mask {i}") for i in range(source_clip_mask.shape[0])]
    wandb.log({"Source Clip Masks": images_source_mask})

    # 可视化fake_out
    images_fake_out = [wandb.Image(fake_out[i].float().detach().cpu(), caption=f"Fake Out {i}") for i in range(fake_out.shape[0])]
    wandb.log({"Fake Outs": images_fake_out})

# 训练过程
def train(
    opt, 
    net_g, 
    net_dI, 
    net_dV, 
    training_data_loader, 
    train_sampler,
    rank,
    optimizer_g, 
    optimizer_dI, 
    optimizer_dV, 
    criterionGAN, 
    criterionL1, 
    criterionMSE, 
    net_g_scheduler, 
    net_dI_scheduler, 
    net_dV_scheduler
):
    device_id = rank % torch.cuda.device_count()
    criterionMSE = criterionMSE.to(device_id)
    criterionGAN = criterionGAN.to(device_id)
    criterionL1 = criterionL1.to(device_id)
    # criterionCosine = criterionCosine.to(device_id)

    # 假定mouth_region_size定义了唇部区域的大小，并在train_data中已正确设置
    mouth_region_size = opt.mouth_region_size
    radius = mouth_region_size // 2
    radius_1_4 = radius // 4

    # 计算口部区域的起始和结束索引
    start_x, start_y = radius, radius_1_4
    end_x, end_y = start_x + mouth_region_size, start_y + mouth_region_size

    # 混合精度训练：Creates a GradScaler once at the beginning of training.
    
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    smooth_sqmask = SmoothSqMask(device=device_id).to(device_id)

    for epoch in range(opt.start_epoch, opt.non_decay + opt.decay + 1):
        train_sampler.set_epoch(epoch)
        net_g.train()
        for iteration, data in enumerate(tqdm(training_data_loader, desc=f"Epoch {epoch} of {opt.non_decay + opt.decay}")):
            source_clip, reference_clip, deep_speech_clip, deep_speech_full, flag = data
            flag = flag.to(device_id)
            # 检查是否有脏数据
            if not (flag.equal(torch.ones(opt.batch_size, 1, device=device_id))):
                print("跳过含有脏数据的批次")
                continue
            
            source_clip = torch.cat(torch.split(source_clip, 1, dim=1), 0).squeeze(1).float().to(device_id)
            reference_clip = torch.cat(torch.split(reference_clip, 1, dim=1), 0).squeeze(1).float().to(device_id)
            deep_speech_clip = torch.cat(torch.split(deep_speech_clip, 1, dim=1), 0).squeeze(1).float().to(device_id)
            deep_speech_full = deep_speech_full.float().to(device_id)

            # 生成mask
            source_clip_mask = smooth_sqmask(source_clip)

            with autocast(enabled=True):
                fake_out = net_g(source_clip_mask, reference_clip, deep_speech_clip)

            fake_out_half = F.avg_pool2d(fake_out, 3, 2, 1, count_include_pad=False)
            source_clip_half = F.interpolate(source_clip, scale_factor=0.5, mode='bilinear')

            # 更新判别器DI
            optimizer_dI.zero_grad()
            with autocast(enabled=True):
                _, pred_fake_dI = net_dI(fake_out)
                loss_dI_fake = criterionGAN(pred_fake_dI, False)
                _, pred_real_dI = net_dI(source_clip)
                loss_dI_real = criterionGAN(pred_real_dI, True)
                loss_dI = (loss_dI_fake + loss_dI_real) * 0.5
            scaler.scale(loss_dI).backward(retain_graph=True)
            scaler.step(optimizer_dI)

            # 更新判别器DV
            optimizer_dV.zero_grad()
            condition_fake_dV = torch.cat(torch.split(fake_out, opt.batch_size, dim=0), 1)

            with autocast(enabled=True):
                _, pred_fake_dV = net_dV(condition_fake_dV)
                loss_dV_fake = criterionGAN(pred_fake_dV, False)
                condition_real_dV = torch.cat(torch.split(source_clip, opt.batch_size, dim=0), 1)
                _, pred_real_dV = net_dV(condition_real_dV)
                loss_dV_real = criterionGAN(pred_real_dV, True)
                loss_dV = (loss_dV_fake + loss_dV_real) * 0.5
            scaler.scale(loss_dV).backward(retain_graph=True)
            scaler.step(optimizer_dV)

            # 更新生成器
            optimizer_g.zero_grad()
            with autocast(enabled=True):
                _, pred_fake_dI = net_dI(fake_out)
                _, pred_fake_dV = net_dV(condition_fake_dV)
                perception_real = net_vgg(source_clip)
                perception_fake = net_vgg(fake_out)
                perception_real_half = net_vgg(source_clip_half)
                perception_fake_half = net_vgg(fake_out_half)

                # -----------------感知损失计算----------------- #
                loss_g_perception = 0
                for i in range(len(perception_real)):
                    loss_g_perception += criterionL1(perception_fake[i], perception_real[i])
                    loss_g_perception += criterionL1(perception_fake_half[i], perception_real_half[i])
                loss_g_perception = (loss_g_perception / (len(perception_real) * 2)) * opt.lamb_perception

                # -----------------GAN损失计算----------------- #
                loss_g_dI = criterionGAN(pred_fake_dI, True) * opt.lambda_g_dI
                loss_g_dV = criterionGAN(pred_fake_dV, True) * opt.lambda_g_dV

                # -----------------唇形同步损失计算----------------- #
                fake_out_clip = torch.cat(torch.split(fake_out, opt.batch_size, dim=0), 1)

                fake_out_clip_mouth_origin_size = fake_out_clip[:, :, start_x:end_x, start_y:end_y]

                # 将唇形部分调整到256x256，适应lip-sync网络
                if mouth_region_size != 256:
                    fake_out_clip_mouth = F.interpolate(fake_out_clip_mouth_origin_size, size=(256, 256), mode='bilinear')
                else:
                    fake_out_clip_mouth = fake_out_clip_mouth_origin_size

                sync_score = net_lipsync(fake_out_clip_mouth, deep_speech_full)
                loss_sync = criterionMSE(sync_score, torch.tensor(1.0).expand_as(sync_score).to(device_id)) * opt.lamb_syncnet_perception

                # -----------------MSE损失计算部分----------------- #
                loss_img = criterionMSE(fake_out, source_clip) * opt.lambda_img

                loss_g = (loss_img + loss_g_perception + loss_g_dI + loss_g_dV + loss_sync)
            scaler.scale(loss_g).backward()
            scaler.step(optimizer_g)
            scaler.update()

            # 记录到WandB
            if rank == 0 and iteration % opt.freq_wandb == 0:
                log_to_wandb(source_clip, source_clip_mask ,fake_out)
                wandb.log({
                    "epoch": epoch, 
                    "loss_dI": loss_dI.item(), 
                    "loss_dV": loss_dV.item(), 
                    "loss_g": loss_g.item(), 
                    "loss_img": loss_img.item(), 
                    "loss_g_perception": loss_g_perception.item(), 
                    "loss_g_dI": loss_g_dI.item(), 
                    "loss_g_dV": loss_g_dV.item(), 
                    "loss_sync": loss_sync.item(),
                    "learning_rate_g": optimizer_g.param_groups[0]["lr"],
                })
                print(
                    f"Epoch {epoch}, Iteration {iteration}, "
                    f"loss_dI: {loss_dI.item()}, loss_dV: {loss_dV.item()}, "
                    f"loss_g: {loss_g.item()}, loss_img: {loss_img.item()}, "
                    f"loss_g_perception: {loss_g_perception.item()}, "
                    f"loss_g_dI: {loss_g_dI.item()}, loss_g_dV: {loss_g_dV.item()}, "
                    f"loss_sync: {loss_sync.item()}",
                    f"learning_rate_g: {optimizer_g.param_groups[0]['lr']}"
                )

        # 更新学习率
        update_learning_rate(net_g_scheduler, optimizer_g)
        update_learning_rate(net_dI_scheduler, optimizer_dI)
        update_learning_rate(net_dV_scheduler, optimizer_dV)

        # dist.barrier()


        if rank == 0:
            # 保存和加载模型的代码
            if epoch % opt.checkpoint == 0 or epoch == opt.non_decay + opt.decay:
                save_checkpoint(epoch, opt, net_g, net_dI, net_dV, optimizer_g, optimizer_dI, optimizer_dV, net_g_scheduler, net_dI_scheduler, net_dV_scheduler)
            if epoch == 1:
                config_dict = vars(opt)
                config_out_path = os.path.join(opt.result_path, f'config_{args.name}.yaml')
                save_config_to_yaml(config_dict, config_out_path)

# 检查点保存
def save_checkpoint(epoch, 
                    opt, 
                    net_g, 
                    net_dI, 
                    net_dV, 
                    optimizer_g, 
                    optimizer_dI, 
                    optimizer_dV, 
                    scheduler_g, 
                    scheduler_dI, 
                    scheduler_dV):
    model_out_path = os.path.join(opt.result_path, f'checkpoint_epoch_{epoch}.pth')
    states = {
        'epoch': epoch+1,
        'state_dict': {
            'net_g': net_g.state_dict(),
            'net_dI': net_dI.state_dict(),
            'net_dV': net_dV.state_dict()
        },
        'optimizer': {
            'net_g': optimizer_g.state_dict(),
            'net_dI': optimizer_dI.state_dict(),
            'net_dV': optimizer_dV.state_dict()
        },
        'scheduler': {
            'net_g': scheduler_g.state_dict(),
            'net_dI': scheduler_dI.state_dict(),
            'net_dV': scheduler_dV.state_dict()
        }
    }
    torch.save(states, model_out_path)
    print(f"Checkpoint saved to {model_out_path}")



def cleanup():
    dist.destroy_process_group()

if __name__ == "__main__":
    # 解析配置文件路径
    config_parser = argparse.ArgumentParser(description="Train lawdNet clip model", add_help=False)
    config_parser.add_argument('--config_path', type=str, required=True, help="Path to the experiment configuration file.")
    # 本次实验的名称 wandb
    config_parser.add_argument('--name', type=str, required=True, help="Name of the experiment.")
    # 添加主节点地址参数
    config_parser.add_argument('--master_addr', type=str, default='localhost', help="Address of the master node for distributed training.")

    # 添加主节点端口参数
    config_parser.add_argument('--master_port', type=str, default='12355', help="Port of the master node for distributed training.")
    
    dist.init_process_group("nccl",timeout=datetime.timedelta(minutes=30))
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    args, remaining_argv = config_parser.parse_known_args()

    # After extracting config_path, use it to load configurations
    # import pdb; pdb.set_trace()
    init_wandb(args.name, rank)

    opt = load_config_and_device(args, rank)

    os.makedirs(opt.result_path, exist_ok=True)

    training_data_loader, train_sampler = load_training_data(opt, world_size, rank)

    net_g, net_dI, net_dV, net_vgg, net_lipsync = init_networks(opt, rank)

    optimizer_g, optimizer_dI, optimizer_dV = setup_optimizers(net_g, net_dI, net_dV)

    net_g_scheduler, net_dI_scheduler, net_dV_scheduler = setup_schedulers(optimizer_g, optimizer_dI, optimizer_dV)

    criterionGAN, criterionL1, criterionMSE = setup_criterion()

    load_pretrained_weights(net_g, net_dI, net_dV, optimizer_g, optimizer_dI, optimizer_dV, net_g_scheduler, net_dI_scheduler, net_dV_scheduler, opt, args)

    train(
        opt, 
        net_g, 
        net_dI, 
        net_dV, 
        training_data_loader, 
        train_sampler, 
        rank,
        optimizer_g, 
        optimizer_dI, 
        optimizer_dV, 
        criterionGAN, 
        criterionL1, 
        criterionMSE, 
        net_g_scheduler, 
        net_dI_scheduler, 
        net_dV_scheduler
    )

    cleanup()

