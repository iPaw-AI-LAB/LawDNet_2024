import cv2
import random
import numpy as np
import os
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast as autocast
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
import argparse
# import time
import yaml
from tqdm import tqdm

from models.Discriminator import Discriminator
from models.VGG19 import Vgg19
from models.LawDNet import LawDNet
from utils.training_utils import get_scheduler, update_learning_rate, GANLoss
from torch.utils.data import DataLoader
from dataset.dataset_DINet_frame import DINetDataset
from sync_batchnorm import convert_model
from config.config import DINetTrainingOptions
from tensor_processing import SmoothSqMask
# from models.Gaussian_blur import Gaussian_bluring

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

def init_wandb(name, rank):
    if rank == 0:
        wandb.login()
        wandb.init(project=name)

def save_config_to_yaml(config, filename):
    with open(filename, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

# 其余初始化、配置加载和数据加载函数保持不变
def load_experiment_config(config_module_path):
    """动态加载指定的配置文件"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("config_module", config_module_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.experiment_config

def load_config_and_device(args,rank):
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
    '''加载配置和设置设备'''

    random.seed(opt.seed + rank)
    np.random.seed(opt.seed+ rank)
    torch.cuda.manual_seed_all(opt.seed+rank)
    torch.manual_seed(opt.seed+rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 根据实验名称直接修改文件夹名字
    path_parts = opt.result_path.rsplit('/', 1)
    # 在倒数第一个/之前插入本次实验的名字
    opt.result_path = f'{path_parts[0]}/{args.name}/{path_parts[1]}'

    return opt, device

# 初始化网络，改为使用DDP
def init_networks(opt, rank):
    net_g = LawDNet(opt.source_channel, opt.ref_channel, opt.audio_channel, 
                    opt.warp_layer_num, opt.num_kpoints, opt.coarse_grid_size).to(rank)
    net_dI = Discriminator(opt.source_channel, opt.D_block_expansion, opt.D_num_blocks, opt.D_max_features).to(rank)

    net_vgg = Vgg19().to(rank)

    # torch2.0 compile
    # net_g = torch.compile(net_g, mode="reduce-overhead")
    # net_dI = torch.compile(net_dI, mode="reduce-overhead")
    # net_vgg = torch.compile(net_vgg, mode="reduce-overhead").to(rank)
    net_g = convert_model(net_g)
    net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
    net_dI = DDP(net_dI, device_ids=[rank], find_unused_parameters=True)
    print("DDP initialized, rank: ", rank)
    # net_vgg 用于特征提取，一般情况下不需要DDP封装

    return net_g, net_dI, net_vgg

def setup_optimizers(net_g, net_dI, opt):
    '''设置网络的优化器'''
    optimizer_g = optim.AdamW(net_g.parameters(), lr=opt.lr_g)
    optimizer_dI = optim.AdamW(net_dI.parameters(), lr=opt.lr_dI)
    return optimizer_g, optimizer_dI

def load_coarse2fine_checkpoint(net_g, opt, args):
    """
    Loads a coarse2fine training checkpoint into the model if the coarse2fine option is set to True.

    Parameters:
    - net_g: the model into which the weights will be loaded.
    - opt: options object that must have `coarse2fine` flag and a `coarse_model_path`.

    Returns:
    - A boolean value indicating whether the checkpoint was loaded successfully.
    """

    if opt.coarse2fine and opt.coarse_model_path:
        # 根据实验名称直接修改文件夹名字
        path_parts = opt.coarse_model_path.rsplit('/', 2)
        # 在倒数第2个/之前插入本次实验的名字
        opt.coarse_model_path = f'{path_parts[0]}/{args.name}/{path_parts[1]}/{path_parts[2]}'
        try:
            print(f'Loading checkpoint for coarse2fine training from: {opt.coarse_model_path}')
            checkpoint = torch.load(opt.coarse_model_path)
            net_g.load_state_dict(checkpoint['state_dict']['net_g'])
            print('Checkpoint loaded successfully for coarse2fine training!')
            return True
        except Exception as e:
            print(f'Failed to load the checkpoint for coarse2fine training: {e}')
            return False
    else:
        # Prints a message if the coarse2fine flag is not set or path is empty.
        if not opt.coarse2fine:
            print("Coarse2fine training flag is not set.")
        if not opt.coarse_model_path:
            print("Path to the coarse model checkpoint is empty.")
        return False
    
# 设置损失函数
def setup_criterion():
    '''设置损失函数'''
    criterionGAN = GANLoss()
    criterionL1 = nn.L1Loss()
    criterionL2 = nn.MSELoss()
    return criterionGAN, criterionL1, criterionL2

# 设置学习率调度器
def setup_schedulers(optimizer_g, optimizer_dI, opt):
    '''设置学习率调度器'''
    net_g_scheduler = get_scheduler(optimizer_g, opt.non_decay, opt.decay)
    net_dI_scheduler = get_scheduler(optimizer_dI, opt.non_decay, opt.decay)
    return net_g_scheduler, net_dI_scheduler

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

def log_to_wandb(source_image_data, fake_out):
    """
    Log source images and generated images to WandB for visualization.
    
    Parameters:
    - source_image_data: The source images from the dataset.
    - fake_out: The generated images from the network.
    """
    # Log source images
    source_clip = source_image_data.float()  # 将数据转换为全精度
    fake_out = fake_out.float()        # 同上 

    source_images = []
    for i in range(source_image_data.shape[0]): 
        img = source_image_data[i].permute(1, 2, 0).detach().cpu().numpy()
        img = (img * 255).round().astype(np.uint8)
        source_images.append(wandb.Image(img, caption=f"Source {i}"))
    wandb.log({"Source Images": source_images})

    # Log generated (fake) images
    fake_images = []
    for i in range(fake_out.shape[0]): 
        img = fake_out[i].permute(1, 2, 0).detach().cpu().numpy()
        img = (img * 255).round().astype(np.uint8)
        fake_images.append(wandb.Image(img, caption=f"Fake {i}"))
    wandb.log({"Generated Images": fake_images})


# 训练函数中的主要修改
def train(
    opt, 
    args,
    net_g, 
    net_dI, 
    net_vgg, 
    training_data_loader, 
    train_sampler,
    rank,
    optimizer_g, 
    optimizer_dI, 
    criterionGAN, 
    criterionL1, 
    criterionL2, 
    net_g_scheduler, 
    net_dI_scheduler
):
    
    device_id = rank % torch.cuda.device_count()
    criterionGAN = criterionGAN.to(device_id)
    criterionL1 = criterionL1.to(device_id)
    criterionL2 = criterionL2.to(device_id)

    smooth_sqmask = SmoothSqMask(device=device_id).to(device_id)

    # 混合精度训练：Creates a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # 添加rank参数
    for epoch in range(opt.start_epoch, opt.non_decay + opt.decay):
        # 设置DistributedSampler
        train_sampler.set_epoch(epoch)
        net_g.train()
        for iteration, (source_image_data, reference_clip_data, deepspeech_feature, flag) in enumerate(tqdm(training_data_loader, desc=f"Epoch {epoch}")):
            flag = flag.to(device_id)
            if not (flag.equal(torch.ones(opt.batch_size, 1, device=device_id))):
                print("Skipping batch with dirty data")
                continue

            source_image_data = source_image_data.float().to(device_id)
            reference_clip_data = reference_clip_data.float().to(device_id)
            deepspeech_feature = deepspeech_feature.float().to(device_id)

            # Apply soft square mask
            source_image_mask = smooth_sqmask(source_image_data)

            with autocast(enabled=True):
                fake_out = net_g(source_image_mask, reference_clip_data, deepspeech_feature)
            # Downsample the output and target for discriminator
            fake_out_half = F.avg_pool2d(fake_out, 3, 2, 1, count_include_pad=False)
            source_image_half = F.interpolate(source_image_data, scale_factor=0.5, mode='bilinear')

            # Update discriminator DI
            optimizer_dI.zero_grad()
            with autocast(enabled=True):
                _, pred_fake_dI = net_dI(fake_out.detach())
                loss_dI_fake = criterionGAN(pred_fake_dI, False)
                _, pred_real_dI = net_dI(source_image_data)
                loss_dI_real = criterionGAN(pred_real_dI, True)
                loss_dI = (loss_dI_fake + loss_dI_real) * 0.5
            scaler.scale(loss_dI).backward(retain_graph=True)
            scaler.step(optimizer_dI)

            # Update generator G
            optimizer_g.zero_grad()
            with autocast(enabled=True):
                _, pred_fake_dI = net_dI(fake_out)
                perception_real = net_vgg(source_image_data)
                perception_fake = net_vgg(fake_out)
                perception_real_half = net_vgg(source_image_half)
                perception_fake_half = net_vgg(fake_out_half)

                # Calculate perception loss
                loss_g_perception = 0
                for i in range(len(perception_real)):
                    loss_g_perception += criterionL1(perception_fake[i], perception_real[i])
                    loss_g_perception += criterionL1(perception_fake_half[i], perception_real_half[i])
                loss_g_perception = (loss_g_perception / (len(perception_real) * 2)) * opt.lamb_perception

                # Calculate GAN loss
                loss_g_dI = criterionGAN(pred_fake_dI, True) * opt.lambda_g_dI

                # Calculate MSE loss
                loss_img = criterionL2(fake_out, source_image_data) * opt.lambda_img

                loss_g = loss_img  + loss_g_perception + loss_g_dI 
            scaler.scale(loss_g).backward()
            scaler.step(optimizer_g)
            # 混合精度训练
            scaler.update()

            # dist.barrier()

            if rank == 0:
                if iteration % opt.freq_wandb == 0:
                    log_to_wandb(source_image_data, fake_out)
                    wandb.log({
                        "epoch": epoch,
                        "loss_dI": loss_dI.item(),
                        "loss_g": loss_g.item(),
                        "loss_img": loss_img.item(),
                        "loss_g_perception": loss_g_perception.item(),
                        "loss_g_dI": loss_g_dI.item(),
                        "learning_rate_g": optimizer_g.param_groups[0]["lr"],
                    })
                    print(f"\nEpoch {epoch}, iteration {iteration}, loss_dI: {loss_dI.item()}, loss_g: {loss_g.item()}, "
                        f"loss_img: {loss_img.item()}, loss_g_perception: {loss_g_perception.item()}, "
                        f"loss_g_dI: {loss_g_dI.item()}, learning_rate_g: {optimizer_g.param_groups[0]['lr']}")

        # Update learning rates
        update_learning_rate(net_g_scheduler, optimizer_g)
        update_learning_rate(net_dI_scheduler, optimizer_dI)

        # 只在主进程中记录到WandB
        if rank == 0 :
            # Save checkpoint
            if epoch % opt.checkpoint == 0:
                save_checkpoint(epoch, opt, net_g, net_dI, optimizer_g, optimizer_dI)
            if epoch == 1:
                config_dict = vars(opt)
                config_out_path = os.path.join(opt.result_path, f'config_{args.name}.yaml')
                save_config_to_yaml(config_dict, config_out_path)


def save_checkpoint(epoch, opt, net_g, net_dI, optimizer_g, optimizer_dI):

    model_out_path = os.path.join(opt.result_path, f'netG_model_epoch_{epoch}.pth')
    states = {
        'epoch': epoch + 1,
        'state_dict': {
            'net_g': net_g.state_dict(),
            'net_dI': net_dI.state_dict(),
        },
        'optimizer': {
            'net_g': optimizer_g.state_dict(),
            'net_dI': optimizer_dI.state_dict(),
        }
    }
    torch.save(states, model_out_path)
    print(f"Checkpoint saved to {model_out_path}")

def cleanup():
    dist.destroy_process_group()



if __name__ == "__main__":
    #  解析命令行参数导入特定配置文件
    # Step 1: Parse --config_path first
    config_parser = argparse.ArgumentParser(description="Train lawdNet frame model", add_help=False)
    config_parser.add_argument('--config_path', type=str, required=True, help="Path to the experiment configuration file.")
    config_parser.add_argument('--name', type=str, required=True, help="Name of the experiment.")
    # 添加主节点地址参数
    config_parser.add_argument('--master_addr', type=str, default='localhost', help="Address of the master node for distributed training.")

    # 添加主节点端口参数
    config_parser.add_argument('--master_port', type=str, default='12355', help="Port of the master node for distributed training.")

    args, remaining_argv = config_parser.parse_known_args()

    # import pdb; pdb.set_trace()
    # rank = int(os.environ["LOCAL_RANK"])
    # world_size = int(os.environ["WORLD_SIZE"])
    # print(f"Rank: {rank}, World Size: {world_size}")
    # print(f"Master Addr: {args.master_addr}, Master Port: {args.master_port}")
    # setup(rank, world_size, master_addr=args.master_addr, master_port=args.master_port)
    # print(f"Rank {rank} initialized.")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    init_wandb(args.name, rank)

    opt, device = load_config_and_device(args,rank)

    os.makedirs(opt.result_path, exist_ok=True)

    training_data_loader, train_sampler = load_training_data(opt, world_size, rank)

    net_g, net_dI, net_vgg = init_networks(opt, rank)

    optimizer_g, optimizer_dI = setup_optimizers(net_g, net_dI, opt)

    load_coarse2fine_checkpoint(net_g, opt, args)

    criterionGAN, criterionL1, criterionL2 = setup_criterion()

    net_g_scheduler, net_dI_scheduler = setup_schedulers(optimizer_g, optimizer_dI, opt)

    train(
        opt, 
        args,
        net_g, 
        net_dI, 
        net_vgg, 
        training_data_loader, 
        train_sampler,
        rank,
        optimizer_g, 
        optimizer_dI, 
        criterionGAN, 
        criterionL1, 
        criterionL2, 
        net_g_scheduler, 
        net_dI_scheduler
    )
    wandb.finish()

    cleanup()

    # time.sleep(5)  