import cv2
import random
import numpy as np
import os
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast as autocast
import wandb
import argparse
import time
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
from models.Gaussian_blur import Gaussian_bluring


# 初始化和登录WandB
def init_wandb(name):
    '''初始化WandB并登录，设置项目和配置'''
    wandb.login()
    run = wandb.init(project=name)


def load_experiment_config(config_module_path):
    """动态加载指定的配置文件"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("config_module", config_module_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.experiment_config


def load_config_and_device(args):
    # 动态加载配置文件
    experiment_config = load_experiment_config(args.config_path)

    # 创建 opt 实例，这里避免 argparse 解析命令行参数
    opt = DINetTrainingOptions().parse_args()

    # 根据动态加载的配置更新 opt
    for key, value in experiment_config.items():
        if hasattr(opt, key):
            setattr(opt, key, value)

    # 假设 wandb 已经初始化
    wandb.config.update(opt)  # 如果使用 wandb，可以这样更新配置
    '''加载配置和设置设备'''
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 根据实验名称直接修改文件夹名字
    path_parts = opt.result_path.rsplit('/', 1)
    # 在倒数第一个/之前插入本次实验的名字
    opt.result_path = f'{path_parts[0]}/{args.name}/{path_parts[1]}'

    return opt, device

# 加载训练数据
def load_training_data(opt):
    '''根据配置加载训练数据'''
    train_data = DINetDataset(opt.train_data, opt.augment_num, opt.mouth_region_size)
    training_data_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    return training_data_loader

# 初始化网络
def init_networks(opt):
    '''初始化网络模型'''
    net_g = LawDNet(opt.source_channel, opt.ref_channel, opt.audio_channel, 
                    opt.warp_layer_num, opt.num_kpoints, opt.coarse_grid_size).cuda()
    net_dI = Discriminator(opt.source_channel, opt.D_block_expansion, opt.D_num_blocks, opt.D_max_features).cuda()
    net_vgg = Vgg19().cuda()

    device_ids = [int(x) for x in opt.cuda_devices.split(',')]
    net_g = nn.DataParallel(net_g, device_ids=device_ids).to(device)
    net_g = convert_model(net_g)
    net_dI = nn.DataParallel(net_dI, device_ids=device_ids).to(device)
    net_vgg = nn.DataParallel(net_vgg, device_ids=device_ids).to(device)

    return net_g, net_dI, net_vgg

# 设置优化器
def setup_optimizers(net_g, net_dI, opt):
    '''设置网络的优化器'''
    optimizer_g = optim.AdamW(net_g.parameters(), lr=opt.lr_g)
    optimizer_dI = optim.AdamW(net_dI.parameters(), lr=opt.lr_dI)
    return optimizer_g, optimizer_dI


def load_coarse2fine_checkpoint(net_g, opt):
    """
    Loads a coarse2fine training checkpoint into the model if the coarse2fine option is set to True.

    Parameters:
    - net_g: the model into which the weights will be loaded.
    - opt: options object that must have `coarse2fine` flag and a `coarse_model_path`.

    Returns:
    - A boolean value indicating whether the checkpoint was loaded successfully.
    """

    if opt.coarse2fine and opt.coarse_model_path:
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
    criterionGAN = GANLoss().cuda()
    criterionL1 = nn.L1Loss().cuda()
    criterionL2 = nn.MSELoss().cuda()
    return criterionGAN, criterionL1, criterionL2

# 设置学习率调度器
def setup_schedulers(optimizer_g, optimizer_dI, opt):
    '''设置学习率调度器'''
    net_g_scheduler = get_scheduler(optimizer_g, opt.non_decay, opt.decay)
    net_dI_scheduler = get_scheduler(optimizer_dI, opt.non_decay, opt.decay)
    return net_g_scheduler, net_dI_scheduler

def save_config_to_yaml(config, filename):
    with open(filename, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

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
    for i in range(source_image_data.shape[0]):  # Limit to first 5 images
        img = source_image_data[i].permute(1, 2, 0).detach().cpu().numpy()
        img = (img * 255).round().astype(np.uint8)
        source_images.append(wandb.Image(img, caption=f"Source {i}"))
    wandb.log({"Source Images": source_images})

    # Log generated (fake) images
    fake_images = []
    for i in range(fake_out.shape[0]):  # Limit to first 5 images
        img = fake_out[i].permute(1, 2, 0).detach().cpu().numpy()
        img = (img * 255).round().astype(np.uint8)
        fake_images.append(wandb.Image(img, caption=f"Fake {i}"))
    wandb.log({"Generated Images": fake_images})


def train(
    opt, 
    net_g, 
    net_dI, 
    net_vgg, 
    training_data_loader, 
    optimizer_g, 
    optimizer_dI, 
    criterionGAN, 
    criterionL1, 
    criterionL2, 
    net_g_scheduler, 
    net_dI_scheduler
):

    smooth_sqmask = SmoothSqMask().cuda()

    # 混合精度训练：Creates a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    for epoch in range(opt.start_epoch, opt.non_decay + opt.decay):
        net_g.train()
        for iteration, data in enumerate(tqdm(training_data_loader, desc=f"Epoch {epoch}")):
            source_image_data, reference_clip_data, deepspeech_feature, flag = data
            if not (flag.equal(torch.ones(opt.batch_size, 1, device='cuda'))):
                print("Skipping batch with dirty data")
                continue

            source_image_data = source_image_data.float().cuda()
            reference_clip_data = reference_clip_data.float().cuda()
            deepspeech_feature = deepspeech_feature.float().cuda()

            # Apply soft square mask
            source_image_mask = smooth_sqmask(source_image_data)

            # Forward pass through the generator
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
                loss_g_perception = sum([criterionL1(perception_fake[i], perception_real[i]) + criterionL1(perception_fake_half[i], perception_real_half[i]) for i in range(len(perception_real))]) / (len(perception_real) * 2)

                # Calculate GAN loss
                loss_g_dI = criterionGAN(pred_fake_dI, True)

                # Calculate MSE loss
                loss_img = criterionL2(fake_out, source_image_data)

                loss_g = loss_img * opt.lambda_img + loss_g_perception * opt.lamb_perception + loss_g_dI * opt.lambda_g_dI
            scaler.scale(loss_g).backward()
            scaler.step(optimizer_g)
            # 混合精度训练
            scaler.update()

            if iteration % opt.freq_wandb == 0:
                log_to_wandb(source_image_data, fake_out)
                wandb.log({
                    "epoch": epoch,
                    "loss_dI": loss_dI.item(),
                    "loss_g": loss_g.item(),
                    "loss_img": loss_img.item(),
                    "loss_g_perception": loss_g_perception.item(),
                    "loss_g_dI": loss_g_dI.item()
                })
                print(f"\nEpoch {epoch}, iteration {iteration}, loss_dI: {loss_dI.item()}, loss_g: {loss_g.item()}, "
                    f"loss_img: {loss_img.item()}, loss_g_perception: {loss_g_perception.item()}, "
                    f"loss_g_dI: {loss_g_dI.item()}")

        # Update learning rates
        update_learning_rate(net_g_scheduler, optimizer_g)
        update_learning_rate(net_dI_scheduler, optimizer_dI)

        # Save checkpoint
        if epoch % opt.checkpoint == 0:
            save_checkpoint(epoch, opt, net_g, net_dI, optimizer_g, optimizer_dI)
        if epoch == 1:
            config_dict = vars(opt)
            config_out_path = os.path.join(opt.result_path, f'config_{time.strftime("%Y-%m-%d-%H-%M-%S")}.yaml')
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


# 主函数
if __name__ == "__main__":

    #  解析命令行参数导入特定配置文件
    # Step 1: Parse --config_path first
    config_parser = argparse.ArgumentParser(description="Train lawdNet frame model", add_help=False)
    config_parser.add_argument('--config_path', type=str, required=True, help="Path to the experiment configuration file.")
    config_parser.add_argument('--name', type=str, required=True, help="Name of the experiment.")
    args, remaining_argv = config_parser.parse_known_args()

    # After extracting config_path, use it to load configurations
    # import pdb; pdb.set_trace()
    init_wandb(args.name)

    opt, device = load_config_and_device(args)

    os.makedirs(opt.result_path, exist_ok=True)

    training_data_loader = load_training_data(opt)

    net_g, net_dI, net_vgg = init_networks(opt)

    optimizer_g, optimizer_dI = setup_optimizers(net_g, net_dI, opt)

    load_coarse2fine_checkpoint(net_g, opt)

    criterionGAN, criterionL1, criterionL2 = setup_criterion()

    net_g_scheduler, net_dI_scheduler = setup_schedulers(optimizer_g, optimizer_dI, opt)

    train(opt, net_g, net_dI, net_vgg, training_data_loader, optimizer_g, optimizer_dI, criterionGAN, criterionL1, criterionL2, net_g_scheduler, net_dI_scheduler)

    wandb.finish()
