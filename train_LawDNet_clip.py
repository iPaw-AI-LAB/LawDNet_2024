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
from models.Gaussian_blur import Gaussian_bluring
from tensor_processing import SmoothSqMask
from tensor_processing_deng import concat_ref_and_src
from models.content_model import AudioContentModel, LipContentModel
from torch.nn.utils import clip_grad_norm_

# 冻结BN层
def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def replace_images(fake_out, source_clip):
    '''
    input: 
    output: fakeout 的 随机 0~4 张图被 source_clip 的 0~4 张图替换
    '''
    # 将 fake_out 和 source_clip 克隆为新的张量
    fake_out_clone = fake_out.clone()
    source_clip_clone = source_clip.clone()

    # 随机选择 0~4 张图的索引
    num_replace = random.randint(0, 4)
    indices = random.sample(range(5), num_replace)

    # 将选中的张图替换到 fake_out 中
    for idx in indices:
        fake_out_clone[idx, :, :, :] = source_clip_clone[idx, :, :, :]

    # 返回修改后的张量
    return fake_out_clone

# 初始化和登录WandB
def init_wandb(name):
    wandb.login()
    run = wandb.init(project=name)

# 加载配置和设备设置
def load_config_and_device(args):
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
    wandb.config.update(opt)  # 如果使用 wandb，可以这样更新配置
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda_devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return opt, device

# Convert Namespace to dictionary
def namespace_to_dict(namespace):
    return vars(namespace)

# Save configuration to a YAML file
def save_config_to_yaml(config, filename):
    with open(filename, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

# 加载训练数据
def load_training_data(opt):
    train_data = DINetDataset(opt.train_data, opt.augment_num, opt.mouth_region_size)
    training_data_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    return training_data_loader

# 初始化网络
def init_networks(opt):
    net_g = LawDNet(opt.source_channel, opt.ref_channel, opt.audio_channel).cuda()
    net_dI = Discriminator(opt.source_channel, opt.D_block_expansion, opt.D_num_blocks, opt.D_max_features).cuda()
    net_dV = Discriminator(opt.source_channel * 5, opt.D_block_expansion, opt.D_num_blocks, opt.D_max_features).cuda()
    net_vgg = Vgg19().cuda()
    net_lipsync = SyncNetPerception(opt.pretrained_syncnet_path).cuda()

    device_ids = [0, 1, 2, 3]
    net_g = nn.DataParallel(net_g, device_ids=device_ids).to(device)
    net_dI = nn.DataParallel(net_dI, device_ids=device_ids).to(device)
    net_dV = nn.DataParallel(net_dV, device_ids=device_ids).to(device)
    net_vgg = nn.DataParallel(net_vgg, device_ids=device_ids).to(device)
    net_lipsync = nn.DataParallel(net_lipsync, device_ids=device_ids).to(device)

    net_g = convert_model(net_g)
    return net_g, net_dI, net_dV, net_vgg, net_lipsync

# 设置优化器
def setup_optimizers(net_g, net_dI, net_dV):
    optimizer_g = optim.AdamW(net_g.parameters(), lr=opt.lr_g)
    optimizer_dI = optim.AdamW(net_dI.parameters(), lr=opt.lr_dI)
    optimizer_dV = optim.AdamW(net_dV.parameters(), lr=opt.lr_dI)
    return optimizer_g, optimizer_dI, optimizer_dV

# 设置损失函数
def setup_criterion():
    criterionGAN = GANLoss().cuda()
    criterionL1 = nn.L1Loss().cuda()
    criterionMSE = nn.MSELoss().cuda()
    criterionCosine = nn.CosineEmbeddingLoss().cuda()
    return criterionGAN, criterionL1, criterionMSE, criterionCosine

# 设置学习率调度器
def setup_schedulers(optimizer_g, optimizer_dI, optimizer_dV):
    net_g_scheduler = get_scheduler(optimizer_g, opt.non_decay, opt.decay)
    net_dI_scheduler = get_scheduler(optimizer_dI, opt.non_decay, opt.decay)
    net_dV_scheduler = get_scheduler(optimizer_dV, opt.non_decay, opt.decay)
    return net_g_scheduler, net_dI_scheduler, net_dV_scheduler

def log_to_wandb(source_clip, fake_out):
    # 可视化原始source_clip
    images_source = [wandb.Image(source_clip[i].cpu(), caption=f"Source Clip {i}") for i in range(source_clip.shape[0])]
    wandb.log({"Source Clips": images_source})
    
    # 可视化fake_out
    images_fake_out = [wandb.Image(fake_out[i].cpu(), caption=f"Fake Out {i}") for i in range(fake_out.shape[0])]
    wandb.log({"Fake Outs": images_fake_out})



# 训练过程
def train(opt, net_g, net_dI, net_dV, training_data_loader, optimizer_g, optimizer_dI, optimizer_dV, criterionGAN, criterionL1, criterionMSE, criterionCosine, net_g_scheduler, net_dI_scheduler, net_dV_scheduler):
    # 保存opt参数设置到本地供检查,加时间戳
    config_dict = namespace_to_dict(opt)
    config_out_path = os.path.join(opt.result_path, f'config_{time.strftime("%Y-%m-%d-%H-%M-%S")}.yaml')
    save_config_to_yaml(config_dict, config_out_path)
    
    smooth_sqmask = SmoothSqMask().cuda()
    for epoch in range(opt.start_epoch, opt.non_decay + opt.decay + 1):
        net_g.train()
        for iteration, data in tqdm(enumerate(training_data_loader)):
            source_clip, reference_clip, deep_speech_clip, deep_speech_full, flag = data

            # 检查是否有脏数据
            if not (flag.equal(torch.ones(opt.batch_size, 1, device='cuda'))):
                print("跳过含有脏数据的批次")
                continue
            
            source_clip = torch.cat(torch.split(source_clip, 1, dim=1), 0).squeeze(1).float().cuda()
            reference_clip = torch.cat(torch.split(reference_clip, 1, dim=1), 0).squeeze(1).float().cuda()
            deep_speech_clip = torch.cat(torch.split(deep_speech_clip, 1, dim=1), 0).squeeze(1).float().cuda()
            deep_speech_full = deep_speech_full.float().cuda()

            # 生成mask
            source_clip_mask = smooth_sqmask(source_clip)

            with autocast(enabled=True):
                fake_out = net_g(source_clip_mask, reference_clip, deep_speech_clip)

            fake_out_half = F.avg_pool2d(fake_out, 3, 2, 1, count_include_pad=False)
            source_clip_half = F.interpolate(source_clip, scale_factor=0.5, mode='bilinear')

            # 更新判别器DI
            optimizer_dI.zero_grad()
            with autocast(enabled=True):
                _, pred_fake_dI = net_dI(fake_out.detach())
                loss_dI_fake = criterionGAN(pred_fake_dI, False)
                _, pred_real_dI = net_dI(source_clip)
                loss_dI_real = criterionGAN(pred_real_dI, True)
                loss_dI = (loss_dI_fake + loss_dI_real) * 0.5
            scaler.scale(loss_dI).backward()
            scaler.step(optimizer_dI)

            # 更新判别器DV
            optimizer_dV.zero_grad()
            with autocast(enabled=True):
                condition_fake_dV = torch.cat(torch.split(fake_out.detach(), opt.batch_size, dim=0), 1)
                _, pred_fake_dV = net_dV(condition_fake_dV)
                loss_dV_fake = criterionGAN(pred_fake_dV, False)
                condition_real_dV = torch.cat(torch.split(source_clip, opt.batch_size, dim=0), 1)
                _, pred_real_dV = net_dV(condition_real_dV)
                loss_dV_real = criterionGAN(pred_real_dV, True)
                loss_dV = (loss_dV_fake + loss_dV_real) * 0.5
            scaler.scale(loss_dV).backward()
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
                loss_g_perception = sum([criterionL1(perception_fake[i], perception_real[i]) + criterionL1(perception_fake_half[i], perception_real_half[i]) for i in range(len(perception_real))]) / (len(perception_real) * 2)

                # -----------------GAN损失计算----------------- #
                loss_g_dI = criterionGAN(pred_fake_dI, True)
                loss_g_dV = criterionGAN(pred_fake_dV, True)

                # -----------------唇形同步损失计算----------------- #
                fake_out_clip = torch.cat(torch.split(fake_out, opt.batch_size, dim=0), 1)
                # 假定mouth_region_size定义了唇部区域的大小，并在train_data中已正确设置
                mouth_region_size = opt.mouth_region_size
                radius = mouth_region_size // 2
                radius_1_4 = radius // 4

                # 计算口部区域的起始和结束索引
                start_x, start_y = radius, radius_1_4
                end_x, end_y = start_x + mouth_region_size, start_y + mouth_region_size

                fake_out_clip_mouth_origin_size = fake_out_clip[:, :, start_x:end_x, start_y:end_y]

                # 将唇形部分调整到256x256，适应lip-sync网络
                if mouth_region_size != 256:
                    fake_out_clip_mouth = F.interpolate(fake_out_clip_mouth_origin_size, size=(256, 256), mode='bilinear')
                else:
                    fake_out_clip_mouth = fake_out_clip_mouth_origin_size

                sync_score = net_lipsync(fake_out_clip_mouth, deep_speech_full)
                loss_sync = criterionMSE(sync_score, torch.tensor(1.0).expand_as(sync_score).cuda())

                # -----------------MSE损失计算部分----------------- #
                loss_img = criterionMSE(fake_out, source_clip)

                loss_g = (loss_img * opt.lambda_img + loss_g_perception * opt.lamb_perception + loss_g_dI * opt.lambda_g_dI + loss_g_dV * opt.lambda_g_dV + loss_sync * opt.lamb_syncnet_perception)
            scaler.scale(loss_g).backward()
            scaler.step(optimizer_g)
            scaler.update()

            # 记录到WandB
            if iteration % opt.freq_wandb == 0:
                log_to_wandb(source_clip,  fake_out)
                wandb.log({"epoch": epoch, "loss_dI": loss_dI.item(), "loss_dV": loss_dV.item(), "loss_g": loss_g.item()})
        
        # 更新学习率
        update_learning_rate(net_g_scheduler, optimizer_g)
        update_learning_rate(net_dI_scheduler, optimizer_dI)
        update_learning_rate(net_dV_scheduler, optimizer_dV)

        if epoch % opt.checkpoint_interval == 0:
            save_checkpoint(epoch, opt, net_g, net_dI, net_dV, optimizer_g, optimizer_dI, optimizer_dV)



# 检查点保存
def save_checkpoint(epoch, opt, net_g, net_dI, net_dV, optimizer_g, optimizer_dI, optimizer_dV):
    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path, exist_ok=True)
    model_out_path = os.path.join(opt.result_path, f'netG_model_epoch_{epoch}.pth')
    states = {
        'epoch': epoch + 1,
        'state_dict': {
            'net_g': net_g.state_dict(),
            'net_dI': net_dI.state_dict(),
            'net_dV': net_dV.state_dict()
        },
        'optimizer': {
            'net_g': optimizer_g.state_dict(),
            'net_dI': optimizer_dI.state_dict(),
            'net_dV': optimizer_dV.state_dict()
        }
    }
    torch.save(states, model_out_path)
    print(f"Checkpoint saved to {model_out_path}")


# 主函数
if __name__ == "__main__":

    # 解析配置文件路径
    config_parser = argparse.ArgumentParser(description="Train lawdNet clip model", add_help=False)
    config_parser.add_argument('--config_path', type=str, required=True, help="Path to the experiment configuration file.")
    # 本次实验的名称 wandb
    config_parser.add_argument('--name', type=str, required=True, help="Name of the experiment.")

    args, remaining_argv = config_parser.parse_known_args()

    # After extracting config_path, use it to load configurations
    # import pdb; pdb.set_trace()
    init_wandb(args.name)
    opt, device = load_config_and_device(args)

    training_data_loader = load_training_data(opt)

    net_g, net_dI, net_dV, net_vgg, net_lipsync = init_networks(opt)

    optimizer_g, optimizer_dI, optimizer_dV = setup_optimizers(net_g, net_dI, net_dV)

    criterionGAN, criterionL1, criterionMSE, criterionCosine = setup_criterion()

    net_g_scheduler, net_dI_scheduler, net_dV_scheduler = setup_schedulers(optimizer_g, optimizer_dI, optimizer_dV)

    train(opt, net_g, net_dI, net_dV, training_data_loader, optimizer_g, optimizer_dI, optimizer_dV, criterionGAN, criterionL1, criterionMSE, criterionCosine, net_g_scheduler, net_dI_scheduler, net_dV_scheduler)

    wandb.finish()
