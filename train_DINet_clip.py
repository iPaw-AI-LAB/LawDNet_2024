import sys
import cv2
import logging
import random

from models.Discriminator import Discriminator
from models.VGG19 import Vgg19

from models.LawDNet_new import LawDNet
from models.Syncnet import SyncNetPerception
# 边缘检测的loss
# from models.EdgeDetector import Sobel_Edge_Detection

from utils.training_utils import get_scheduler, update_learning_rate, GANLoss
from config.config import DINetTrainingOptions
from sync_batchnorm import convert_model
from torch.utils.data import DataLoader
from dataset.dataset_DINet_clip import DINetDataset

from models.Gaussian_blur import Gaussian_bluring
from tensor_processing import SmoothSqMask
from tensor_processing_deng import concat_ref_and_src
from models.content_model import AudioContentModel,LipContentModel
from torch.nn.utils import clip_grad_norm_

# from models.Gaussian_blur import Gaussian_bluring
# from tqdm import tqdm
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.nn.functional as F
from tqdm import tqdm

import wandb
# from apex import amp
from torch.cuda.amp import autocast as autocast

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

if __name__ == "__main__":
    '''
            clip training code of DINet
            in the resolution you want, using clip training code after frame training
    '''
    ### 创建一个wandb实例 ###################################
    
    wandb.login()
    
    run = wandb.init(
    project="北科大-mouthsize=288-lowsize-60-kpoints=8-实验新1-复现效果",
    config={
        "training_type": "4 clip",
        "grid_size": "52 40",
    })

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("已设置 PyTorch 只使用第一张 GPU 卡")
    
    # load config
    opt = DINetTrainingOptions().parse_args()
    device_ids = [0,1,2,3]
    # 保存命令行参数到WandB
    wandb.config.update(opt)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    # load training data
    train_data = DINetDataset(opt.train_data,opt.augment_num,opt.mouth_region_size)
    training_data_loader = DataLoader(dataset=train_data,  batch_size=opt.batch_size, shuffle=True, drop_last=True)
    train_data_length = len(training_data_loader)
    # init network
    net_g = LawDNet(opt.source_channel,opt.ref_channel,opt.audio_channel).cuda()

    net_dI = Discriminator(opt.source_channel ,opt.D_block_expansion, opt.D_num_blocks, opt.D_max_features).cuda()
    net_dV = Discriminator(opt.source_channel * 5, opt.D_block_expansion, opt.D_num_blocks, opt.D_max_features).cuda()
    net_vgg = Vgg19().cuda()
    net_lipsync = SyncNetPerception(opt.pretrained_syncnet_path).cuda()
    # net_lip_content_model = LipContentModel().cuda()
    # net_audio_content_model = AudioContentModel().cuda()

    # parallel
    net_g = nn.DataParallel(net_g, device_ids=device_ids)
    net_g = convert_model(net_g)
    net_dI = nn.DataParallel(net_dI, device_ids=device_ids)
    net_dV = nn.DataParallel(net_dV, device_ids=device_ids)
    net_vgg = nn.DataParallel(net_vgg, device_ids=device_ids)
    # net_lip_content_model = nn.DataParallel(net_lip_content_model, device_ids=device_ids)
    # net_audio_content_model = nn.DataParallel(net_audio_content_model, device_ids=device_ids)
    
    # setup optimizer
    optimizer_g = optim.AdamW(net_g.parameters(), lr=opt.lr_g)
    optimizer_dI = optim.AdamW(net_dI.parameters(), lr=opt.lr_dI)
    optimizer_dV = optim.AdamW(net_dV.parameters(), lr=opt.lr_dI)
    # optimizer_lip_content_model = optim.AdamW(net_lip_content_model.parameters(), lr=opt.lr_g*0.1)
    # optimizer_audio_content_model = optim.AdamW(net_audio_content_model.parameters(), lr=opt.lr_g*0.1)

    # 混合精度训练：Creates a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    ##### load frame trained DInet weight
    if opt.pretrained_frame_DINet_path != '':
        print('loading frame trained DINet weight from: {}'.format(opt.pretrained_frame_DINet_path))
        checkpoint = torch.load(opt.pretrained_frame_DINet_path) #origin
        net_g.load_state_dict(checkpoint['state_dict']['net_g'])
        print('loading frame trained DINet weight finished!')
    else:
        print("没有加载 前三步的 frame trained DINet weight")

    # set criterion
    criterionGAN = GANLoss().cuda()
    criterionL1 = nn.L1Loss().cuda()
    criterionMSE = nn.MSELoss().cuda()
    criterionCosine = nn.CosineEmbeddingLoss().cuda()
    
    # set scheduler
    net_g_scheduler = get_scheduler(optimizer_g, opt.non_decay, opt.decay)
    net_dI_scheduler = get_scheduler(optimizer_dI, opt.non_decay, opt.decay)
    net_dV_scheduler = get_scheduler(optimizer_dV, opt.non_decay, opt.decay)
    # net_lip_content_model_scheduler = get_scheduler(optimizer_lip_content_model, opt.non_decay, opt.decay)
    # net_audio_content_model_scheduler = get_scheduler(optimizer_audio_content_model, opt.non_decay, opt.decay)
    
    # set label of syncnet perception loss
    real_tensor = torch.tensor(1.0).cuda()
    
    # set gaussian bluring
    mask_gaussian = Gaussian_bluring(radius=30,sigma=10,padding='same')
    smooth_sqmask = SmoothSqMask().cuda()
    mouth_region_size = opt.mouth_region_size
    radius = mouth_region_size//2
    radius_1_4 = radius//4
    
    # start train
    for epoch in range(opt.start_epoch, opt.non_decay+opt.decay+1):
        net_g.train()
        for iteration, data in tqdm(enumerate(training_data_loader)):
            
            source_clip, _, reference_clip,deep_speech_clip,deep_speech_full,reference_continual,flag = data

            # 是否存在脏数据
            if not (flag.equal(torch.ones(opt.batch_size,1,device='cuda'))):
                print("flag跳过训练",flag)
                continue

            source_clip = torch.cat(torch.split(source_clip, 1, dim=1), 0).squeeze(1).float().cuda()
            reference_clip = torch.cat(torch.split(reference_clip, 1, dim=1), 0).squeeze(1).float().cuda()
            deep_speech_clip = torch.cat(torch.split(deep_speech_clip, 1, dim=1), 0).squeeze(1).float().cuda()
            reference_continual = torch.cat(torch.split(reference_continual, 1, dim=1), 0).squeeze(1).float().cuda()
            deep_speech_full = deep_speech_full.float().cuda()
            
            source_clip_mask = smooth_sqmask(source_clip)

            if iteration % opt.freq_wandb == 0:
                images = []
                for i in range(source_clip.shape[0]):
                    img = source_clip[i].permute(1, 2, 0).detach().cpu().numpy()
                    img = (img * 255).round().astype(np.uint8)
                    images.append(img)
                merged_img = np.concatenate(images, axis=1)
                wandb.log({f"DINet-source_GT": wandb.Image(merged_img)}) 
                # cv2.imwrite('4_DINet-source_GT-out.jpg', merged_img[:,:,::-1]) 

            if iteration % opt.freq_wandb == 0:
                images = []
                for i in range(source_clip_mask.shape[0]):
                    img = source_clip_mask[i].permute(1, 2, 0).detach().cpu().numpy()
                    img = (img * 255).round().astype(np.uint8)
                    images.append(img)
                merged_img = np.concatenate(images, axis=1)
                wandb.log({f"DINet-source_input_with_mask": wandb.Image(merged_img)}) 
                # cv2.imwrite('4_DINet-source_input_with_mask.jpg', merged_img[:,:,::-1]) 

            with autocast(enabled=True): # 混合精度训练
                fake_out = net_g(source_clip_mask,reference_clip,deep_speech_clip)

            if iteration % opt.freq_wandb == 0:
                images = []
                for i in range(fake_out.shape[0]):
                    img = fake_out[i].permute(1, 2, 0).detach().cpu().numpy()
                    img = (img * 255).round().astype(np.uint8)   
                    images.append(img)       
                merged_img = np.concatenate(images, axis=1)
                wandb.log({f"fake-out": wandb.Image(merged_img)}) # rgb
                # cv2.imwrite('4_merged_images_fake-out.jpg', merged_img[:,:,::-1]) # bgr

            fake_out_half = F.avg_pool2d(fake_out, 3, 2, 1, count_include_pad=False)
            source_clip_half = F.interpolate(source_clip, scale_factor=0.5, mode='bilinear')
            # (1) Update DI network
            optimizer_dI.zero_grad()
            # 混合精度训练
            with autocast(enabled=True):
                _,pred_fake_dI = net_dI(fake_out)
                loss_dI_fake = criterionGAN(pred_fake_dI, False)
                _,pred_real_dI = net_dI(source_clip)
                loss_dI_real = criterionGAN(pred_real_dI, True)
                # Combined DI loss
                loss_dI = (loss_dI_fake + loss_dI_real) * 0.5

            scaler.scale(loss_dI).backward(retain_graph=True)
            scaler.step(optimizer_dI)

            # (2) Update DV network
            optimizer_dV.zero_grad()

            # 随机替换 fake_out 中的图像为 source_clip 中的图像
            if epoch > 1000:
                fake_out_random_replace = replace_images(fake_out, source_clip)
                condition_fake_dV = torch.cat(torch.split(fake_out_random_replace, opt.batch_size, dim=0), 1)
                # 仅用于Dv
                if iteration % opt.freq_wandb == 0:
                    images = []
                    for i in range(fake_out_random_replace.shape[0]):
                        img = fake_out_random_replace[i].permute(1, 2, 0).detach().cpu().numpy()
                        img = (img * 255).round().astype(np.uint8)   
                        images.append(img)       
                    merged_img = np.concatenate(images, axis=1)
                    wandb.log({f"fake_out_random_replace": wandb.Image(merged_img)}) # rgb
                    # cv2.imwrite('4_merged_images_fake-out.jpg', merged_img[:,:,::-1]) # bgr

            else:
                condition_fake_dV = torch.cat(torch.split(fake_out, opt.batch_size, dim=0), 1)
                # reference_continual_channel = torch.cat(torch.split(reference_continual, opt.batch_size, dim=0), 1)
            #### 插入参考帧
            # condition_fake_dV = concat_ref_and_src(condition_fake_dV, reference_continual_channel)

            # if iteration % opt.freq_wandb == 0:
            #     images = []
            #     for i in range(reference_continual.shape[0]):
            #         img = reference_continual[i].permute(1, 2, 0).detach().cpu().numpy()
            #         img = (img * 255).round().astype(np.uint8)   
            #         images.append(img)       
            #     merged_img = np.concatenate(images, axis=1)
            #     wandb.log({f"reference_continual--参考帧送入判别器V": wandb.Image(merged_img)}) # rgb
            #     # cv2.imwrite('4_merged_images_fake-out.jpg', merged_img[:,:,::-1]) # bgr


            # 混合精度训练
            with autocast(enabled=True):
                _, pred_fake_dV = net_dV(condition_fake_dV)
                loss_dV_fake = criterionGAN(pred_fake_dV, False)
                condition_real_dV = torch.cat(torch.split(source_clip, opt.batch_size, dim=0), 1)
                ###### 插入参考帧
                # condition_real_dV= concat_ref_and_src(condition_real_dV, reference_continual_channel)

                _, pred_real_dV = net_dV(condition_real_dV)
                loss_dV_real = criterionGAN(pred_real_dV, True)
                # Combined DV loss
                loss_dV = (loss_dV_fake + loss_dV_real) * 0.5

            scaler.scale(loss_dV).backward(retain_graph=True)
            scaler.step(optimizer_dV)

            # (2) Update LipContentModel & AudioContentModel
            # optimizer_lip_content_model.zero_grad()
            # optimizer_audio_content_model.zero_grad()
            # fake_out_clip = torch.cat(torch.split(fake_out, opt.batch_size, dim=0), 1)
            # fake_out_clip_mouth = fake_out_clip[:, :, train_data.radius:train_data.radius + train_data.mouth_region_size,
            # train_data.radius_1_4:train_data.radius_1_4 + train_data.mouth_region_size]

            # import pdb
            # pdb.set_trace()
            # lip_content   = net_lip_content_model(fake_out_clip_mouth)
            # audio_content = net_audio_content_model(deep_speech_full)  # 29 * 9 取中间5个求loss
            # cos_sim = F.cosine_similarity(lip_content, audio_content, dim=1)

            # # # 将余弦相似度作为损失函数
            # loss_content = 1 - cos_sim.mean()

            # lip_content   = net_lip_content_model(fake_out_clip_mouth)
            # loss_content = criterionMSE(lip_content,deep_speech_full[:,:,2:7])

            # 混合精度训练
            # scaler.scale(loss_content*opt.lambda_content).backward(retain_graph=True)
            # clip_grad_norm_(net_audio_content_model.parameters(), max_norm=20, norm_type=2)
            # clip_grad_norm_(net_lip_content_model.parameters(), max_norm=20, norm_type=2)
            # scaler.step(optimizer_lip_content_model)
            # scaler.step(optimizer_audio_content_model)

            # (3) Update DINet
            optimizer_g.zero_grad()
            # 混合精度训练
            with autocast(enabled=True):
                _, pred_fake_dI = net_dI(fake_out)
                _, pred_fake_dV = net_dV(condition_fake_dV)
                # compute perception loss
                perception_real = net_vgg(source_clip)
                perception_fake = net_vgg(fake_out)
                perception_real_half = net_vgg(source_clip_half)
                perception_fake_half = net_vgg(fake_out_half)
                loss_g_perception = 0
                for i in range(len(perception_real)):
                    loss_g_perception += criterionL1(perception_fake[i], perception_real[i])
                    loss_g_perception += criterionL1(perception_fake_half[i], perception_real_half[i])
                loss_g_perception = (loss_g_perception / (len(perception_real) * 2)) 
                # # gan dI loss
                loss_g_dI = criterionGAN(pred_fake_dI, True)
                # # gan dV loss
                loss_g_dV = criterionGAN(pred_fake_dV, True)
                ## sync perception loss

                fake_out_clip = torch.cat(torch.split(fake_out, opt.batch_size, dim=0), 1)
                fake_out_clip_mouth_origin_size = fake_out_clip[:, :, train_data.radius:train_data.radius + train_data.mouth_region_size,
                train_data.radius_1_4:train_data.radius_1_4 + train_data.mouth_region_size]

                ## 将唇形部分变为256*256: 因为 net_lipsync 是用256*256训练的
                if opt.mouth_region_size != 256:
                    fake_out_clip_mouth = F.interpolate(fake_out_clip_mouth_origin_size, size=(256,256), mode='bilinear')
                else:
                    fake_out_clip_mouth = fake_out_clip_mouth_origin_size
                    
                ### 唇形同步损失
                sync_score = net_lipsync(fake_out_clip_mouth, deep_speech_full)
                loss_sync = criterionMSE(sync_score, real_tensor.expand_as(sync_score)) 

                # 图像MSE损失
                loss_img = criterionMSE(fake_out, source_clip) 

                # combine all losses
                loss_g = (loss_img * opt.lambda_img + 
                          opt.lamb_perception * loss_g_perception + 
                          opt.lambda_g_dI * loss_g_dI + 
                          opt.lambda_g_dV * loss_g_dV + 
                          opt.lamb_syncnet_perception * loss_sync 
                          # opt.lambda_content * loss_content
                          )
                
            # 混合精度训练
            # scaler.scale(loss_content).backward(retain_graph=True)
            # clip_grad_norm_(net_audio_content_model.parameters(), max_norm=20, norm_type=2)
            # clip_grad_norm_(net_lip_content_model.parameters(), max_norm=20, norm_type=2)
            # scaler.step(optimizer_lip_content_model)
            # scaler.step(optimizer_audio_content_model)

            scaler.scale(loss_g).backward()
            scaler.step(optimizer_g)
            scaler.update()
            
            wandb.log({"epoch": epoch,  
                        "1:loss_dI": float(loss_dI),
                        str(opt.lambda_g_dI) + ":loss_g_dI": float(loss_g_dI*opt.lambda_g_dI),
                        "1:loss_dV": float(loss_dV),
                        str(opt.lambda_g_dV) + ":loss_g_dV": float(loss_g_dV*opt.lambda_g_dV),
                        str(opt.lamb_perception)+ ":loss_g_perception": float(loss_g_perception * opt.lamb_perception),
                        str(opt.lamb_syncnet_perception)+ ":loss_sync": float(loss_sync * opt.lamb_syncnet_perception),
                        str(opt.lambda_img) + ":loss_img": float(loss_img*opt.lambda_img),
                        # str(opt.lambda_content)+":loss_content": float(loss_content*opt.lambda_content),
                       }
                      )
            
        # update learning rate
        update_learning_rate(net_g_scheduler, optimizer_g)
        update_learning_rate(net_dI_scheduler, optimizer_dI)
        update_learning_rate(net_dV_scheduler, optimizer_dV)

        # update_learning_rate(net_lip_content_model_scheduler, optimizer_lip_content_model)

        # checkpoint
        if epoch %  opt.checkpoint == 0:
            if not os.path.exists(opt.result_path):
                os.mkdir(opt.result_path)
            model_out_path = os.path.join(opt.result_path, 'netG_model_epoch_{}.pth'.format(epoch))
            states = {
                'epoch': epoch + 1,
                'state_dict': {'net_g': net_g.state_dict(),'net_dI': net_dI.state_dict(),'net_dV': net_dV.state_dict()},
                'optimizer': {'net_g': optimizer_g.state_dict(), 'net_dI': optimizer_dI.state_dict(), 'net_dV': optimizer_dV.state_dict()}
            }
            torch.save(states, model_out_path)
            print("Checkpoint saved to {}".format(epoch))
            
    wandb.finish()