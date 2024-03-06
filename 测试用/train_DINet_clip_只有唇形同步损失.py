import sys
import cv2
# print("sys.path",sys.path)
# sys.path.append('/data/dengjunli/DJL_workspace/DINet-2大数据集训练/DINet')
# sys.path.remove('/data/dengjunli/talking_lip-放在这里只是看的别跑')
# sys.path.remove('/data/dengjunli/talking_lip-放在这里只是看的别跑/wav2lip')
# sys.path.remove('/data/dengjunli/talking_lip-放在这里只是看的别跑/codeformer')
# sys.path.remove('/data/dengjunli/talking_lip') 
# sys.path.remove('/data/dengjunli/talking_lip/wav2lip') 
# sys.path.remove('/data/dengjunli/talking_lip/codeformer') 

# print("sys.path",sys.path)


import logging

from models.Discriminator import Discriminator
from models.VGG19 import Vgg19
from models.DINet import DINet
from models.Syncnet import SyncNetPerception
# dengjunli 加上边缘检测的loss
from models.EdgeDetector import Sobel_Edge_Detection

from utils.training_utils import get_scheduler, update_learning_rate, GANLoss
from config.config import DINetTrainingOptions
from sync_batchnorm import convert_model
from torch.utils.data import DataLoader
from dataset.dataset_DINet_clip import DINetDataset

from models.Gaussian_blur import Gaussian_bluring

from tqdm import tqdm

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.nn.functional as F

import wandb


if __name__ == "__main__":
    '''
            clip training code of DINet
            in the resolution you want, using clip training code after frame training
            
        '''
        
    ### 创建一个wandb实例 ###################################
    
    wandb.login()
    
    run = wandb.init(
    # Set the project where this run will be logged
    project="DINet-小数据集-测试syncnet有效性",
    # Track hyperparameters and run metadata
    config={
        "edge": "sobel",
    })
    
    ### 创建一个wandb实例 ###################################
        
    # load config
    opt = DINetTrainingOptions().parse_args()


    # 保存命令行参数到WandB
    wandb.config.update(opt)

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    # load training data
    train_data = DINetDataset(opt.train_data,opt.augment_num,opt.mouth_region_size)
    training_data_loader = DataLoader(dataset=train_data,  batch_size=opt.batch_size, shuffle=True,drop_last=True)
    train_data_length = len(training_data_loader)
    # init network
    net_g = DINet(opt.source_channel,opt.ref_channel,opt.audio_channel).cuda()
    # net_dI = Discriminator(opt.source_channel ,opt.D_block_expansion, opt.D_num_blocks, opt.D_max_features).cuda()
    # net_dV = Discriminator(opt.source_channel * 5, opt.D_block_expansion, opt.D_num_blocks, opt.D_max_features).cuda()
    # net_vgg = Vgg19().cuda()
    net_lipsync = SyncNetPerception(opt.pretrained_syncnet_path).cuda()
    ### 加一个实时检测边缘的sobel loss dengjunli
    # net_edge = Sobel_Edge_Detection().cuda()
    
    
    # DI 和 DV 分别是什么
    
    
    # # parallel
    net_g = nn.DataParallel(net_g)
    net_g = convert_model(net_g)
    # net_dI = nn.DataParallel(net_dI)
    # net_dV = nn.DataParallel(net_dV)
    # net_vgg = nn.DataParallel(net_vgg)
    ### 加一个实时检测边缘的sobel loss
    # net_edge = Sobel_Edge_Detection().cuda()
    
    
    # setup optimizer
    # optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr_g)
    # optimizer_dI = optim.Adam(net_dI.parameters(), lr=opt.lr_dI)
    # optimizer_dV = optim.Adam(net_dV.parameters(), lr=opt.lr_dI)
    
    ### 加一个实时检测边缘的sobel loss dengjunli
    # optimizer_edge = optim.Adam(net_dV.parameters(), lr=opt.lr_edge)
    
    
    
    # load frame trained DInet weight
    print('loading frame trained DINet weight from: {}'.format(opt.pretrained_frame_DINet_path))
    checkpoint = torch.load(opt.pretrained_frame_DINet_path)
    net_g.load_state_dict(checkpoint['state_dict']['net_g'])
    print('loading frame trained DINet weight finished!')

    # print("不加载预训练模型！！！！！！！！！")

    # set criterion
    criterionGAN = GANLoss().cuda()
    criterionL1 = nn.L1Loss().cuda()
    criterionMSE = nn.MSELoss().cuda()
    # set edge loss dengjunli 也用MSe
    
    
    # set scheduler
    # net_g_scheduler = get_scheduler(optimizer_g, opt.non_decay, opt.decay)
    # net_dI_scheduler = get_scheduler(optimizer_dI, opt.non_decay, opt.decay)
    # net_dV_scheduler = get_scheduler(optimizer_dV, opt.non_decay, opt.decay)
    # set edge scheduler dengjunli
    # net_edge_scheduler = get_scheduler(optimizer_edge, opt.non_decay, opt.decay)
    
    # set label of syncnet perception loss
    real_tensor = torch.tensor(1.0).cuda()
    
    # set gaussian bluring
    # g_bluring_whole_image = Gaussian_bluring(radius=3,sigma=3)
    # g_bluring_mouth_region = Gaussian_bluring(radius=3,sigma=5)
    # mouth_region_size = opt.mouth_region_size
    # radius = mouth_region_size//2
    # radius_1_4 = radius//4
    
    
    # start train
    for epoch in range(opt.start_epoch, opt.non_decay+opt.decay+1):
        net_g.eval() ####### 降低显存
        for iteration, data in enumerate(training_data_loader):
            # forward
            source_clip,source_clip_mask, reference_clip,deep_speech_clip,deep_speech_full,flag = data
            print("flag:",flag)

            flag_true = torch.tensor([True], dtype=torch.bool).expand_as(flag)
            flag_false = torch.tensor([False], dtype=torch.bool).expand_as(flag)

            print("flag 是否能排除:",torch.equal(flag, flag_true))
            # print("flag:",flag)
            if torch.equal(flag, flag_false):
                # import pdb
                # pdb.set_trace()
                print(flag,"跳过训练")
                continue
            
            # print("*****************************************")
            # print("source_clip.shape",source_clip.shape)
            # print("source_clip_mask.shape",source_clip_mask.shape)
            # print("reference_clip.shape",reference_clip.shape)
            # print("deep_speech_clip.shape",deep_speech_clip.shape)
            # print("deep_speech_full.shape",deep_speech_full.shape)
            # print("*****************************************")
            
            source_clip = torch.cat(torch.split(source_clip, 1, dim=1), 0).squeeze(1).float().cuda()
            source_clip_mask = torch.cat(torch.split(source_clip_mask, 1, dim=1), 0).squeeze(1).float().cuda()
            reference_clip = torch.cat(torch.split(reference_clip, 1, dim=1), 0).squeeze(1).float().cuda()
            deep_speech_clip = torch.cat(torch.split(deep_speech_clip, 1, dim=1), 0).squeeze(1).float().cuda()
            deep_speech_full = deep_speech_full.float().cuda()
            
            # print("*****************************************")
            # print("source_clip.shape",source_clip.shape)
            # print("source_clip_mask.shape",source_clip_mask.shape)
            # print("reference_clip.shape",reference_clip.shape)
            # print("deep_speech_clip.shape",deep_speech_clip.shape)
            # print("deep_speech_full.shape",deep_speech_full.shape)
            # print("*****************************************")
            
            '''
            source_clip.shape torch.Size([5, 3, 416, 320])                                                                                         
            source_clip_mask.shape torch.Size([5, 3, 416, 320])                                                                                    
            reference_clip.shape torch.Size([5, 15, 416, 320])                                                                                     
            deep_speech_clip.shape torch.Size([5, 29, 5])                                                                                          
            deep_speech_full.shape torch.Size([1, 29, 9])  
            
            '''
            
            # dengjunli
            # ## 因为在此处还不是tensor，所以不能用g_bluring(source_image_data)
            # source_image_data = g_bluring_whole_image(source_clip) # dengjunli
            # # dengjunli

            
            
            # source_clip_mask[:,:,radius:radius + mouth_region_size,
            # radius_1_4:radius_1_4 + mouth_region_size] = g_bluring_mouth_region(source_clip_mask[:,:,radius:radius + mouth_region_size,
            # radius_1_4:radius_1_4 + mouth_region_size])

            # if iteration % 10 == 0:
            #     images = []
            #     for i in range(source_clip.shape[0]):
            #         img = source_clip[i].permute(1, 2, 0).detach().cpu().numpy()
            #         img = (img * 255).round().astype(np.uint8)
            #         images.append(img)
            #     merged_img = np.concatenate(images, axis=1)
            #     wandb.log({f"DINet-source_GT": wandb.Image(merged_img)}) 
            # # 保存合并后的图片到本地磁盘上

            # if iteration % 10 == 0:
            #     images = []
            #     for i in range(source_clip_mask.shape[0]):
            #         img = source_clip_mask[i].permute(1, 2, 0).detach().cpu().numpy()
            #         img = (img * 255).round().astype(np.uint8)
            #         images.append(img)
            #     merged_img = np.concatenate(images, axis=1)
            #     wandb.log({f"DINet-source_input_with_mask": wandb.Image(merged_img)}) 
            # # 保存合并后的图片到本地磁盘上

            # if iteration % 10 == 0:
            #     refer_imgs = []
            #     for i in range(int(reference_clip.shape[0])):
            #         for j in range(0 , int(reference_clip.shape[1]) ,3):
            #             img = reference_clip[i][j:j+3].permute(1, 2, 0).detach().cpu().numpy()
            #             img = (img * 255).round().astype(np.uint8)
            #             refer_imgs.append(img)   
            #             # print("ref_img_tensor.shape",img.shape)
            #     merged_img_2 = np.concatenate(refer_imgs, axis=1)
            #     wandb.log({f"ref_img_tensor-train": wandb.Image(merged_img_2)}) 
            #     # cv2.imwrite('./inference_ref_img_tensor-从train拿过来的.jpg', merged_img_2[:,:,::-1])
            #     # print("从train拿过来的参考帧 merged_img_2.shape",merged_img_2.shape)
            
            # #####
            fake_out = net_g(source_clip_mask,reference_clip,deep_speech_clip)
            # #####

            # if iteration % 10 == 0:
            #     images = []
            #     # 将 tensor 转换为 Numpy 数组并保存为 WandB 图像对象
            #     for i in range(fake_out.shape[0]):
            #         img = fake_out[i].permute(1, 2, 0).detach().cpu().numpy()
            #         img = (img * 255).round().astype(np.uint8)   
            #         images.append(img)       
            #     merged_img = np.concatenate(images, axis=1)
            #     wandb.log({f"fake-out": wandb.Image(merged_img)}) # rgb
            #     # cv2.imwrite('merged_images_fake-out.jpg', merged_img[:,:,::-1]) # bgr




            # (1)(2) 两步训练有什么区别？？？？
            # source_clip 的 含义是什么？？？？

            ## sync perception loss
            fake_out_clip = torch.cat(torch.split(fake_out, opt.batch_size, dim=0), 1)
            fake_out_clip_mouth = fake_out_clip[:, :, train_data.radius:train_data.radius + train_data.mouth_region_size,
            train_data.radius_1_4:train_data.radius_1_4 + train_data.mouth_region_size]

            ### 唇形同步损失
            sync_score = net_lipsync(fake_out_clip_mouth, deep_speech_full)
            ########

            # import pdb
            # pdb.set_trace()
            print("注释掉了origin 唇形同步损失，用GT验证和random验证", fake_out_clip_mouth.shape)
            print("fake_out_clip_mouth.shape", fake_out_clip_mouth.shape) # 1 15 256 256
            print("deep_speech_full.shape", deep_speech_full.shape) # 1 29 9

            source_clip_clip = torch.cat(torch.split(source_clip, opt.batch_size, dim=0), 1)

            source_clip_mouth = source_clip_clip[:, :, train_data.radius:train_data.radius + train_data.mouth_region_size,
            train_data.radius_1_4:train_data.radius_1_4 + train_data.mouth_region_size]
            print("source_clip_mouth.shape", source_clip_mouth.shape) # 1 15 256 256

            print("gt输入的唇形同步损失，分数应该为1，损失应该为0:",criterionMSE( net_lipsync(source_clip_mouth, deep_speech_full), real_tensor.expand_as(sync_score)))
            print("random输入的唇形同步损失，分数应该为0，损失应该为1:",criterionMSE( net_lipsync(torch.randn_like(source_clip_mouth), deep_speech_full), real_tensor.expand_as(sync_score)))
            #########

            sync_loss_actrully_training = criterionMSE(sync_score, real_tensor.expand_as(sync_score))

            sync_loss_GT =criterionMSE( net_lipsync(source_clip_mouth, deep_speech_full), real_tensor.expand_as(sync_score))

            sync_loss_random = criterionMSE( net_lipsync(torch.randn_like(source_clip_mouth), deep_speech_full), real_tensor.expand_as(sync_score))

            wandb.log({"sync 真实的损失 应很小 --》0":  sync_loss_actrully_training})
            wandb.log({"sync GT的损失 应为0:": sync_loss_GT})
            wandb.log({"sync random的损失 应很大 --》1": sync_loss_random})

            if sync_loss_GT > 0.25:
                images = []
                # 将 tensor 转换为 Numpy 数组并保存为 WandB 图像对象
                for i in range(source_clip.shape[0]):
                    img = source_clip[i].permute(1, 2, 0).detach().cpu().numpy()
                    img = (img * 255).round().astype(np.uint8)   
                    images.append(img)       
                merged_img = np.concatenate(images, axis=1)
                wandb.log({f" GT  的感知损失大于0.25 不合理， source_clip GT 如图": wandb.Image(merged_img)}) # rgb


            if sync_loss_actrully_training > 0.25:
                images = []
                # 将 tensor 转换为 Numpy 数组并保存为 WandB 图像对象
                for i in range(fake_out.shape[0]):
                    img = fake_out[i].permute(1, 2, 0).detach().cpu().numpy()
                    img = (img * 255).round().astype(np.uint8)   
                    images.append(img)       
                merged_img = np.concatenate(images, axis=1)
                wandb.log({f" 用预训练模型的 fake-out 的感知损失大于0.25 不合理 fake-out 如图": wandb.Image(merged_img)}) # rgb

                images = []
                # 将 tensor 转换为 Numpy 数组并保存为 WandB 图像对象
                for i in range(source_clip.shape[0]):
                    img = source_clip[i].permute(1, 2, 0).detach().cpu().numpy()
                    img = (img * 255).round().astype(np.uint8)   
                    images.append(img)       
                merged_img = np.concatenate(images, axis=1)
                wandb.log({f" 用预训练模型的 fake-out 的感知损失大于0.25 不合理 fake-out 对应的GT 如图": wandb.Image(merged_img)}) # rgb
            
    wandb.finish()