from models.Discriminator import Discriminator
from models.VGG19 import Vgg19
from models.LawDNet_new import LawDNet

from utils.training_utils import get_scheduler, update_learning_rate,GANLoss
from torch.utils.data import DataLoader
from dataset.dataset_DINet_frame import DINetDataset
from sync_batchnorm import convert_model
from config.config import DINetTrainingOptions
from tensor_processing import SmoothSqMask

from models.Gaussian_blur import Gaussian_bluring

import cv2
import random
import numpy as np
import os
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim

# from apex import amp
from torch.cuda.amp import autocast as autocast
import wandb

if __name__ == "__main__":
    '''
        frame training code of DINet
        we use coarse-to-fine training strategy
        so you can use this code to train the model in arbitrary resolution
    '''
    wandb.login()
    run = wandb.init(
    project="北科大-1000数据集-实验新1-复现效果",
    config={
        "training_type": "4 clip",
        "grid_size": "52 40",
    })

    # 设置 PyTorch 只使用第一张 GPU 卡
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("已设置 PyTorch 只使用第一张 GPU 卡")

    # load config
    opt = DINetTrainingOptions().parse_args()
    device_ids = [0,1,2,3]
    # 保存命令行参数到WandB
    wandb.config.update(opt)
    # set seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    # load training data in memory
    train_data = DINetDataset(opt.train_data,opt.augment_num,opt.mouth_region_size)
    training_data_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True,drop_last=True)
    train_data_length = len(training_data_loader)
    # init network
    net_g = LawDNet(opt.source_channel,opt.ref_channel,opt.audio_channel).cuda()
    net_dI = Discriminator(opt.source_channel ,opt.D_block_expansion, opt.D_num_blocks, opt.D_max_features).cuda()
    net_vgg = Vgg19().cuda()

    # parallel
    net_g = nn.DataParallel(net_g, device_ids=device_ids)
    net_g = convert_model(net_g)
    net_dI = nn.DataParallel(net_dI, device_ids=device_ids)
    net_vgg = nn.DataParallel(net_vgg, device_ids=device_ids)

    # setup optimizer
    optimizer_g = optim.AdamW(net_g.parameters(), lr=opt.lr_g)
    optimizer_dI = optim.AdamW(net_dI.parameters(), lr=opt.lr_dI)

    # 混合精度训练：Creates a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # coarse2fine
    if opt.coarse2fine:
        print('loading checkpoint for coarse2fine training: {}'.format(opt.coarse_model_path))
        checkpoint = torch.load(opt.coarse_model_path)
        net_g.load_state_dict(checkpoint['state_dict']['net_g'])

    # set criterion
    criterionGAN = GANLoss().cuda()
    criterionL1 = nn.L1Loss().cuda()
    criterionL2 = nn.MSELoss().cuda()

    start_epoch = opt.start_epoch
    non_decay = opt.non_decay
    decay = opt.decay

    if opt.mouth_region_size == 64:
        print("使用 64x64 的嘴巴区域")
        non_decay = 1
        decay = 1
    elif opt.mouth_region_size == 128:
        print("使用 128x128 的嘴巴区域")
        non_decay = 1
        decay = 1
    elif opt.mouth_region_size == 256:
        print("使用 256x256 的嘴巴区域")
        non_decay = 1
        decay = 1
    else:
        print("输入的 嘴巴区域 尺寸跟原作者不一致")

    net_g_scheduler = get_scheduler(optimizer_g, non_decay, decay)
    net_dI_scheduler = get_scheduler(optimizer_dI, non_decay, decay)

    # 用于做soft正方形mask
    mask_gaussian = Gaussian_bluring(radius=10,sigma=4,padding='same')
    smooth_sqmask = SmoothSqMask().cuda()
    mouth_region_size = opt.mouth_region_size
    radius = mouth_region_size//2
    radius_1_4 = radius//4

    img_w = radius * 2 + radius_1_4 * 2
    img_h = int(img_w * 1.3)

    for epoch in range(start_epoch, non_decay+decay):
        net_g.train()
        for iteration, data in enumerate(training_data_loader):
            # read data
            source_image_data, _, reference_clip_data,deepspeech_feature,flag = data
            if not (flag.equal(torch.ones(opt.batch_size,1,device='cuda:0'))):
                continue

            source_image_data_big = source_image_data.float().cuda()
            reference_clip_data_big = reference_clip_data.float().cuda()
            deepspeech_feature_big = deepspeech_feature.float().cuda()

            # 正方形的softmask ######################################
            source_image_mask_big = smooth_sqmask(source_image_data_big)
            # 正方形的softmask ######################################

            source_image_data = F.interpolate(source_image_data_big, size=(img_h, img_w), mode='bilinear', align_corners=False)
            source_image_mask = F.interpolate(source_image_mask_big, size=(img_h, img_w), mode='bilinear', align_corners=False)
            reference_clip_data = F.interpolate(reference_clip_data_big, size=(img_h, img_w), mode='bilinear', align_corners=False)

            if iteration % opt.freq_wandb == 0:
                images = []
                for i in range(source_image_data.shape[0]):
                    img = source_image_data[i].permute(1, 2, 0).detach().cpu().numpy()
                    img = (img * 255).round().astype(np.uint8)
                    images.append(img)
                merged_img = np.concatenate(images, axis=1)
                wandb.log({f"DINet-source_GT": wandb.Image(merged_img)}) 
                # cv2.imwrite('123_DINet-source_GT.jpg', merged_img[:,:,::-1])

            if iteration % opt.freq_wandb == 0:
                images = []
                for i in range(source_image_mask.shape[0]):
                    img = source_image_mask[i].permute(1, 2, 0).detach().cpu().numpy()
                    img = (img * 255).round().astype(np.uint8)
                    images.append(img)
                merged_img = np.concatenate(images, axis=1)
                wandb.log({f"DINet-source_input_with_mask": wandb.Image(merged_img)}) 
                # cv2.imwrite('123_DINet-source_input_with_mask.jpg', merged_img[:,:,::-1])

            # network forward
            with autocast(enabled=True): # 混合精度训练
                fake_out = net_g(source_image_mask,reference_clip_data,deepspeech_feature)

            if iteration % opt.freq_wandb == 0:
                images = []
                for i in range(fake_out.shape[0]):
                    img = fake_out[i].permute(1, 2, 0).detach().cpu().numpy()
                    img = (img * 255).round().astype(np.uint8)   
                    images.append(img)       
                merged_img = np.concatenate(images, axis=1)
                wandb.log({f"fake-out": wandb.Image(merged_img)}) # rgb
                # cv2.imwrite('123_merged_images_fake-out.jpg', merged_img[:,:,::-1]) # bgr

            # down sample output image and real image
            fake_out_half = F.avg_pool2d(fake_out, 3, 2, 1, count_include_pad=False)
            target_tensor_half = F.interpolate(source_image_data, scale_factor=0.5, mode='bilinear')
            # (1) Update D network
            optimizer_dI.zero_grad()
            # compute fake loss
            with autocast(enabled=True):
                _,pred_fake_dI = net_dI(fake_out)
                loss_dI_fake = criterionGAN(pred_fake_dI, False)
                # compute real loss
                _,pred_real_dI = net_dI(source_image_data)
                loss_dI_real = criterionGAN(pred_real_dI, True)
                # Combined DI loss
                loss_dI = (loss_dI_fake + loss_dI_real) * 0.5

            # # 混合精度训练
            scaler.scale(loss_dI).backward(retain_graph=True)
            scaler.step(optimizer_dI)

            # (2) Update G network
            # 混合精度训练
            with autocast(enabled=True):
                _, pred_fake_dI = net_dI(fake_out)
                optimizer_g.zero_grad()
                # compute perception loss
                perception_real = net_vgg(source_image_data)
                perception_fake = net_vgg(fake_out)
                perception_real_half = net_vgg(target_tensor_half)
                perception_fake_half = net_vgg(fake_out_half)
                loss_g_perception = 0
                for i in range(len(perception_real)):
                    loss_g_perception += criterionL1(perception_fake[i], perception_real[i])
                    loss_g_perception += criterionL1(perception_fake_half[i], perception_real_half[i])
                loss_g_perception = (loss_g_perception / (len(perception_real) * 2)) 
                # # gan dI loss
                loss_g_dI = criterionGAN(pred_fake_dI, True)
                loss_mse = criterionL2(fake_out, source_image_data)

                # combine perception loss and gan loss
                loss_g = loss_g_perception * opt.lamb_perception + loss_mse * opt.lambda_img + \
                         loss_g_dI * opt.lambda_g_dI

            scaler.scale(loss_g).backward()
            scaler.step(optimizer_g)
            # 混合精度训练
            scaler.update()

            wandb.log({"epoch": epoch,  
                       "len(training_data_loader)": len(training_data_loader),
                        "1:loss_dI": float(loss_dI),
                        str(opt.lambda_g_dI) + ":loss_g_dI": float(loss_g_dI*opt.lambda_g_dI),
                        str(opt.lamb_perception)+ ":loss_g_perception": float(loss_g_perception),
                        str(opt.lambda_img)+"loss_mse": float(loss_mse * opt.lambda_img),
                       }
                      )
            
            # print("epoch:",epoch,"iteration:",iteration,"loss_dI:",loss_dI,"loss_g_dI:",
            #       loss_g_dI,"loss_g_perception:",loss_g_perception,"loss_mse:",loss_mse)

        update_learning_rate(net_g_scheduler, optimizer_g)
        update_learning_rate(net_dI_scheduler, optimizer_dI)
        #checkpoint
        if epoch %  opt.checkpoint == 0:
            if not os.path.exists(opt.result_path):
                os.mkdir(opt.result_path)
            model_out_path = os.path.join(opt.result_path, 'netG_model_epoch_{}.pth'.format(epoch))
            states = {
                'epoch': epoch + 1,
                'state_dict': {'net_g': net_g.state_dict(), 'net_dI': net_dI.state_dict()},#
                'optimizer': {'net_g': optimizer_g.state_dict(), 'net_dI': optimizer_dI.state_dict()}#
            }
            torch.save(states, model_out_path)
            print("Checkpoint saved to {}".format(epoch))

    wandb.finish()
