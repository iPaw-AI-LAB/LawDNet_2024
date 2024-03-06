import torch
import cv2
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from sync_batchnorm import SynchronizedBatchNorm1d as BatchNorm1d
from torch.cuda.amp import autocast as autocast
import torchvision.transforms as transforms
import sys
sys.path.append("..")
from torch_affine_ops import standard_grid
import matplotlib.pyplot as plt
from tensor_processing_deng import save_feature_map


# class LocalAffineWarp(nn.Module):
#     '''
#        Local Affine Warping by Layers
#     '''
#     def __init__(self, para_ch=1024, num_kpoints=5, feature_ch=3):
#         super(LocalAffineWarp, self).__init__()
#         self.para_ch = para_ch
#         self.num_kpoints = num_kpoints
#         self.feature_ch = feature_ch
#         self.commn_linear = nn.Sequential(
#             nn.Linear(para_ch, para_ch),
#             nn.ReLU()
#         )

#         self.kpoints_normalized_layer = nn.Sequential(
#                 nn.Linear(para_ch, 2 * num_kpoints * feature_ch),
#                 nn.Tanh()
#             )

#         self.similarity_params_layer = nn.Sequential(
#                 nn.Linear(para_ch, 4 * num_kpoints * feature_ch),
#                 nn.Tanh()
#             )

#         self.radius_layer = nn.Sequential(
#                 nn.Linear(para_ch, num_kpoints * feature_ch),
#                 nn.Sigmoid()
#             )
            
#         self.tanh = nn.Tanh()
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, feature_map, para_code):
#         B,C,H,W = feature_map.shape

#         para_code = self.commn_linear(para_code)

#         kpoints_normalized = self.kpoints_normalized_layer(para_code).view(B, self.num_kpoints, self.feature_ch, 2)

#         radius = self.radius_layer(para_code).view(B, self.num_kpoints, self.feature_ch, 1)

#         similarity_params = self.similarity_params_layer(para_code).view(B, self.num_kpoints, self.feature_ch, 4)

#         warped_feature_map =  self.local_similarity_warping_by_layers(feature_map, kpoints_normalized, radius,similarity_params)

#         return warped_feature_map

#     def local_similarity_warping_by_layers(self, source_img, kpoints_normalized, radius, similarity_params):
#         '''
#         source_img : B*C*H*W
#         kpoints : B*N*C*2
#         radius : B*N*C*1
#         similarity_params : B*N*C*4
#         '''

#         B,C,H,W = source_img.shape
#         device = source_img.device
#         center = torch.tensor([H/2,W/2],device=device)

#         # regard the channel as the Depth of layers
#         Depth_coordinate = torch.linspace(-1+1/C,1-1/C,C,device=device)

#         N = kpoints_normalized.shape[1]

#         grid_identity = standard_grid((H,W),B,device=device) # B*H*W*2
#         #F.affine_grid(torch.eye(3,device=device).unsqueeze(0)[:,0:2,:], torch.Size((B, N, H, W))) # B*H*W*2

#         grid_identity = grid_identity.view(B,1,1,H,W,2).repeat(1,N,C,1,1,1).unsqueeze(-2)# B*N*H*W*1*2

#         offset_grid = grid_identity - kpoints_normalized.view(B,N,C,1,1,1,2)# B*N*C*H*W*1*2

#         dist_grid = torch.norm(offset_grid,dim=-1) # B*N*C*H*W*1
        
#         exp_minus_dist_grid = torch.exp(- dist_grid/(radius.view(B,N,C,1,1,1)))# B*N*C*H*W*1


#         scale_grid = (similarity_params[:,:,:,0:1]+1).view(B,N,C,1,1,1).repeat(1,1,1,H,W,1)*exp_minus_dist_grid+(1-exp_minus_dist_grid)*1.0
        
#         angle_grid = (similarity_params[:,:,:,1:2]*torch.pi).view(B,N,C,1,1,1).repeat(1,1,1,H,W,1)*exp_minus_dist_grid

#         trans_grid = similarity_params[:,:,:,-2:].view(B,N,C,1,1,2).repeat(1,1,1,H,W,1)*exp_minus_dist_grid

#         Depth_coordinate = Depth_coordinate.view(1,C,1,1,1).repeat(B,1,H,W,1)

#         rotation_matrix_grid = torch.cat([torch.cos(angle_grid), torch.sin(angle_grid), -torch.sin(angle_grid), torch.cos(angle_grid)], dim=-1).view(B,N,C,H,W,2,2)

#         grid_warped_2d = scale_grid*(offset_grid.matmul(rotation_matrix_grid.transpose(-1,-2))).view(B,N,C,H,W,2) + trans_grid + kpoints_normalized.view(B,N,C,1,1,2)

#         soft_max_weights = F.softmax(exp_minus_dist_grid,dim=1) # B*N*C*H*W*1

#         grid_warped_2d = (grid_warped_2d*soft_max_weights).sum(dim=1) # B*C*H*W*2

#         # print('Depth_coordinate.shape:',Depth_coordinate.shape)
#         # print('grid_warped_2d.shape:',grid_warped_2d.shape)


#         grid_warped_3d = torch.cat([grid_warped_2d,Depth_coordinate],dim=-1)


#         ### grid 
        
#         warped_img = F.grid_sample(source_img.unsqueeze(1), grid_warped_3d)  # B*1*C*H*W

#         return warped_img.squeeze(1)


class LocalAffineWarp(nn.Module):
    '''
    Local Affine Warping by Layers, invented by Yihao Luo,
    which is a generalization of Spatial Transformer Network,
    allowing local affine transformation smoothly on key points
    instead of global affine transformation
    
    input_shape: B*C*H*W
    para_code_shape: B*para_ch, 
        used to generate the parameters of local affine transformation
        though some simple net layers
    num_kpoints: number of key points, N
    feature_ch: number of channels of input images, C
    '''
    def __init__(self, para_ch=1024, num_kpoints=5, feature_ch=3, standard_grid_size=(60,60)):
        super(LocalAffineWarp, self).__init__()
        self.para_ch = para_ch
        self.num_kpoints = num_kpoints
        self.feature_ch = feature_ch
        self.commn_linear = nn.Sequential(
            nn.Linear(para_ch, para_ch),
            nn.ReLU()
        )

        self.kpoints_normalized_layer = nn.Sequential(
                nn.Linear(para_ch, 2 * num_kpoints * feature_ch),
                nn.Tanh()
            )

        self.similarity_params_layer = nn.Sequential(
                nn.Linear(para_ch, 4 * num_kpoints * feature_ch),
                nn.Tanh()
            )

        self.radius_layer = nn.Sequential(
                nn.Linear(para_ch, num_kpoints * feature_ch),
                nn.Sigmoid()
            )
            
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.stander_grid = standard_grid(standard_grid_size)

    def forward(self, feature_map, para_code):

        B,C,H,W = feature_map.shape

        para_code = self.commn_linear(para_code)

        kpoints_normalized = self.kpoints_normalized_layer(para_code).view(B, self.num_kpoints, self.feature_ch, 2)

        radius = self.radius_layer(para_code).view(B, self.num_kpoints, self.feature_ch, 1)

        similarity_params = self.similarity_params_layer(para_code).view(B, self.num_kpoints, self.feature_ch, 4)

        warped_feature_map =  self.local_similarity_warping_by_layers(feature_map, kpoints_normalized, radius,similarity_params)

        return warped_feature_map

    def local_similarity_warping_by_layers(self, source_img, kpoints_normalized, radius, similarity_params):
        '''
        source_img : B*C*H*W
        kpoints : B*N*C*2
        radius : B*N*C*1
        similarity_params : B*N*C*4
        '''

        B,C,H,W = source_img.shape
        device = source_img.device

        # regard the channel as the Depth of layers
        Depth_coordinate = torch.linspace(-1+1/C,1-1/C,C,device=device)

        N = kpoints_normalized.shape[1]

        H0, W0 = self.stander_grid.shape[-3:-1]

        grid_identity = self.stander_grid.unsqueeze(0).unsqueeze(0).repeat(1,N,C,1,1,1).unsqueeze(-2).to(device)# B*N*H*W*1*2

        # import pdb; pdb.set_trace()
        offset_grid = grid_identity - kpoints_normalized.view(B,N,C,1,1,1,2)# B*N*C*H0*W0*1*2

        dist_grid = torch.norm(offset_grid,dim=-1) # B*N*C*H0*W0*1
        
        exp_minus_dist_grid = torch.exp(- dist_grid/(radius.view(B,N,C,1,1,1)))# B*N*C*H0*W0*1

        scale_grid = (similarity_params[:,:,:,0:1]+1).view(B,N,C,1,1,1).repeat(1,1,1,H0,W0,1)*exp_minus_dist_grid+(1-exp_minus_dist_grid)*1.0
        
        angle_grid = (similarity_params[:,:,:,1:2]*torch.pi).view(B,N,C,1,1,1).repeat(1,1,1,H0,W0,1)*exp_minus_dist_grid

        trans_grid = similarity_params[:,:,:,-2:].view(B,N,C,1,1,2).repeat(1,1,1,H0,W0,1)*exp_minus_dist_grid

        Depth_coordinate = Depth_coordinate.view(1,C,1,1,1).repeat(B,1,H,W,1)

        rotation_matrix_grid = torch.cat([torch.cos(angle_grid), torch.sin(angle_grid), -torch.sin(angle_grid), torch.cos(angle_grid)], dim=-1).view(B,N,C,H0,W0,2,2)

        grid_warped_2d = scale_grid*(offset_grid.matmul(rotation_matrix_grid.transpose(-1,-2))).view(B,N,C,H0,W0,2) + trans_grid + kpoints_normalized.view(B,N,C,1,1,2)

        soft_max_weights = F.softmax(exp_minus_dist_grid,dim=1) # B*N*C*H*W*1

        grid_warped_2d = (grid_warped_2d*soft_max_weights).sum(dim=1) # B*C*H*W*2

        grid_warped_2d = grid_warped_2d.transpose(-1,-2).permute(0,1,4,2,3).contiguous().view(B,C*2,H0,W0)

        grid_warped_2d = transforms.Resize((H,W))(grid_warped_2d).to(device)


        # print('Depth_coordinate.shape:',Depth_coordinate.shape)
        # print('grid_warped_2d.shape:',grid_warped_2d.shape)

        grid_warped_2d = transforms.Resize((H,W))(grid_warped_2d).to(device)

        grid_warped_2d = grid_warped_2d.view(B,C,2,H,W).permute(0,1,3,4,2)

        grid_warped_3d = torch.cat([grid_warped_2d,Depth_coordinate],dim=-1)

        ### grid 
        warped_img = F.grid_sample(source_img.unsqueeze(1), grid_warped_3d)  # B*1*C*H*W

        return warped_img.squeeze(1)


class ResBlock1d(nn.Module):
    '''
        basic block
    '''
    def __init__(self, in_features,out_features, kernel_size, padding):
        super(ResBlock1d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv1 = nn.Conv1d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                               padding=padding)
        if out_features != in_features:
            self.channel_conv = nn.Conv1d(in_features,out_features,1)
        self.norm1 = BatchNorm1d(in_features)
        self.norm2 = BatchNorm1d(in_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        # out = self.norm1(x) # 取消batchnorm层
        # out = self.relu(out) # origin
        # out = self.relu(x) # origin
        # out = self.conv1(out) # origin

        out = self.conv1(x) # dengjunli
        # out = self.norm2(out) # 取消batchnorm层
        out = self.relu(out)
        out = self.conv2(out)
        if self.in_features != self.out_features:
            out += self.channel_conv(x)
        else:
            out += x
        return out

class ResBlock2d(nn.Module):
    '''
            basic block
    '''
    def __init__(self, in_features,out_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                               padding=padding)
        if out_features != in_features:
            self.channel_conv = nn.Conv2d(in_features,out_features,1)
        # self.norm1 = BatchNorm2d(in_features)
        # self.norm2 = BatchNorm2d(in_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        # out = self.norm1(x) # 注释掉batchnorm层
        # out = self.relu(out) # origin
        # out = self.relu(x) # origin
        # out = self.conv1(out) # origin
        out = self.conv1(x) # dengjunli
        # out = self.norm2(out) # 注释掉batchnorm层
        out = self.relu(out)
        out = self.conv2(out)
        if self.in_features != self.out_features:
            out += self.channel_conv(x)
        else:
            out += x
        return out

class UpBlock2d(nn.Module):
    '''
            basic block
    '''
    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(UpBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding)
        # self.norm = BatchNorm2d(out_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        # out = self.norm(out) # 注释掉batchnorm层
        out = F.relu(out)
        return out

class DownBlock1d(nn.Module):
    '''
            basic block
    '''
    def __init__(self, in_features, out_features, kernel_size, padding):
        super(DownBlock1d, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding,stride=2)
        # self.norm = BatchNorm1d(out_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.conv(x)
        # out = self.norm(out) # 注释掉batchnorm层
        out = self.relu(out)
        return out

class DownBlock2d(nn.Module):
    '''
            basic block
    '''
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, stride=2):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, stride=stride)
        # self.norm = BatchNorm2d(out_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.conv(x)
        # out = self.norm(out) # 注释掉batchnorm层
        out = self.relu(out)
        return out

class SameBlock1d(nn.Module):
    '''
            basic block
    '''
    def __init__(self, in_features, out_features,  kernel_size, padding):
        super(SameBlock1d, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding)
        # self.norm = BatchNorm1d(out_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.conv(x)
        # out = self.norm(out) # 注释掉batchnorm层
        out = self.relu(out)
        return out

class SameBlock2d(nn.Module):
    '''
            basic block
    '''
    def __init__(self, in_features, out_features,  kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding)
        # self.norm = BatchNorm2d(out_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.conv(x)
        # out = self.norm(out) # 取消batch norm层
        out = self.relu(out)
        return out
    

class GatedSameBlock2d(nn.Module):
    """
    Gated convolution layer. Consitis of 2 vanilla convolutions (`C_1` and `C_2`)
    and an activation function (`\phi`). Let `X` be input of `GatedConv2d` layer,
    then output is calculated as:

        Gating  = C_1(X)
        Feature = C_2(X)
        Output  = \phi(Feature) * \sigma(Gating)

    where `\sigma` is sigmoid activation function.
    Origin: https://arxiv.org/pdf/1806.03589v2.pdf
    https://github.com/sirius-image-inpainting/Free-Form-Image-Inpainting-With-Gated-Convolution/blob/main/model/layers.py
    """
    def __init__(self, in_features, out_features,  kernel_size=3, padding=1):
        super(GatedSameBlock2d, self).__init__()
        self.conv_gating = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding)
        self.conv_feature = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                        kernel_size=kernel_size, padding=padding)
        # self.norm = BatchNorm2d(out_features)
        self.gating_act = nn.Sigmoid()
        self.feature_act = nn.LeakyReLU(0.2)

        self.relu = nn.ReLU()
    def forward(self, x):

        gating = self.conv_gating(x)
        feature = self.conv_feature(x)
        output = self.feature_act(feature) * self.gating_act(gating)

        # out = self.conv(x)
        # # out = self.norm(out) # 取消batch norm层
        # out = self.relu(out)
        return output


class GatedDownBlock2d(nn.Module):
    """
    Gated convolution layer. Consitis of 2 vanilla convolutions (`C_1` and `C_2`)
    and an activation function (`\phi`). Let `X` be input of `GatedConv2d` layer,
    then output is calculated as:

        Gating  = C_1(X)
        Feature = C_2(X)
        Output  = \phi(Feature) * \sigma(Gating)

    where `\sigma` is sigmoid activation function.
    Origin: https://arxiv.org/pdf/1806.03589v2.pdf
    https://github.com/sirius-image-inpainting/Free-Form-Image-Inpainting-With-Gated-Convolution/blob/main/model/layers.py
    """
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, stride=2):
        super(GatedDownBlock2d, self).__init__()
        self.conv_gating = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, stride=stride)
        
        self.conv_feature = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, stride=stride)

        self.gating_act = nn.Sigmoid()
        self.feature_act = nn.LeakyReLU(0.2)
        
        # self.norm = BatchNorm2d(out_features)
        # self.relu = nn.ReLU()
    def forward(self, x):
        gating = self.conv_gating(x)
        feature = self.conv_feature(x)
        output = self.feature_act(feature) * self.gating_act(gating)
        return output

        ########################

        # out = self.conv(x)
        # # out = self.norm(out) # 注释掉batchnorm层
        # out = self.relu(out)
        # return out




# class AdaAT(nn.Module):
#     '''
#        AdaAT operator
#     '''
#     def __init__(self,  para_ch,feature_ch):
#         super(AdaAT, self).__init__()
#         self.para_ch = para_ch
#         self.feature_ch = feature_ch
#         self.commn_linear = nn.Sequential(
#             nn.Linear(para_ch, para_ch),
#             nn.ReLU()
#         )
#         self.scale = nn.Sequential(
#                     nn.Linear(para_ch, feature_ch),
#                     nn.Sigmoid()
#                 )
#         self.rotation = nn.Sequential(
#                 nn.Linear(para_ch, feature_ch),
#                 nn.Tanh()
#             )
#         self.translation = nn.Sequential(
#                 nn.Linear(para_ch, 2 * feature_ch),
#                 nn.Tanh()
#             )
#         self.tanh = nn.Tanh()
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, feature_map,para_code):
        # batch,d, h, w = feature_map.size(0), feature_map.size(1), feature_map.size(2), feature_map.size(3)
        # para_code = self.commn_linear(para_code)
        # scale = self.scale(para_code).unsqueeze(-1) * 2
        # angle = self.rotation(para_code).unsqueeze(-1) * 3.14159#
        # rotation_matrix = torch.cat([torch.cos(angle), -torch.sin(angle), torch.sin(angle), torch.cos(angle)], -1)
        # rotation_matrix = rotation_matrix.view(batch, self.feature_ch, 2, 2)
        # translation = self.translation(para_code).view(batch, self.feature_ch, 2)
        # grid_xy, grid_z = make_coordinate_grid_3d((d, h, w), feature_map.type())
        # grid_xy = grid_xy.unsqueeze(0).repeat(batch, 1, 1, 1, 1)
        # grid_z = grid_z.unsqueeze(0).repeat(batch, 1, 1, 1)
        # scale = scale.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)
        # rotation_matrix = rotation_matrix.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1, 1)
        # translation = translation.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)
        # trans_grid = torch.matmul(rotation_matrix, grid_xy.unsqueeze(-1)).squeeze(-1) * scale + translation
        # full_grid = torch.cat([trans_grid, grid_z.unsqueeze(-1)], -1)
        # trans_feature = F.grid_sample(feature_map.unsqueeze(1), full_grid).squeeze(1)
        # return trans_feature

class LawDNet(nn.Module):
    def __init__(self, source_channel = 5,ref_channel = 15,audio_channel = 29, warp_layer_num =2, num_kpoints=5):
        super(LawDNet, self).__init__()

        self.warp_layer_num = warp_layer_num
        self.num_kpoints = num_kpoints

        self.source_in_conv = nn.Sequential(
            SameBlock2d(source_channel,64,kernel_size=7, padding=3),
            DownBlock2d(64, 128, kernel_size=3, padding=1),
            DownBlock2d(128,256,kernel_size=3, padding=1),
        )
        self.ref_in_conv = nn.Sequential(
            SameBlock2d(ref_channel, 64, kernel_size=7, padding=3),
            DownBlock2d(64, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 256, kernel_size=3, padding=1),
        )
        self.trans_conv = nn.Sequential(
            # 20 →10
            SameBlock2d(512, 128, kernel_size=3, padding=1),
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            # 10 →5
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            # 5 →3
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            # 3 →2
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),

        )
        self.audio_encoder = nn.Sequential(
            SameBlock1d(audio_channel, 128, kernel_size=5, padding=2),
            ResBlock1d(128, 128, 3, 1),
            DownBlock1d(128, 128, 3, 1),
            ResBlock1d(128, 128, 3, 1),
            DownBlock1d(128, 128, 3, 1),
            SameBlock1d(128, 128, kernel_size=3, padding=1)
        )

        appearance_conv_list = []
        lawLayer_list = []
        for i in range(self.warp_layer_num):
            lawLayer_list.append(LocalAffineWarp(para_ch=256, num_kpoints=self.num_kpoints, feature_ch=256))
            appearance_conv_list.append(
                nn.Sequential(
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                )
            
            )

        self.appearance_conv_list = nn.ModuleList(appearance_conv_list)
        self.lawLayer_list = nn.ModuleList(lawLayer_list)

        # self.adaAT = AdaAT(256, 256)


        self.out_conv = nn.Sequential(
            SameBlock2d(512, 128, kernel_size=3, padding=1),
            UpBlock2d(128,128,kernel_size=3, padding=1),
            ResBlock2d(128, 128, 3, 1),
            UpBlock2d(128, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 3, kernel_size=(7, 7), padding=(3, 3)),
            nn.Sigmoid()
        )
        self.global_avg2d = nn.AdaptiveAvgPool2d(1)
        self.global_avg1d = nn.AdaptiveAvgPool1d(1)

    # 混合精度训练
    # @autocast()
    def forward(self, source_img,ref_img,audio_feature):
        with autocast():
            ## source image encoder
            source_in_feature = self.source_in_conv(source_img)

            # 输入维度
            # print("source_img.shape:",source_img.shape) # 5*batchsize 3 416 320
            # print("source_in_feature.shape:",source_in_feature.shape) # 5*batchsize 256 104 80

            ## reference image encoder
            ref_in_feature = self.ref_in_conv(ref_img)
            # save_feature_map(ref_in_feature,"没law之前的特征图")

            # print("ref_img.shape:",ref_img.shape) # 5*batchsize 15 416 320
            # print("ref_in_feature.shape:",ref_in_feature.shape) # 5*batchsize 256 104 80

            ## alignment encoder
            img_para = self.trans_conv(torch.cat([source_in_feature,ref_in_feature],1)) # 5*batchsize 128 7 5 
            img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2) # 5*batchsize 128

            ## audio encoder
            audio_para = self.audio_encoder(audio_feature) # 5*batchsize 128 2
            audio_para = self.global_avg1d(audio_para).squeeze(2) # 5*batchsize 128

            ## concat alignment feature and audio feature
            trans_para = torch.cat([img_para,audio_para],1) # 5*batchsize 256

            ## use Law Layer to do local affine warping
            merge_feature_list = [source_in_feature]
            for i in range(self.warp_layer_num):
                ref_trans_feature = self.appearance_conv_list[i](ref_in_feature)
                warped_ref_trans_feature = self.lawLayer_list[i](ref_trans_feature,trans_para) # 5*batchsize 256 10 8

            merge_feature_list.append(warped_ref_trans_feature)

            merge_feature = torch.cat(merge_feature_list,1) # 5*batchsize 512 10 8

            out = self.out_conv(merge_feature) # 5*batchsize 3 416 320
        return out


    


