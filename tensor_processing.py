'''
说明：用于对张量进行处理的函数，包括高斯模糊，正脸对齐，特征图可视化等
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch_affine_ops import *

class Gaussian_bluring(nn.Module):
  def __init__(self, radius=1,sigma=1,padding=0,device='cuda'):
    super(Gaussian_bluring, self).__init__()

    self.sigma = sigma
    self.padding = padding
    self.radius = radius
    self.size = self.radius*2+1

    self.kernel = self.get_gaussian_kernel()

    self.weight = nn.Parameter(data=self.kernel.unsqueeze(0).unsqueeze(0), requires_grad=False).to(device)
 
  def forward(self, x):
    """
    x: B*1*H*W or H*W
    """
    with_batch = True if len(x.shape) == 4 else False
    channel_num = x.shape[-3]
    x = x.view((-1,1,)+x.shape[-2:])  ## batch * channel * 1 
    x_reshape = F.conv2d(x,self.weight, padding=self.padding)
    if with_batch:
      x_reshape = x_reshape.view((-1,channel_num,)+x_reshape.shape[-2:])  ## 拆回去
    return x_reshape
  
  def get_gaussian_kernel(self):
    constant = 1/(2 * torch.pi * self.sigma**2)
    gaussian_kernel = torch.zeros((self.size, self.size))
    for i in range(0, self.size, 1):
        for j in range(0, self.size, 1):
            x = i-self.radius
            y = j-self.radius
            gaussian_kernel[i, j] = np.exp(-0.5/(self.sigma**2)*(x**2+y**2))

    gaussian_kernel = gaussian_kernel/gaussian_kernel.sum() 

    # print("self.kernel.shape:",gaussian_kernel.shape)
    return gaussian_kernel

class SmoothMask(nn.Module):
  def __init__(self, radius=15,sigma=4,padding='same'):
    super(SmoothMask, self).__init__()
    self.gaussian_bluringer = Gaussian_bluring(radius=radius,sigma=sigma,padding=padding)
    # self.bias = bias
    # self.scaling = scaling
    self.standard_shape = (364, 280)
    self.standard_resizer = transforms.Resize(self.standard_shape)
    

  def forward(self, input_face, landmarks, filling='black'):
    """
    input_face: B*3*H*W
    landmarks: B*68*2
    """
    B, C, H, W = input_face.shape
    
    device = input_face.device
    scaling = self.standard_shape[0]/H
    
    landmarks_standard = landmarks*scaling 


    landmarks_keys_part0 = landmarks_standard[...,range(30,36),:] #
    landmarks_keys_part1 = landmarks_standard[...,range(2,15),:] #
    landmarks_keys_part2 = landmarks_standard[...,range(48,68),:] # 
    landmarks_keys_part3 = landmarks_standard[...,range(6,11),:] #
    landmarks_center = landmarks_standard[...,30:31,:]#

    landmarks_keys_part01 = (landmarks_keys_part0 - landmarks_center)*1.5 + landmarks_center
    landmarks_keys_part11 = (landmarks_keys_part1 - landmarks_center)*0.5 + landmarks_center
    landmarks_keys_part12 = (landmarks_keys_part1 - landmarks_center)*0.7 + landmarks_center
    landmarks_keys_part13 = (landmarks_keys_part1 - landmarks_center)*0.9 + landmarks_center
    #landmarks_keys_part14 = (landmarks_keys_part1 - landmarks_center)*1.1 + landmarks_center
    landmarks_keys_part21 = (landmarks_keys_part2 - landmarks_center)*1.3 + landmarks_center

    landmarks_keys = torch.cat((landmarks_keys_part0,landmarks_keys_part01,landmarks_keys_part12, landmarks_keys_part13,landmarks_keys_part11,landmarks_keys_part2,landmarks_keys_part3,landmarks_keys_part21),axis=-2)

    landmarks_index_Y_test = torch.clip(landmarks_keys[...,1], 0, 363).round().type(torch.LongTensor).to(device)
    landmarks_index_X_test = torch.clip(landmarks_keys[...,0], 0, 279).round().type(torch.LongTensor).to(device)

    heatmap_mask = torch.zeros((B,1,self.standard_shape[0],self.standard_shape[1])).to(device)
    
    heatmap_mask[...,landmarks_index_Y_test,landmarks_index_X_test] = 1
    # diffusion the landmarks to the masked area
    w_mask = self.gaussian_bluringer(heatmap_mask) 


    w_mask[(w_mask > 1e-7)] = 1
    

    w_mask = self.gaussian_bluringer(w_mask)
    

    resizer = transforms.Resize((H,W))

    w_mask = resizer(w_mask)

    w_origin = 1 - w_mask

    if filling == 'black':
      masked_output =  w_origin*input_face  #+ w_mask*torch.zeros_like(input_face)

    if filling == 'mean':
      
      masked_output =  w_origin*input_face  + w_mask*torch.mean(input_face,dim=[-1,-2],keepdim=True)

    return masked_output

def warp_img_torch(img, transform_matrix, output_size):

    # please pay exact attention on the order of H and W,
    # and the normalization of the grid in Torch, but not in OpenCV
    device = img.device
    B, C, H, W = img.shape
    T = torch.Tensor([[2 / (W-1), 0, -1],
              [0, 2 / (H-1), -1],
              [0, 0, 1]]).to(device).repeat(B,1,1)
    
    T2 = torch.Tensor([[2 / (output_size[1]-1), 0, -1],[0, 2 / (output_size[0]-1), -1],[0, 0, 1]]).to(device).repeat(B,1,1)
    M_torch = torch.matmul(T2,torch.matmul(transform_matrix,torch.linalg.inv(T)))
    grid_trans = torch.linalg.inv(M_torch)[:,0:2,:]

    grid = F.affine_grid(grid_trans, torch.Size((B, C, output_size[0], output_size[1])))
    img = F.grid_sample(img, grid)
    return img


class FaceAlign(nn.Module):
  
  def __init__(self, ratio=2, device='cuda'):
    super(FaceAlign, self).__init__()

    self.list68to25 = list(range(17,37))+[39,42,45,48,54]


    self.standard_lm_25 = torch.tensor([[2.13256e-04, 1.06454e-01],
       [7.52622e-02, 3.89150e-02],
       [1.81130e-01, 1.87482e-02],
       [2.90770e-01, 3.44891e-02],
       [3.93397e-01, 7.73906e-02],
       [5.86856e-01, 7.73906e-02],
       [6.89483e-01, 3.44891e-02],
       [7.99124e-01, 1.87482e-02],
       [9.04991e-01, 3.89150e-02],
       [9.80040e-01, 1.06454e-01],
       [4.90127e-01, 2.03352e-01],
       [4.90127e-01, 3.07009e-01],
       [4.90127e-01, 4.09805e-01],
       [4.90127e-01, 5.15625e-01],
       [3.66880e-01, 5.87326e-01],
       [4.26036e-01, 6.09345e-01],
       [4.90127e-01, 6.28106e-01],
       [5.54217e-01, 6.09345e-01],
       [6.13373e-01, 5.87326e-01],
       [1.21737e-01, 2.16423e-01],
       [3.34606e-01, 2.31733e-01],
       [6.45647e-01, 2.31733e-01],
       [8.58516e-01, 2.16423e-01],
       [2.54149e-01, 7.80233e-01],
       [7.26104e-01, 7.80233e-01]], device=device).float()


    up = (self.standard_lm_25[2,:]+ self.standard_lm_25[7,:])/2
    mark = self.standard_lm_25[12,:]
    scale = (mark-up)[1]
    center = up + (mark- up)*6.5/4
    center[0] = mark[0]
    self.standard_lm_25 -= center
    self.standard_lm_25 /= ratio*scale 

    self.feather = SmoothSqMask()

  def Face_align_matrix(self, landmarks_tensor, out_W = 280):
    
    device = landmarks_tensor.device

    B = landmarks_tensor.shape[0]
    out_H = out_W*1.3

    lmrks_25 = landmarks_tensor[:,self.list68to25,:].view(-1,25,2)
    
    stand_lm_25 = self.standard_lm_25.clone()

    stand_lm_25 *= out_W/2
    stand_lm_25 = stand_lm_25.to(device)
    stand_lm_25 += torch.tensor([out_W/2, out_H/2],device=device).float()

    Similarity_Matrix_3d_final = SimilarityTransform_torch_2D(lmrks_25, stand_lm_25.repeat(B,1,1))

    return Similarity_Matrix_3d_final

  def forward(self, feed_img, landmarks_tensor, out_W = 280):
    
    face_align_matrix = self.Face_align_matrix(landmarks_tensor, out_W = out_W)

    face_align_img = warp_img_torch(feed_img, face_align_matrix, (int(1.3*out_W), int(out_W)))

    face_align_img = torch.clip(face_align_img, 0, 255)

    face_align_img = torch.clip(face_align_img, 0, 255)

    lmrks_align = torch.matmul(landmarks_tensor,face_align_matrix[:,:2,:2].transpose(-1,-2)) + face_align_matrix[:,:2,-1:].transpose(-1,-2)

    return face_align_img, lmrks_align, face_align_matrix

  def feathering(self, fake_part, origin_img):

    out_part_alpha = 0.8*self.feather(torch.ones_like(fake_part[:,0:1,:,:]))

    fake_part_alpha = 1 - out_part_alpha

    return fake_part_alpha*fake_part + out_part_alpha*origin_img


  def recover(self, fake_part, origin_img, face_align_matrix):

    B, C, H, W = origin_img.shape

    device = origin_img.device

    output_size = fake_part.shape[-2:]



    output_size_tensor = torch.tensor(output_size,device=device).float()

    face_align_matrix_inversed = torch.inverse(face_align_matrix)

    # grid = F.affine_grid(torch.eye(3,device=device).unsqueeze(0)[:,0:2,:].repeat(B,1,1), torch.Size((B, C, output_size, output_size)))
    grid = standard_grid(output_size,B,device=device)

    grid_index = (grid + 1.0-1.0/output_size_tensor.flip(-1))*(output_size_tensor.flip(-1)-1)/(2-2/output_size_tensor.flip(-1))
    grid_index_reversed = torch.matmul(grid_index.view(B,-1,2),face_align_matrix_inversed[:,:2,:2].transpose(-1,-2)) + face_align_matrix_inversed[:,:2,-1:].transpose(-1,-2)
    grid_index_reversed = grid_index_reversed.view(B,output_size[0],output_size[1],2).long()

  
    #lmrks_align_reversed_np = lmrks_align_reversed[layer_index].cpu().numpy()

    reform_img_2 = origin_img.permute(1,0,2,3).contiguous()

    grid_index_reversed_2 = torch.cat([torch.range(0,B-1,device=device).long().view(B,1,1,1).repeat(1,output_size[0],output_size[1],1),grid_index_reversed],dim=-1)

    index_set = (grid_index_reversed_2[...,0],
                grid_index_reversed_2[...,2],
                grid_index_reversed_2[...,1])

    new_value = fake_part.permute(1,0,2,3)

    channel = 3
    for i in range(channel):
        reform_img_2[i] = reform_img_2[i].index_put(index_set, new_value[i]) 

    # reform_img_2[:,grid_index_reversed_2[:,:,:,2],grid_index_reversed_2[:,:,:,1], grid_index_reversed_2[:,:,:,0]] = fake_img.permute(1,0,2,3)

    reform_img = reform_img_2.permute(1,0,2,3)

    return reform_img


class SmoothSqMask(nn.Module):
  def __init__(self, radius=2,sigma=1,padding=0,standard_shape=(103,80),device='cuda'):
    super(SmoothSqMask, self).__init__()
    self.gaussian_bluringer = Gaussian_bluring(radius=radius,sigma=sigma,padding=padding,device=device)
    # self.bias = bias
    # self.scaling = scaling
    self.standard_shape = standard_shape
  
    self.w0 = int(0.1*self.standard_shape[1])
    self.w1 = int(0.9*self.standard_shape[1])
    self.H0 = int(0.4*self.standard_shape[1])
    self.H1 = int(1.2*self.standard_shape[1])
    
  def forward(self, input_face):
    """
    input_face: B*3*H*W
    landmarks: B*68*2
    """
    B, C, H, W = input_face.shape
    mask_alpha = torch.zeros((B,1,self.standard_shape[0],self.standard_shape[1]),device=input_face.device)
    mask_alpha[:,:,self.H0:self.H1,self.w0:self.w1] = 1
    mask_alpha = self.gaussian_bluringer(mask_alpha)
    
    resizer = transforms.Resize((H,W))
    mask_alpha = resizer(mask_alpha)

    mask_alpha = 1 - mask_alpha
    masked_output =  mask_alpha*input_face 
  
    return masked_output


### 以上为yihao，以下为junli





def concat_ref_and_src(src, ref):
    '''
    input: B*1,15,h,w ; B*1,15,h,w  （原图 ，参考帧）
    output: B*1,30,h,w
    '''

    # 原图和参考帧的张量维度
    original_frames = src
    reference_frames = ref

    # 提取A人物的原图（1，15，h，w）
    A_original = original_frames[0:1, :, :, :]
    # 提取A人物的参考帧（1，15，h，w）
    A_reference = reference_frames[0:1, :, :, :]

    # 提取B人物的原图（1，15，h，w）
    B_original = original_frames[1:2, :, :, :]
    # 提取B人物的参考帧（1，15，h，w）
    B_reference = reference_frames[1:2, :, :, :]

    # 将A人物的原图和参考帧按通道拼接为（1，30，h，w）
    # import pdb; pdb.set_trace()
    try:
        A_combined = torch.cat((A_original, A_reference), dim=1)
    except Exception as e:
        import pdb; pdb.set_trace()

    # 将B人物的原图和参考帧按通道拼接为（1，30，h，w）
    B_combined = torch.cat((B_original, B_reference), dim=1)

    combined = torch.cat((A_combined, B_combined), dim=0)

    return combined

def gamma_correction(img, gamma=2.0):
    # 使用Gamma值校正图像
    corrected = np.power(img, 1 / gamma)
    return np.clip(corrected, 0, 1)

def save_feature_map(ref_in_feature,filename_prefix):

    # plt.imshow(feature_image, cmap='viridis')  # 使用viridis色图，您也可以选择其他色图如'gray'
    # plt.axis('off')
    # plt.colorbar()  # 添加颜色条以显示数值范围
    # plt.savefig('feature_visualization.png', bbox_inches='tight', pad_inches=0)

    # # 归一化到0-255范围
    # feature_image_normalized = ((feature_image - feature_image.min()) / (feature_image.max() - feature_image.min()) * 255).astype(np.uint8)
    # cv2.imwrite('/home/dengjunli/data/dengjunli/autodl拿过来的/DINet-update/Exp-of-Junli/特征图可视化-feature_visualization.png', feature_image_normalized)
    # print('特征图可视化-feature_visualization.png已保存')

    # 可视化并保存ref_in_feature的前几个特征通道
    num_channels_to_save = 10  # 您可以修改这个值以保存更多或更少的通道
    image_path = "/home/dengjunli/data/dengjunli/autodl拿过来的/DINet-update/Exp-of-Junli/"

    for channel in range(num_channels_to_save):
        feature_image = ref_in_feature[0, channel].cpu().detach().numpy()

        # 使用Gamma校正提亮特征图像
        feature_image = gamma_correction(feature_image)

        # 使用matplotlib的viridis colormap进行彩色映射
        feature_colored = plt.get_cmap('viridis')((feature_image - feature_image.min()) / (feature_image.max() - feature_image.min()))
        feature_colored = (feature_colored[:, :, :3] * 255).astype(np.uint8)  # 去除alpha通道并转换为0-255


        # 为每个通道保存一个唯一的文件名
        filename = f'{filename_prefix}_channel_{channel}_colored.png'
        cv2.imwrite(image_path + filename, cv2.cvtColor(feature_colored, cv2.COLOR_RGB2BGR))

        # 归一化到0-255范围
        feature_image_normalized = ((feature_image - feature_image.min()) / (feature_image.max() - feature_image.min()) * 255).astype(np.uint8)
        
        # 为每个通道保存一个唯一的文件名
        filename = f'{filename_prefix}_channel_{channel}_gray.png'
        cv2.imwrite(image_path + filename, feature_image_normalized)
        print(f'{filename_prefix}_channel_{channel}_colored.png', '已保存')

    return None


