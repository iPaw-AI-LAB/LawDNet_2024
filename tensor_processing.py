import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch_affine_ops import *

class Gaussian_bluring(nn.Module):
  def __init__(self, radius=1,sigma=1,padding=0):
    super(Gaussian_bluring, self).__init__()

    self.sigma = sigma
    self.padding = padding
    self.radius = radius
    self.size = self.radius*2+1

    self.kernel = self.get_gaussian_kernel()

    self.weight = nn.Parameter(data=self.kernel.unsqueeze(0).unsqueeze(0), requires_grad=False).cuda()
 
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

# class SmoothMask(nn.Module):
#   def __init__(self, radius=60,sigma=20,padding='same',bias=0.08,scaling=80):
#     super(SmoothMask, self).__init__()
#     self.gaussian_bluringer = Gaussian_bluring(radius=radius,sigma=sigma,padding=padding)
#     self.bias = bias
#     self.scaling = scaling

#   def forward(self, input_face, landmarks, filling='black'):
#     """
#     input_face: B*3*H*W
#     landmarks: B*68*2
#     """
#     B, C, H, W = input_face.shape
#     device = input_face.device

#     landmarks_keys_part1 = landmarks[...,range(3,14),:] #
#     landmarks_keys_part2 = landmarks[...,range(48,68),:] # 
#     landmarks_keys_part3 = landmarks[...,range(6,11),:] #

#     landmarks_keys_part11 = (landmarks_keys_part1 - landmarks[...,30:31,:])*0.7 + landmarks[...,30:31,:]
#     landmarks_keys_part21 = (landmarks_keys_part1 - landmarks[...,30:31,:])*0.9 + landmarks[...,30:31,:]
#     landmarks_keys_part22 = (landmarks_keys_part2 - landmarks[...,30:31,:])*1.2 + landmarks[...,30:31,:]

#     landmarks_keys = torch.cat((landmarks_keys_part11,landmarks_keys_part2,landmarks_keys_part3,landmarks_keys_part21,landmarks_keys_part22),axis=-2)

#     landmarks_index_Y_test = torch.clip(landmarks_keys[...,1], 0, H-1).round().type(torch.LongTensor).to(device)
#     landmarks_index_X_test = torch.clip(landmarks_keys[...,0], 0, W-1).round().type(torch.LongTensor).to(device)


#     heatmap_mask = torch.zeros_like(input_face[:,0:1,:,:])

#     heatmap_mask[...,landmarks_index_Y_test,landmarks_index_X_test] = 1

#     heatmap_mask = self.gaussian_bluringer (heatmap_mask)
#     heatmap_mask = heatmap_mask/heatmap_mask.max() # 归一化

#     # 5. 将遮罩赋值给source_clip_mask
#     w_mask = torch.sigmoid((heatmap_mask-self.bias)*self.scaling) #########


#     w_mask[(w_mask < 1e-5)] = 0

#     w_mask[(w_mask > 1-1e-5)] = 1

#     w_origin = 1 - w_mask

#     if filling == 'black':
#       masked_output =  w_origin*input_face  #+ w_mask*torch.zeros_like(input_face)

#     if filling == 'mean':
      
#       masked_output =  w_origin*input_face  + w_mask*torch.mean(input_face,dim=[-1,-2],keepdim=True)

#     return masked_output


# def warp_img_torch(img, transform_matrix, output_size):

#     # please pay exact attention on the order of H and W,
#     # and the normalization of the grid in Torch, but not in OpenCV
#     device = img.device
#     B, C, H, W = img.shape
#     T = torch.Tensor([[2 / (W-1), 0, -1],
#               [0, 2 / (H-1), -1],
#               [0, 0, 1]]).to(device).repeat(B,1,1)
    
#     T2 = torch.Tensor([[2 / (output_size[1]-1), 0, -1],[0, 2 / (output_size[0]-1), -1],[0, 0, 1]]).to(device).repeat(B,1,1)
#     M_torch = torch.matmul(T2,torch.matmul(transform_matrix,torch.linalg.inv(T)))
#     grid_trans = torch.linalg.inv(M_torch)[:,0:2,:]

#     grid = F.affine_grid(grid_trans, torch.Size((B, C, output_size[0], output_size[1])))
#     img = F.grid_sample(img, grid)
#     return img

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
  def __init__(self, radius=2,sigma=1,padding=0,standard_shape=(103,80)):
    super(SmoothSqMask, self).__init__()
    self.gaussian_bluringer = Gaussian_bluring(radius=radius,sigma=sigma,padding=padding)
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

# class FaceAlign(nn.Module):
#   def __init__(self, coverage_rt=0.9 ,device='cuda'):
#     super(FaceAlign, self).__init__()

#     self.coverage_rt = coverage_rt

#     self.standard_lm_25 = torch.tensor([[2.13256e-04, 1.06454e-01],
#        [7.52622e-02, 3.89150e-02],
#        [1.81130e-01, 1.87482e-02],
#        [2.90770e-01, 3.44891e-02],
#        [3.93397e-01, 7.73906e-02],
#        [5.86856e-01, 7.73906e-02],
#        [6.89483e-01, 3.44891e-02],
#        [7.99124e-01, 1.87482e-02],
#        [9.04991e-01, 3.89150e-02],
#        [9.80040e-01, 1.06454e-01],
#        [4.90127e-01, 2.03352e-01],
#        [4.90127e-01, 3.07009e-01],
#        [4.90127e-01, 4.09805e-01],
#        [4.90127e-01, 5.15625e-01],
#        [3.66880e-01, 5.87326e-01],
#        [4.26036e-01, 6.09345e-01],
#        [4.90127e-01, 6.28106e-01],
#        [5.54217e-01, 6.09345e-01],
#        [6.13373e-01, 5.87326e-01],
#        [1.21737e-01, 2.16423e-01],
#        [3.34606e-01, 2.31733e-01],
#        [6.45647e-01, 2.31733e-01],
#        [8.58516e-01, 2.16423e-01],
#        [2.54149e-01, 7.80233e-01],
#        [7.26104e-01, 7.80233e-01]], device=device).float()

 
#   def forward(self, feed_img, bbox, landmarks_tensor):
#     """
#     feed_img: B*C*H*W
#     bbox: B*4
#     landmarks: B*68*2
#     """
    
#     Batch_size, C, H, W = feed_img.shape

#     bbox_size = bbox[:,2:4] - bbox[:,0:2]
#     w, h = bbox_size[:,0], bbox_size[:,1]

#     input_shape2 = ['None', 3, 192, 192]

#     input_size = tuple(input_shape2[2:4][::-1])

#     center = (bbox[:,0:2] + bbox[:,2:4])/2


#     #center = landmarks_tensor[:,30,:]

#     list68to25 = list(range(17,37))+[39,42,45,48,54]
#     lmrks_25 = landmarks_tensor[:,list68to25,:].view(Batch_size,25,2)


#     # uni_landmarks_25 = torch.from_numpy(uni_landmarks_68).to(device).repeat(Batch_size,1,1)

#     Similarity_Matrix_3d_final = SimilarityTransform_torch_2D(lmrks_25, self.standard_lm_25.repeat(Batch_size,1,1))

#     theta = torch.arctan(Similarity_Matrix_3d_final[:,1,0]/Similarity_Matrix_3d_final[:,0,0])*180/torch.pi

#     output_size = min(feed_img.shape[-2:])
#     # output_size_w = 746
#     # output_size_h = 1102 

#     coverage = self.coverage_rt*min(feed_img.shape[-2:])/torch.max(bbox_size.float(),dim=-1, keepdim=True)[0]

#     face_align_matrix = transform_torch(center, (int(output_size), output_size), coverage, theta)

#     center_align = torch.matmul(face_align_matrix[:,:2,:2], center.unsqueeze(-1)) + face_align_matrix[:,:2,-1:]

#     face_align_matrix[:,:2,2:3] += - center_align + output_size/2


#     face_align_img = warp_img_torch(feed_img, face_align_matrix, (int(output_size), output_size))

#     face_align_img = torch.clip(face_align_img, 0, 255)

#     lmrks_align = torch.matmul(landmarks_tensor,face_align_matrix[:,:2,:2].transpose(-1,-2)) + face_align_matrix[:,:2,-1:].transpose(-1,-2)

#     return face_align_img, lmrks_align, face_align_matrix


#   def recover(self, fake_part, origin_img, face_align_matrix):

#     Batch_size, C, H, W = origin_img.shape

#     device = origin_img.device

#     assert fake_part.shape[-2] == fake_part.shape[-1]

#     output_size = fake_part.shape[-2]

#     face_align_matrix_inversed = torch.inverse(face_align_matrix)

#     # grid = F.affine_grid(torch.eye(3,device=device).unsqueeze(0)[:,0:2,:].repeat(B,1,1), torch.Size((B, C, output_size, output_size)))
#     grid = standard_grid((output_size,output_size),Batch_size,device=device)

#     grid_index =  (grid+1.0-1.0/output_size)*(output_size-1)/(2-2/output_size)
#     grid_index_reversed = torch.matmul(grid_index.view(Batch_size,-1,2),face_align_matrix_inversed[:,:2,:2].transpose(-1,-2)) + face_align_matrix_inversed[:,:2,-1:].transpose(-1,-2)
#     grid_index_reversed = grid_index_reversed.view(Batch_size,output_size,output_size,2).long()

  
#     #lmrks_align_reversed_np = lmrks_align_reversed[layer_index].cpu().numpy()

#     reform_img_2 = origin_img.permute(1,0,2,3).contiguous()

#     grid_index_reversed_2 = torch.cat([torch.range(0,Batch_size-1,device=device).long().view(Batch_size,1,1,1).repeat(1,output_size,output_size,1),grid_index_reversed],dim=-1)

#     index_set = (grid_index_reversed_2[...,0],
#                 grid_index_reversed_2[...,2],
#                 grid_index_reversed_2[...,1])

#     new_value = fake_part.permute(1,0,2,3)


#     for i in range(channel:=3):
#         reform_img_2[i] = reform_img_2[i].index_put(index_set, new_value[i]) 

#     # reform_img_2[:,grid_index_reversed_2[:,:,:,2],grid_index_reversed_2[:,:,:,1], grid_index_reversed_2[:,:,:,0]] = fake_img.permute(1,0,2,3)

#     reform_img = reform_img_2.permute(1,0,2,3)

#     return reform_img