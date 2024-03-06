import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.io as io
import torchvision.transforms as transforms
from torch.cuda.amp import autocast as autocast


class Sobel_Edge_Detection(nn.Module):
    '''
    Sobel算子检测边缘
    '''
    def __init__(self,dilation=1,padding=0):
        super(Sobel_Edge_Detection, self).__init__()
        self.dilation = dilation
        self.padding = padding
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        kernel = torch.cat([sobel_x, sobel_y], dim=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        
    def sobel_to_Gray(self, x):
        
        '''
        x: 3 channel image
        return: 1 channel gray image
        '''
        three_channel_sobel = self.forward(x)
        
        a1 = 0.299
        a2 = 0.587
        a3 = 0.114
        
        x = a1 * three_channel_sobel[:,0,:,:] + a2 * three_channel_sobel[:,1,:,:] + a3 * three_channel_sobel[:,2,:,:]
        
        return x
    
    def forward(self, x):   ## 要让这些参数不更新，就要加上requires_grad=False
        '''
        input: 3 channel image
        output: 3 channel edge image
        '''
        with autocast():
            with_batch = True if len(x.shape) == 4 else False
            channel_num = x.shape[-3]
            x = x.view((-1,1,)+x.shape[-2:]) # b*3 1 h w
            
            x_reshape = F.conv2d(x, self.weight, padding=self.padding,dilation=self.dilation) # b*3 2 h w (padding)
            x_reshape = torch.norm(x_reshape, dim=-3, keepdim=True) # b*3 1 h w
            
            # print("x_reshape.max:",torch.max(x_reshape))
            # print("x_reshape.min:",torch.min(x_reshape)) 
            
            # 要加sigmoid 防止梯度爆炸 或者归一化
            
            if with_batch:
                x_reshape = x_reshape.view((-1,channel_num,)+x_reshape.shape[-2:])
            return x_reshape

  
if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 打开视频文件
    video_path = '../asserts/inference_result/BobCorker_0_synthetic_face.mp4'
    video_capture = cv2.VideoCapture(video_path)
    
    
    ## 确认是RGB还是BGR

    # 转换视频帧的尺寸和通道数
    transform = transforms.Compose([
        transforms.ToPILImage(),  # 将张量转换为 PIL 图像 ，从BGR转换为RGB
        transforms.ToTensor(),  # 将图像转换为张量
    ])

    H = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    # 创建一个大小为 (batch_size, channels, height, width) 的空张量
    batch_size, channels, height, width = 2, 3, H, W
    frames_tensor = torch.empty((batch_size, channels, height, width))

    # 读取视频帧
    
    success, frame = video_capture.read()
    for i in range(2):
        print("i:",i)
        transformed_frame = transform(frame)
        frames_tensor[i] = transformed_frame
        
        # 读取下一帧
        success, frame = video_capture.read()

    # 释放视频捕获对象
    video_capture.release()
    
    # batch_frames_tensor = torch.cat((frames_tensor[0], frames_tensor[1]), dim=0)

    # 将视频张量移动到 GPU（如果可用）
    if torch.cuda.is_available():
        frames_tensor = frames_tensor.cuda()

    # 打印视频张量的形状
    print('batch视频张量的形状', frames_tensor.shape) # 1 3 416 320
    
    input_batch = frames_tensor


    sobel = Sobel_Edge_Detection(dilation=2) # dilation=5
    sobel.to(device)
    edges = sobel.sobel_to_Gray(input_batch)
    
    print("输出结果的形状shape:",edges.shape) # 1 3 h w
    
    # 转换张量为 PIL 图像
    transform = transforms.ToPILImage() # 0 - 1   0 - 255 
    
    pil_images = [transform(tensor) for tensor in edges]
    
    # 将PIL图像保存到文件系统中
    for i, img in enumerate(pil_images):
        img.save(f'example_{i}.jpg')
        
        
        
        

    # ##############################
    
    # image = transform(edges.squeeze())
    
    # image_path = '../image_no_norm-sobel.jpg'
    # image.save(image_path)
    
    # ##############################

    # edges_gray = sobel.sobel_to_Gray(input_batch)  
    # print("edges_gray.shape:",edges_gray.shape) # 1 1 h w
    
    # image = transform(edges_gray.squeeze())
    
    # image_path = '../image_no_norm-sobel-gray.jpg'
    # image.save(image_path)
    
    
    # # 分离每个通道的图像
    # channels = edges.shape[0]
    
    # import pdb
    # pdb.set_trace()
    
    # for i in range(channels):
    #     image = transform(edges[i].squeeze())

    #     # 保存图像
    #     image_path = f'./image_no_norm-sobel_{i}.jpg'
    #     image.save(image_path)
        
    