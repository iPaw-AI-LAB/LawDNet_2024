import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


class GaussianBlur(nn.Module):
    def __init__(self, size, sigma=1.0):
        super(GaussianBlur, self).__init__()
        self.size = torch.tensor(size).cuda()
        self.sigma = torch.tensor(sigma).cuda()
        self.weight = self._create_kernel().cuda()


    def forward(self, x):

        channels = torch.split(x, 1, dim=1)  # b, c, h, w  # 都-3
        blurred_channels = []
        
        for channel in channels:
            blurred_channel = F.conv2d(channel, self.weight.unsqueeze(0).unsqueeze(0), padding=int(self.size//2))
            blurred_channels.append(blurred_channel)  # 1 1 h w
            
        # # 合并通道
        blurred_x = torch.cat(blurred_channels, dim=1)
        print("blurred_x",blurred_x.shape)       
        return blurred_x

    def _create_kernel(self):
        kernel = torch.zeros((self.size, self.size))
        center = self.size // 2

        for i in range(self.size):
            for j in range(self.size):
                kernel[i, j] = torch.exp(-((i - center) ** 2 + (j - center) ** 2) / (2 * self.sigma ** 2))

        kernel = kernel / torch.sum(kernel)
        weights = nn.Parameter(data=kernel, requires_grad=False)
        print("kernel:",weights)

        return weights
    

# 网上的  https://blog.csdn.net/bxdzyhx/article/details/120403956
def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,kernel_size=kernel_size, groups=channels, bias=False, padding=kernel_size//2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter


    
# luoyihao写的
# class Gaussian_bluring(nn.Module):
#   def __init__(self, radius=1,sigma=1,padding=0):
#     super(Gaussian_bluring, self).__init__()

#     self.sigma = sigma
#     self.padding = padding
#     self.radius = radius
#     self.size = self.radius*2+1
#     # self.padding = (self.size - 1) // 2 # padding
    

#     self.kernel = self.get_gaussian_kernel()

#     self.weight = nn.Parameter(data=self.kernel.unsqueeze(0).unsqueeze(0), requires_grad=False)
 
#   def forward(self, x):
#     with_batch = True if len(x.shape) == 4 else False
#     channel_num = x.shape[-3]
    
#     # print("x.shape:",x.shape)
#     # import pdb
#     # pdb.set_trace()
#     x = x.view((-1,1,)+x.shape[-2:])    # TypeError: Tuple must have size 2, but has size 4
#     # x.shape 416 320 3
#     # 需要变成 batchsize W H C
    
#     pad = nn.ReflectionPad2d((self.size - 1) // 2)
#     x = pad(x)
    
#     x_reshape = F.conv2d(x,self.weight.cuda(), padding=self.padding)
#     if with_batch:
#       x_reshape = x_reshape.view((-1,channel_num,)+x_reshape.shape[-2:])
#     return x_reshape
  
#   def get_gaussian_kernel(self):
#     constant = 1/(2 * torch.pi * self.sigma**2)
#     gaussian_kernel = torch.zeros((self.size, self.size))
#     for i in range(0, self.size, 1):
#         for j in range(0, self.size, 1):
#             x = i-self.radius
#             y = j-self.radius
#             gaussian_kernel[i, j] = constant*np.exp(-0.5/(self.sigma**2)*(x**2+y**2))

#     gaussian_kernel = gaussian_kernel/gaussian_kernel.sum() 
#     return gaussian_kernel


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
    

if __name__ == '__main__':

    
    # 创建一个大小为5的高斯模糊层（sigma默认为1.0）
    import torchvision.io as io
    
    # # 读取图片  用torch 自带的 io
    # # image_path = 'path/to/your/image.jpg'
    # image = io.read_image('../000000020434.jpg')
    # io.write_jpeg(image, '../io保存.jpg')
    # # 在第一维度上增加一个batch维度
    # image = image.unsqueeze(0).cuda().to(torch.float32)
    
    
    ##
    image = Image.open('../000000020434.jpg')
    # 转换为Tensor
    image = TF.to_tensor(image)
    
    print("image.type:",type(image))
    print("imag.shape:",image.shape)
    
    image = image.unsqueeze(0).cuda()
    
    
    origin_image = TF.to_pil_image(image.squeeze(0))
    # 保存图像
    origin_image.save("../原图_image.jpg")  # 替
    

    # 查看图片的形状
    # print(image.shape) # 4 360 320 

    # 查看图片的数据类型
    # print(image.dtype) # torch.uint8

    
    gaussian_blur = GaussianBlur(3)

    # 创建一个输入图像张量（假设为3x64x64的张量）
    # input_image = torch.randn(1, 3, 64, 64)
    
    # 将输入图像通过高斯模糊层进行模糊处理  
    # # 邓珺礼自己写的
    blurred_image = gaussian_blur(image)
    print("blurred_image.type:",blurred_image.type())

    blurred_image = blurred_image.squeeze(0)
    blurred_image = TF.to_pil_image(blurred_image)

    # 保存图像
    blurred_image.save("../高斯模糊后的_image.jpg")  # 替

    # 打印模糊后的图像形状
    # print(blurred_image.shape)
    
    ################# 高斯模糊的第二种方法 网上找的
    
    blur_layer = get_gaussian_kernel().cuda()

    blured_img = blur_layer(image)
    print("blured_img.type()",blured_img.type())
    
    blured_img = blured_img.squeeze(0)
    blured_img = TF.to_pil_image(blured_img)
    blured_img.save("../网上_高斯模糊后的_image.jpg")  # 替


    ################  torch 自带的


    blurrer = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
    blurred_imgs = blurrer(image)
    blurred_imgs = blurred_imgs.squeeze(0)
    blurred_imgs = TF.to_pil_image(blurred_imgs)
    blurred_imgs.save("../TF自带_高斯模糊后的_image.jpg")
    
    
    
    #####  luoyihao写的

    g_bluring = Gaussian_bluring(radius=5,sigma=5)
    # g_bluring.to(device)
    # print("g_bluring.type()",g_bluring.type())
    print("g_bluring.weight.type()",g_bluring.weight.type())
    print("g_bluring.weight.shape",g_bluring.weight.shape)
    # print("g_bluring.weight",g_bluring.weight)
    # print("image.type()",image.type())
    print("image.shape",image.shape) # 1 3 426 640
    print("image.max",torch.max(image)) # 1
    print("image.min",torch.min(image)) # 0
    img_bluring = g_bluring(image)

    img_bluring = img_bluring.squeeze(0)
    img_bluring = TF.to_pil_image(img_bluring)
    img_bluring.save("../罗翼昊写的_高斯模糊后的_image.jpg")