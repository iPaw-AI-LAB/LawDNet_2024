import torch
import torch.nn as nn
import torch.nn.functional as F

class Gaussian_bluring(nn.Module):
  def __init__(self, radius=1,sigma=1,padding=0):
    super(Gaussian_bluring, self).__init__()

    self.sigma = sigma
    self.padding = padding
    self.radius = radius
    self.size = self.radius*2+1

    self.kernel = self.get_gaussian_kernel()

    self.weight = nn.Parameter(data=self.kernel.unsqueeze(0).unsqueeze(0), requires_grad=False)
 
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
    return gaussian_kernel


g_bluring = Gaussian_bluring(radius=5,sigma=5)
g_bluring.to(device)
img_bluring = g_bluring(input_batch)
#edges = sobel(input_batch[0,1:2,:,:])
    
    

    

if __name__ == '__main__':
    
    
    # import torchvision.transforms.functional as TF
    # from PIL import Image
    
    # # 创建一个大小为5的高斯模糊层（sigma默认为1.0）
    # import torchvision.io as io
    
    # # 读取图片
    # # image_path = 'path/to/your/image.jpg'
    # image = io.read_image('../000000020434.jpg')
    # # 在第一维度上增加一个batch维度
    # image = image.unsqueeze(0)

    # # 查看图片的形状
    # print(image.shape) # 4 360 320 

    # # 查看图片的数据类型
    # print(image.dtype) # torch.uint8
    
    # gaussian_blur = MultiChannelGaussianBlur(5)

    # # 创建一个输入图像张量（假设为3x64x64的张量）
    # # input_image = torch.randn(1, 3, 64, 64)
    

    # # 将输入图像通过高斯模糊层进行模糊处理
    # blurred_image = gaussian_blur(image)
    
    # image = TF.to_pil_image(blurred_image)

    # # 保存图像
    # image.save("../高斯模糊后的_image.jpg")  # 替

    # # 打印模糊后的图像形状
    # print(blurred_image.shape)
    
    # set the path that includes all models
   
    model_path = './pre_train'
    model_path = Path(model_path)
    image_path = Path('./testimg/IMG_6031.JPG')

    input_image = Image.open(image_path)
    input_image = input_image.convert("RGB")

    image_size = input_image.size
    print("image_size", image_size)

    image_size = 512

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    trans_totensor = transforms.Compose([transforms.Resize((image_size,image_size), interpolation=Image.BILINEAR),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor()])

    input_tensor = trans_totensor(input_image).to(device)
    
    
    

