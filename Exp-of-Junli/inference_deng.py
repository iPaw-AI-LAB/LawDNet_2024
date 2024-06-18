# 导入必要的库
import os
import cv2
import numpy as np
import sys
sys.path.append('../')
import subprocess
from datetime import datetime
import torch
import random
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from ffmpy import FFmpeg
from collections import OrderedDict
import warnings
import torchlm
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from config.config import DINetTrainingOptions
from torchlm.tools import faceboxesv2
from torchlm.models import pipnet
from tensor_processing import *
from audio_processing import extract_deepspeech

# 项目特定的模块
from models.LawDNet import LawDNet

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 使CUDA错误更易于追踪
warnings.filterwarnings("ignore")  # 初始化和配置警告过滤，忽略不必要的警告

'''
模型路径
'''


# 设置模型文件路径
deepspeech_model_path = "../asserts/output_graph.pb"  # DeepSpeech模型文件路径

lawdnet_model_path =  "/home/dengjunli/data/dengjunli/autodl拿过来的/DINet-update/output/training_model_weight/288-mouth-CrossAttention-插值coarse-to-fine-2/clip_training_256/checkpoint_epoch_120.pth"

# 其他模型文件路径，可根据需要取消注释使用
# lawdnet_model_path = "../asserts/training_model_weight-标准框60关键点8/clip_training_256/netG_model_epoch_38.pth"  # 标准框，60关键点5训练的模型
## ⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️最好的
# lawdnet_model_path = "../asserts/training_model_weight-360分辨率正式模型备份/clip_training_256/netG_model_epoch_60.pth"  # 大嘴标准框，60关键点5训练的模型

'''

'''

args = ['--opt.mouth_region_size','288']
        
opt = DINetTrainingOptions().parse_args(args)

os.environ["CUDA_VISIBLE_DEVICES"] = str(2) # 选择GPU
print(" torch.cuda.device_count() ",torch.cuda.device_count())
print(" torch.cuda.get_device_name(0) ",torch.cuda.get_device_name(0))

gpu_index = 0 #3 # 选择GPU 
if torch.cuda.is_available() and gpu_index >= 0 and torch.cuda.device_count() > gpu_index:
    device = f'cuda:{gpu_index}'
else:
    device = 'cpu'

# 设置随机种子
random.seed(opt.seed + gpu_index)
np.random.seed(opt.seed+ gpu_index)
torch.cuda.manual_seed_all(opt.seed+gpu_index)
torch.manual_seed(opt.seed+gpu_index)

# 导入配置文件更新opt

out_W = int(opt.mouth_region_size * 1.25) # 输出视频的宽度 # 360
B = 10  # batchsize 并行化推理
output_name = 'seed-5kp-60standard—epoch60-720P-复现'

video_path = './template/26-_主播说联播_把小事当大事干-也能通过平凡成就非凡.mp4'

audio_path = './template/taylor-20s.wav'

# 如果是从音频文件提取DeepSpeech特征
# deepspeech_tensor, _ = extract_deepspeech(audio_path, deepspeech_model_path)
# torch.save(deepspeech_tensor, './template/template_audio_deepspeech.pt')

# 直接从已保存的Tensor文件加载
deepspeech_tensor = torch.load('./template/template_audio_deepspeech.pt')
print(f"Loaded DeepSpeech features tensor shape: {deepspeech_tensor.shape}")

max_frames = 600  # 使用本段视频的帧数  # None

'''

'''

def read_video_np(video_path, max_frames=None):
    """
    读取视频文件并将其转换为帧序列。
    
    参数:
    video_path (str): 视频文件的路径。
    max_frames (int): 最大读取帧数。默认为800，如果设置为None，则读取整个视频。
    
    返回:
    list: 包含视频帧的列表。
    """
    cap = cv2.VideoCapture(video_path)  # 创建视频捕获对象
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频帧数
    H, W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频的高和宽
    print('video length: ', length)
    print('video size: ', 'H--', H, 'W--', W)
    
    frames = []  # 初始化帧列表
    i = 0  # 帧计数器
    while True:
        ret, frame = cap.read()  # 读取下一帧
        if not ret:
            break  # 如果读取失败，结束循环
        frames.append(frame)  # 如果读取成功，添加到列表中
        
        i += 1
        if max_frames is not None and i >= max_frames:  # 检查是否达到最大帧数限制
            break

    cap.release()  # 释放视频捕获对象
    return frames

'''

'''




# 从视频路径中提取视频文件名（无扩展名）
video_name = os.path.splitext(os.path.basename(video_path))[0]

# 将视频名称添加到输出名称中，以创建唯一的输出标识符
output_name = f"{output_name}_{video_name}"

# 读取视频帧并转换为numpy数组
video_frames = read_video_np(video_path, max_frames)  # 读取视频帧
video_frames = np.array(video_frames, dtype=np.float32)

# 将视频帧从BGR转换为RGB格式
video_frames = video_frames[..., ::-1]  # 使用...保持前面的维度不变，只反转最后一个维度

# 打印视频帧数组的形状
print("使用的视频 ",video_frames.shape)

'''

'''


# 确定输出长度，以适应最小的帧数或特征长度，并确保它是batch size的倍数
len_out = min(len(video_frames), deepspeech_tensor.shape[0]) // B * B

# 调整视频帧和DeepSpeech特征数组的长度以匹配len_out
video_frames = video_frames[:len_out]
deepspeech_tensor = deepspeech_tensor[:len_out]

'''

'''

# 加载LawDNet模型
net_g = LawDNet(opt.source_channel, opt.ref_channel, opt.audio_channel, 
                opt.warp_layer_num, opt.num_kpoints, opt.coarse_grid_size).to(device) # 2个输出通道，5个关键点
checkpoint = torch.load(lawdnet_model_path)
state_dict = checkpoint['state_dict']['net_g']

# 修正模型权重命名并加载
new_state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())  # 移除前缀'module.'
net_g.load_state_dict(new_state_dict)
net_g.eval()

# 初始化面部对齐和平滑平方掩码器
facealigner = FaceAlign(ratio=1.6, device=device) # 固定比例1.6
sqmasker = SmoothSqMask(device=device).to(device) # 直接使用cuda()，假设环境已经设定好


'''

'''

# 绑定模型到指定设备
torchlm.runtime.bind(faceboxesv2(device=device))
torchlm.runtime.bind(pipnet(backbone="resnet18", pretrained=True, num_nb=10, num_lms=68, net_stride=32,
                            input_size=256, meanface_type="300w", map_location=device.__str__()))

# 提取第一帧的landmarks作为初始化
landmarks, _ = torchlm.runtime.forward(video_frames[0])

# 批量处理视频帧以提取landmarks
# 初始化landmarks列表
landmarks_list = []

# 对视频帧进行遍历，逐帧提取landmarks
for frame in tqdm(video_frames, desc='Processing video frames, extracting landmarks...'):
    # 对当前帧进行landmark和bounding box的提取
    landmarks, bboxes = torchlm.runtime.forward(frame)
    
    # 根据置信度（bounding box的第五个元素）选择最可信的landmark
    highest_trust_index = np.argmax(bboxes[:, 4])
    most_trusted_landmarks = landmarks[highest_trust_index]
    
    # 将选中的landmarks添加到列表中
    landmarks_list.append(most_trusted_landmarks)

# 转换landmarks列表为numpy数组
landmarks_list = np.array(landmarks_list)


# 转换landmarks列表为numpy数组
landmarks_list = np.array(landmarks_list)


'''

'''

# 随机选择5帧作为参考帧，或者手动指定参考帧索引
reference_index = torch.randint(0, len(video_frames), (5,)).tolist()
# reference_index = [6, 0, 0, 0, 0]

# 从视频帧中提取参考帧并转换为张量
reference_tensor = torch.tensor(video_frames[reference_index], dtype=torch.float).to(device)
# 调整张量维度以匹配网络输入 [batch_size, channels, height, width]
reference_tensor = reference_tensor.permute(0, 3, 1, 2)

# 从landmarks列表中提取对应的参考landmarks
reference_landmarks = torch.tensor(landmarks_list[reference_index], dtype=torch.float).to(device)

# 使用facealigner调整参考帧和参考landmarks，确保它们与目标分辨率匹配
reference_tensor, _, _ = facealigner(reference_tensor, reference_landmarks, out_W=out_W)

# 显示第一帧参考图像
# plt.imshow(reference_tensor[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
# plt.axis('off')  # 不显示坐标轴
# plt.title('参考图像')
# plt.savefig('生成图source_img.png', bbox_inches='tight', pad_inches=0)

# 将参考张量的值归一化到[0, 1]范围
reference_tensor = reference_tensor / 255.0
print(f"reference_tensor.shape: {reference_tensor.shape}")


'''

'''

def reference(model, masked_source, reference, audio_tensor):
    with torch.no_grad():
        return model(masked_source, reference, audio_tensor)

visualize_every_n_batches = 30  # 指定可视化的批次间隔
outframes = np.zeros_like(video_frames)  # 初始化输出帧数组

for i in tqdm(range(len_out // B), desc="处理批次"):
    # 准备源视频帧和landmarks的张量
    source_tensor = torch.tensor(video_frames[i * B:(i + 1) * B].copy(), dtype=torch.float).to(device).permute(0, 3, 1, 2)
    landmarks_tensor = torch.tensor(landmarks_list[i * B:(i + 1) * B], dtype=torch.float).to(device)
    
    # 使用facealigner对视频帧进行预处理
    feed_tensor, _, affine_matrix = facealigner(source_tensor, landmarks_tensor, out_W=out_W)
    _, C, H, W = feed_tensor.shape

    # 应用掩码处理
    feed_tensor_masked = sqmasker(feed_tensor / 255.0)
    
    # 准备参考帧和音频张量
    reference_tensor_B = reference_tensor.unsqueeze(0).expand(B, -1, -1, -1, -1).reshape(B, 5 * 3, H, W)
    print("reference_tensor.shape", reference_tensor.shape)
    print("reference_tensor_B.shape", reference_tensor_B.shape)
    audio_tensor = deepspeech_tensor[i * B:(i + 1) * B].to(device)
    
    # 生成输出
    # print("feed_tensor_masked.shape", feed_tensor_masked.shape)
    # print("reference_tensor_B.shape", reference_tensor_B.shape)
    # print("audio_tensor.shape", audio_tensor.shape)
    
    output_B = reference(net_g, feed_tensor_masked, reference_tensor_B, audio_tensor).float().clamp_(0, 1)
    
    # 恢复到原始尺寸和位置
    outframes_B = facealigner.recover(output_B * 255.0, source_tensor, affine_matrix).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    
    # 更新输出帧数组
    outframes[i * B:(i + 1) * B] = outframes_B

    # 可视化输出
    # if i % visualize_every_n_batches == 0:
    #     plt.figure(figsize=(10, 5))
    #     plt.imshow(outframes_B[0])
    #     plt.title(f"Batch {i} - First output image")
    #     # plt.axis('off')
    #     plt.show()

'''

'''


outframes = outframes.astype(np.uint8) 
outframes = outframes[:,:,:,::-1] # RGB to BGR

# 准备输出视频的路径
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
output_path = f'./{timestamp_str}.mp4'

# 初始化视频写入器
fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter(output_path, fourcc, 25, (video_frames.shape[2], video_frames.shape[1]), True)

# 将处理后的帧写入视频文件
for frame in outframes:
    videoWriter.write(frame)

# 释放视频写入器资源
videoWriter.release()

'''

'''


def video_add_audio(video_path: str, audio_path: str, output_dir: str) -> str:
    # 检查output_dir是否存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 检查音频格式是否支持
    _ext_audio = audio_path.split('.')[-1]
    if _ext_audio not in ['mp3', 'wav']:
        raise ValueError('Audio format not supported.')

    # 根据音频格式选择编码器
    _codec = 'copy' if _ext_audio == 'mp3' else 'aac'

    # 构造输出视频文件的完整路径
    video_basename_without_ext = os.path.splitext(os.path.basename(video_path))[0]
    output_video_path = os.path.join(output_dir, f"{video_basename_without_ext}_{output_name}.mp4")

    # 使用ffmpeg合并视频和音频
    ff_command = f"ffmpeg -i \"{video_path}\" -i \"{audio_path}\" -map 0:v -map 1:a -c:v copy -c:a {_codec} -shortest \"{output_video_path}\""
    os.system(ff_command)

    # 删除临时视频文件
    os.remove(video_path)

    return output_video_path

# 调用函数，将音频添加到视频
result_video_path = video_add_audio(output_path, audio_path, './output_video')

print(f"Generated video with audio at: {result_video_path}")