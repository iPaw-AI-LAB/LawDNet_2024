# 导入必要的库
import os
import cv2
import numpy as np
import sys
import time
sys.path.append('../')
from datetime import datetime
import torch
import random
from collections import OrderedDict
import warnings
import torchlm
import subprocess
from tqdm import tqdm
from config.config import DINetTrainingOptions
from torchlm.tools import faceboxesv2
from torchlm.models import pipnet
from tensor_processing import SmoothSqMask, FaceAlign
# from audio_processing import extract_deepspeech
from extract_deepspeech_pytorch2 import transcribe_and_process_audio

from deepspeech_pytorch.utils import load_decoder, load_model
from deepspeech_pytorch.configs.inference_config import TranscribeConfig, LMConfig
from deepspeech_pytorch.loader.data_loader import ChunkSpectrogramParser

from moviepy.editor import VideoFileClip, ImageSequenceClip, AudioFileClip

# 项目特定的模块
from models.LawDNet import LawDNet

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 使CUDA错误更易于追踪
warnings.filterwarnings("ignore")  # 初始化和配置警告过滤，忽略不必要的警告

# def timer_decorator(func):
#     def wrapper(*args, **kwargs):
#         start_time = time.time()
#         result = func(*args, **kwargs)
#         end_time = time.time()
#         print(f"{func.__name__} 耗时: {end_time - start_time:.2f} 秒")
#         return result
#     return wrapper

def preprocess_data(video_frames, 
                    #landmarks_list, 
                    opt, 
                    B, 
                    output_dir, 
                    video_path, 
                    device):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    preprocess_start = time.time()
    
    out_W = int(opt.mouth_region_size * 1.25)
    
    len_out = len(video_frames) // B * B
    video_frames = video_frames[:len_out]
    
    facealigner = FaceAlign(ratio=1.6, device=device)
    sqmasker = SmoothSqMask(device=device).to(device)
    
    # 检查是否存在预处理后的张量文件
    preprocessed_data_path = os.path.join(output_dir, f'{video_name}_preprocessed_data.pth')
    if os.path.exists(preprocessed_data_path):
        print("从本地加载预处理数据...")
        preprocessed_data = torch.load(preprocessed_data_path)
        reference_tensor = preprocessed_data['reference_tensor'].to(device)
        all_feed_tensor_masked = preprocessed_data['all_feed_tensor_masked'].to(device)
        all_source_tensor = preprocessed_data['all_source_tensor'].to(device)
        all_affine_matrix = preprocessed_data['all_affine_matrix'].to(device)
    else:
        print("预处理数据并保存到本地...")
        # 预处理参考帧
        reference_index = torch.randint(0, len(video_frames), (5,)).tolist()
        reference_tensor = torch.tensor(video_frames[reference_index], dtype=torch.float).to(device)
        reference_tensor = reference_tensor.permute(0, 3, 1, 2)

        landmarks_list = save_landmarks(video_frames, video_path, output_dir, device)
        reference_landmarks = torch.tensor(landmarks_list[reference_index], dtype=torch.float).to(device)
        
        reference_tensor, _, _ = facealigner(reference_tensor, reference_landmarks, out_W=out_W)
        reference_tensor = reference_tensor / 255.0
        
        # 创建视频帧的副本以确保连续的内存布局
        video_frames = np.ascontiguousarray(video_frames[:len_out])
        
        # 预处理所有帧
        all_source_tensor = torch.tensor(video_frames, dtype=torch.float).to(device).permute(0, 3, 1, 2)
        all_landmarks_tensor = torch.tensor(landmarks_list[:len_out], dtype=torch.float).to(device)
        all_feed_tensor, _, all_affine_matrix = facealigner(all_source_tensor, all_landmarks_tensor, out_W=out_W)
        all_feed_tensor_masked = sqmasker(all_feed_tensor / 255.0)
        
        # 保存预处理后的数据
        torch.save({
            'reference_tensor': reference_tensor.cpu(),
            'all_feed_tensor_masked': all_feed_tensor_masked.cpu(),
            'all_source_tensor': all_source_tensor.cpu(),
            'all_affine_matrix': all_affine_matrix.cpu()
        }, preprocessed_data_path)
    
    preprocess_end = time.time()
    print(f"预处理耗时: {preprocess_end - preprocess_start:.2f} 秒")

    return reference_tensor, all_feed_tensor_masked, all_source_tensor, all_affine_matrix


def load_and_initialize_model(lawdnet_model_path, opt, device):
    net_g = LawDNet(opt.source_channel, opt.ref_channel, opt.audio_channel, 
                    opt.warp_layer_num, opt.num_kpoints, opt.coarse_grid_size).to(device)
    print(f"正在加载 LawDNet 模型: {lawdnet_model_path}")
    
    checkpoint = torch.load(lawdnet_model_path, map_location=device)
    state_dict = checkpoint['state_dict']['net_g']
    new_state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())
    net_g.load_state_dict(new_state_dict)
    net_g.eval()
    
    return net_g


def read_video_np(video_path, start_time_sec, max_frames=None):
    assert os.path.exists(video_path), f"Video file not found: {video_path}"
    cap = cv2.VideoCapture(video_path)
    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Original FPS: {fps}")
    
    # 如果视频帧率不是 25 fps，则进行转换
    if fps != 25:
        print("Converting video to 25 fps...")
        temp_video_path = os.path.splitext(video_path)[0] + "_25fps.mp4"
        convert_video_to_25fps(video_path, temp_video_path)
        cap.release()
        cap = cv2.VideoCapture(temp_video_path)
        video_path = temp_video_path

    # 计算从第几帧开始读取
    start_frame = int(start_time_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        i += 1
        if max_frames is not None and i >= max_frames:
            break
    cap.release()
    return frames


def save_video_frames(video_path, start_time_sec, max_frames, output_dir):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames = read_video_np(video_path, start_time_sec, max_frames)
    np.save(os.path.join(output_dir, f'{video_name}_video_frames.npy'), frames)
    print(f"视频帧已保存到: {os.path.join(output_dir, f'{video_name}_video_frames.npy')}")
    return frames


def load_video_frames(video_path, output_dir):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames_path = os.path.join(output_dir, f'{video_name}_video_frames.npy')
    if os.path.exists(frames_path):
        print(f"从本地加载视频帧: {frames_path}")
        return np.load(frames_path)
    else:
        print("未找到本地保存的视频帧")
        return None
    

def reference(model, masked_source, reference, audio_tensor):
    with torch.no_grad():
        return model(masked_source, reference, audio_tensor)
    

def convert_video_to_25fps(input_video_path, output_video_path):
    command = [
        "ffmpeg",
        "-i", input_video_path,  # 输入视频文件
        "-r", "25",  # 设置输出视频帧率为 25 fps
        output_video_path  # 输出视频文件
    ]
    subprocess.run(command, check=True)


def save_video_with_audio(outframes, output_video_path, audio_path, output_dir, fps=25):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置临时文件路径
    temp_video_path = os.path.join(output_dir, "temp_video.mp4")

    # 将列表转换为 NumPy 数组并反转颜色通道
    outframes_array = np.array(outframes)
    outframes_bgr = outframes_array[..., ::-1]

    # 获取视频尺寸
    height, width = outframes_bgr[0].shape[:2]
    
    # 使用 OpenCV 直接写入视频文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    
    for frame in outframes_bgr:
        out.write(frame.astype(np.uint8))
    out.release()

    # 首先尝试使用 moviepy
    try:
        print("尝试使用 moviepy...")
        from moviepy.editor import VideoFileClip, AudioFileClip
        video_clip = VideoFileClip(temp_video_path)
        audio_clip = AudioFileClip(audio_path)
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac')
        print("使用 moviepy 成功合并视频和音频")
    except Exception as e:
        print(f"moviepy 失败: {e}")
        print("尝试使用 ffmpeg...")
        try:
            ffmpeg_command = [
                "ffmpeg",
                "-i", temp_video_path,
                "-i", audio_path,
                "-c:v", "libx264",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                output_video_path
            ]
            subprocess.run(ffmpeg_command, check=True)
            print("使用 ffmpeg 成功合并视频和音频")
        except Exception as e:
            print(f"ffmpeg 也失败了: {e}")
            raise Exception("无法合并视频和音频")

    # 删除临时文件
    os.remove(temp_video_path)

    return output_video_path


def save_landmarks(video_frames, video_path, output_dir, device):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    torchlm.runtime.bind(faceboxesv2(device=device))
    torchlm.runtime.bind(pipnet(backbone="resnet18", pretrained=True, num_nb=10, num_lms=68, net_stride=32,
                                input_size=256, meanface_type="300w", map_location=device.__str__()))

    landmarks_path = os.path.join(output_dir, f'{video_name}_landmarks.npy')
    if os.path.exists(landmarks_path):
        print(f"从本地加载 landmarks: {landmarks_path}")
        return np.load(landmarks_path)
    
    print("提取并保存 landmarks...")
    landmarks_list = []
    for frame in tqdm(video_frames, desc='处理视频帧，提取 landmarks...'):
        landmarks, bboxes = torchlm.runtime.forward(frame)
        highest_trust_index = np.argmax(bboxes[:, 4])
        most_trusted_landmarks = landmarks[highest_trust_index]
        landmarks_list.append(most_trusted_landmarks)
    
    landmarks_array = np.array(landmarks_list)
    np.save(landmarks_path, landmarks_array)
    print(f"Landmarks 已保存到: {landmarks_path}")
    return landmarks_array


def generate_video_with_audio(video_frames, 
                              #landmarks_list,
                              deepspeech_tensor,
                              audio_path, 
                              net_g,
                              output_dir,
                              BatchSize,
                              opt,
                              output_name,
                              device,
                              video_path):
    start_time = time.time()
    
    # 预处理部分
    preprocess_start = time.time()
    B = BatchSize
    # out_W = int(opt.mouth_region_size * 1.25)
    
    len_out = min(len(video_frames), deepspeech_tensor.shape[0]) // B * B
    video_frames = video_frames[:len_out]
    deepspeech_tensor = deepspeech_tensor[:len_out].to(device)
    
    facealigner = FaceAlign(ratio=1.6, device=device)
    # sqmasker = SmoothSqMask(device=device).to(device)
    
    # 检查是否存在预处理后的张量文件
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    preprocessed_data_path = os.path.join(output_dir, f'{video_name}_preprocessed_data.pth')
    # preprocessed_data_path = os.path.join(output_dir, 'douyin_preprocessed_data.pth')
    if os.path.exists(preprocessed_data_path):
        print("从本地加载预处理数据...")
        preprocessed_data = torch.load(preprocessed_data_path)
        reference_tensor = preprocessed_data['reference_tensor'].to(device)
        all_feed_tensor_masked = preprocessed_data['all_feed_tensor_masked'].to(device)
        all_source_tensor = preprocessed_data['all_source_tensor'].to(device)
        all_affine_matrix = preprocessed_data['all_affine_matrix'].to(device) 
    else:
        print("预处理数据并保存到本地...")
        # 预处理参考帧
        reference_tensor, \
        all_feed_tensor_masked, \
        all_source_tensor, \
        all_affine_matrix = preprocess_data(video_frames, 
                                            #landmarks_list, 
                                            opt, 
                                            B, 
                                            output_dir,
                                            video_path,
                                            device)
        
        # 保存预处理后的数据
        torch.save({
            'reference_tensor': reference_tensor.cpu(),
            'all_feed_tensor_masked': all_feed_tensor_masked.cpu(),
            'all_source_tensor': all_source_tensor.cpu(),
            'all_affine_matrix': all_affine_matrix.cpu()
        }, preprocessed_data_path)
    
    # 预处理参考张量
    reference_tensor_expanded = reference_tensor.unsqueeze(0).expand(B, -1, -1, -1, -1).reshape(B, 5 * 3, all_feed_tensor_masked.shape[2], all_feed_tensor_masked.shape[3])

    outframes = np.zeros_like(video_frames[:len_out])
    preprocess_end = time.time()
    print(f"预处理耗时: {preprocess_end - preprocess_start:.2f} 秒")
    
    # 主循环部分
    main_loop_start = time.time()
    with torch.no_grad():
        for i in tqdm(range(len_out // B), desc="Processing batches"):
            start_idx = i * B
            end_idx = (i + 1) * B
            
            feed_tensor_masked = all_feed_tensor_masked[start_idx:end_idx]
            audio_tensor = deepspeech_tensor[start_idx:end_idx].to(device)
            
            output_B = net_g(feed_tensor_masked, reference_tensor_expanded, audio_tensor).float().clamp_(0, 1)
            
            outframes_B = facealigner.recover(output_B * 255.0, all_source_tensor[start_idx:end_idx], all_affine_matrix[start_idx:end_idx]).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            outframes[start_idx:end_idx] = outframes_B

    outframes = outframes.astype(np.uint8)
    main_loop_end = time.time()
    print(f"主循环耗时: {main_loop_end - main_loop_start:.2f} 秒")
    
    # 生成输出视频路径
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_video_path = os.path.join(output_dir, f"{timestamp_str}_{output_name}.mp4")
    
    # 保存视频并添加音频
    end_time = time.time()
    print(f"到合成视频前，耗时: {end_time - start_time:.2f} 秒")
    return save_video_with_audio(outframes, output_video_path, audio_path, output_dir, fps=25)


if __name__ == "__main__":
    # 设置参数
    total_start_time = time.time()
    video_path = './data/figure_five_two.mp4'
    audio_path = "./data/青岛3.wav" 
    # audio_path = "../../Chat_TTS/test_success.wav" 
    output_dir = './output_video'
    lawdnet_model_path = "./pretrain_model/checkpoint_epoch_170.pth"
    BatchSize = 5
    mouthsize = '288'
    gpu_index = 0
    output_name = '速度测试'
    dp2_path = './pretrain_model/LibriSpeech_Pretrained_v3.ckpt'
    start_time_sec = 0
    max_frames = None
    precision = 16

    # 设置设备
    device = f'cuda:{gpu_index}' if torch.cuda.is_available() and gpu_index >= 0 else 'cpu'

    # 初始化模型配置
    args = ['--mouth_region_size', mouthsize]
    opt = DINetTrainingOptions().parse_args(args)

    # 加载和初始化模型
    net_g = load_and_initialize_model(lawdnet_model_path, opt, device)

    # 准备数据
    os.makedirs(output_dir, exist_ok=True)
    video_frames = load_video_frames(video_path, output_dir)
    if video_frames is None:
        video_frames = save_video_frames(video_path, start_time_sec, max_frames, output_dir)

    video_frames = np.array(video_frames, dtype=np.float32)
    video_frames = video_frames[..., ::-1]
    
    # landmarks_list = save_landmarks(video_frames, video_path, output_dir, device)
    
    # 提取 DeepSpeech 特征
    deepspeech_start_time = time.time()

    if not os.path.exists(dp2_path):
        raise FileNotFoundError('Please download the pretrained model of DeepSpeech.')
    
    dp2_model = load_model(device=device, model_path=dp2_path)
    print("正在加载 DeepSpeech pytorch 2 模型...")

    cfg = TranscribeConfig()
    spect_parser = ChunkSpectrogramParser(audio_conf=dp2_model.spect_cfg, normalize=True)
    decoder = load_decoder(
        labels=dp2_model.labels,
        cfg=cfg.lm  
    )

    dp2_model.eval()

    deepspeech_tensor, _ = transcribe_and_process_audio(
        audio_path=audio_path,
        model_path=dp2_path,
        device=device,
        precision=precision,
        model=dp2_model,
        cfg=cfg,
        spect_parser=spect_parser,
        decoder=decoder
    )
    deepspeech_end_time = time.time()
    print(f"DeepSpeech 耗时: {deepspeech_end_time - deepspeech_start_time:.2f} 秒") 

    # 生成视频
    result_video_path = generate_video_with_audio(video_frames, 
                                                  #landmarks_list,
                                                  deepspeech_tensor,
                                                  audio_path, 
                                                  net_g,
                                                  output_dir,
                                                  BatchSize,
                                                  opt,
                                                  output_name,
                                                  device,
                                                  video_path)

    print(f"生成的视频保存在: {result_video_path}")
    total_end_time = time.time()
    print(f"总耗时: {total_end_time - total_start_time:.2f} 秒")
    print(f"生成的视频保存在: {result_video_path}")
