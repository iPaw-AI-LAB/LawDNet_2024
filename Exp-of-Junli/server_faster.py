from flask import Flask, request, send_file
import os
import torch
import numpy as np
from collections import OrderedDict
import torchlm
from torchlm.tools import faceboxesv2
from torchlm.models import pipnet
import sys
sys.path.append('../')
from config.config import DINetTrainingOptions
from models.LawDNet import LawDNet
from tensor_processing import SmoothSqMask, FaceAlign
from extract_deepspeech_pytorch2 import transcribe_and_process_audio

from deepspeech_pytorch.utils import load_decoder, load_model
from deepspeech_pytorch.configs.inference_config import TranscribeConfig, LMConfig
from deepspeech_pytorch.loader.data_loader import ChunkSpectrogramParser

import cv2
from moviepy.editor import VideoFileClip, AudioFileClip
import time
from datetime import datetime
from tqdm import tqdm
import threading
import queue
import subprocess

from inference_function_DP2_faster import save_video_frames, \
                                            load_video_frames, \
                                            save_landmarks, \
                                            save_video_with_audio, \
                                            generate_video_with_audio, \
                                            preprocess_data

app = Flask(__name__)

# 全局变量
device = None
net_g = None
opt = None
facealigner = None
sqmasker = None
video_frames = None
landmarks_list = None
reference_tensor = None
all_feed_tensor_masked = None
all_source_tensor = None
all_affine_matrix = None

dp2_path = None
dp2_model = None
cfg = None
spect_parser = None
decoder = None

output_dir = None
B = None
scripted_net_g = None  # 添加 scripted_net_g


def initialize_server():
    global device, net_g, opt, facealigner, sqmasker, video_frames, video_frames_length, video_frames_shape,landmarks_list
    global reference_tensor, all_feed_tensor_masked, all_source_tensor, all_affine_matrix
    global dp2_path, dp2_model, cfg, spect_parser, decoder, output_dir
    global B, scripted_net_g  # 添加 scripted_net_g

    # 设置参数
    B = 1
    video_path = './data/figure_five_two.mp4' 
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    lawdnet_model_path = "./pretrain_model/checkpoint_epoch_170.pth"
    mouthsize = '288'
    gpu_index = 0
    dp2_path = './pretrain_model/LibriSpeech_Pretrained_v3.ckpt'
    output_dir = './output_video'
    device = f'cuda:{gpu_index}' if torch.cuda.is_available() and gpu_index >= 0 else 'cpu'

    '''
    dp2
    '''
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
    '''
    dp2
    '''

    # 初始化模型配置
    args = ['--mouth_region_size', mouthsize]
    opt = DINetTrainingOptions().parse_args(args)

    # 加载和初始化模型
    net_g = load_and_initialize_model(lawdnet_model_path, opt, device)

    # 创建 TorchScript 模型
    # print("正在创建 TorchScript 模型...")
    # B = 1  # 使用批次大小为1进行追踪
    # example_feed_tensor = torch.randn(B, 3, 468, 320).to(device)
    # example_reference_tensor = torch.randn(B, 5*3, 468, 320).to(device)
    # example_audio_tensor = torch.randn(B, 29, 5).to(device)  # 假设音频张量的形状
    
    # scripted_net_g = torch.jit.trace(net_g, (example_feed_tensor, example_reference_tensor, example_audio_tensor))
    # print("TorchScript 模型创建完成")

    # 准备数据
    os.makedirs(output_dir, exist_ok=True)

    # landmarks_list = save_landmarks(video_frames, video_path, output_dir, device)

    # 预处理数据
    facealigner = FaceAlign(ratio=1.6, device=device)
    sqmasker = SmoothSqMask(device=device).to(device)
    
    preprocessed_data_path = os.path.join(output_dir, f'{video_name}_preprocessed_data.pth')
    if os.path.exists(preprocessed_data_path):
        video_frames = load_video_frames(video_path, output_dir)
        video_frames_length = len(video_frames)
        video_frames_shape = video_frames.shape
        del video_frames
        torch.cuda.empty_cache()

        print("从本地加载预处理数据...")
        preprocessed_data = torch.load(preprocessed_data_path)
        reference_tensor = preprocessed_data['reference_tensor'].to(device)
        all_feed_tensor_masked = preprocessed_data['all_feed_tensor_masked'].to(device)
        all_source_tensor = preprocessed_data['all_source_tensor'].to(device)
        all_affine_matrix = preprocessed_data['all_affine_matrix'].to(device)
    else:
        video_frames = load_video_frames(video_path, output_dir)
        if video_frames is None:
            video_frames = save_video_frames(video_path, 0, None, output_dir)

        video_frames = np.array(video_frames, dtype=np.float32)
        video_frames = video_frames[..., ::-1]

        print("预处理数据并保存到本地...")
        reference_tensor, \
        all_feed_tensor_masked, \
        all_source_tensor, \
        all_affine_matrix = preprocess_data(video_frames, 
                                            # landmarks_list, 
                                            opt, 
                                            B,
                                            output_dir,
                                            video_path,
                                            device
                                            )
        torch.save({
            'reference_tensor': reference_tensor.cpu(),
            'all_feed_tensor_masked': all_feed_tensor_masked.cpu(),
            'all_source_tensor': all_source_tensor.cpu(),
            'all_affine_matrix': all_affine_matrix.cpu()
        }, preprocessed_data_path)

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

@app.route('/generate_video', methods=['POST'])
def generate_video():
    if 'audio' not in request.files:
        return '没有上传音频文件', 400
    
    audio_file = request.files['audio']
    audio_path = os.path.join(output_dir, 'temp_audio.wav')
    audio_file.save(audio_path)

    # 提取 DeepSpeech 特征

    
    deepspeech_tensor, _ = transcribe_and_process_audio(
        audio_path=audio_path,
        model_path=dp2_path,
        device=device,
        precision=16,
        model=dp2_model,
        cfg=cfg,
        spect_parser=spect_parser,
        decoder=decoder
    )

    # 生成视频
    result_video_path = generate_video_with_audio(deepspeech_tensor, audio_path)

    return send_file(result_video_path, mimetype='video/mp4')

def write_frames(video_writer, frame_queue):
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        video_writer.write(frame)
        frame_queue.task_done()

def generate_video_with_audio(deepspeech_tensor, audio_path):
    global reference_tensor, all_feed_tensor_masked, all_source_tensor, all_affine_matrix, scripted_net_g
    
    len_out = min(video_frames_length, deepspeech_tensor.shape[0]) // B * B
    deepspeech_tensor = deepspeech_tensor[:len_out].to(device)
    
    # reference_tensor_expanded = reference_tensor.unsqueeze(0).expand(B, -1, -1, -1, -1).reshape(B, 5 * 3, all_feed_tensor_masked.shape[2], all_feed_tensor_masked.shape[3])
    reference_tensor_expanded = reference_tensor.unsqueeze(0).expand(B, -1, -1, -1, -1)
    reference_tensor_expanded = reference_tensor_expanded.reshape(B, 5 * 3, all_feed_tensor_masked.shape[2], all_feed_tensor_masked.shape[3])

    # 初始化 VideoWriter 和帧队列
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    temp_video_path = os.path.join(output_dir, f"{timestamp_str}_temp_output.mp4")
    frame_queue = queue.Queue(maxsize=500)  # 限制队列大小以控制内存使用
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width = video_frames_shape[1:3]
    video_writer = cv2.VideoWriter(temp_video_path, fourcc, 25, (width, height))

    # 启动帧写入线程
    writer_thread = threading.Thread(target=write_frames, args=(video_writer, frame_queue))
    writer_thread.start()

    with torch.no_grad():
        for i in tqdm(range(len_out // B), desc="生成视频帧"):
            start_idx = i * B
            end_idx = (i + 1) * B
            
            feed_tensor_masked = all_feed_tensor_masked[start_idx:end_idx]
            audio_tensor = deepspeech_tensor[start_idx:end_idx].to(device)
            with torch.cuda.amp.autocast():
                output_B = net_g(feed_tensor_masked, reference_tensor_expanded, audio_tensor).float().clamp_(0, 1)
            outframes_B = facealigner.recover(output_B * 255.0, all_source_tensor[start_idx:end_idx], all_affine_matrix[start_idx:end_idx]).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            
            for frame in outframes_B:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_queue.put(frame_bgr)

    # 通知写入线程完成
    frame_queue.put(None)
    writer_thread.join()
    video_writer.release()

    # 返回静音视频，没拼接音频
    print("静音视频生成完成")
    return temp_video_path

    # # 添加音频
    # try:
    #     video = VideoFileClip(temp_video_path)
    #     audio = AudioFileClip(audio_path)
    #     final_clip = video.set_audio(audio)
    #     final_output_path = os.path.join(output_dir, f"{timestamp_str}_output_with_audio.mp4")
    #     final_clip.write_videofile(final_output_path, 
    #                             fps=25, 
    #                             codec='libx264', 
    #                             audio_codec='aac',
    #                             preset='superfast',
    #                             bitrate='5000k',
    #                             threads=16,
    #                             ffmpeg_params=['-crf', '18'])
    # except Exception as e:
    #     print(f"moviepy 处理失败: {e}")
    #     print("尝试使用 ffmpeg...")
        
    #     final_output_path = os.path.join(output_dir, f"{timestamp_str}_output_with_audio.mp4")
    #     ffmpeg_command = [
    #         'ffmpeg',
    #         '-i', temp_video_path,
    #         '-i', audio_path,
    #         '-c:v', 'libx264',
    #         '-preset', 'ultrafast',
    #         '-crf', '23',
    #         '-c:a', 'aac',
    #         '-b:a', '192k',
    #         '-shortest',
    #         final_output_path
    #     ]
        
    #     try:
    #         subprocess.run(ffmpeg_command, check=True)
    #         print("ffmpeg 处理成功")
    #     except subprocess.CalledProcessError as e:
    #         print(f"ffmpeg 处理失败: {e}")
    #         raise

    # # 清理临时文件
    # os.remove(temp_video_path)

    # # 返回合并音视频的文件
    # return final_output_path

if __name__ == "__main__":
    initialize_server()
    app.run(host='0.0.0.0', port=5051)