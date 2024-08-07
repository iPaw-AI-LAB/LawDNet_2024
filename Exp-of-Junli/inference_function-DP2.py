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


from moviepy.editor import VideoFileClip, ImageSequenceClip, AudioFileClip

# 项目特定的模块
from models.LawDNet import LawDNet

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 使CUDA错误更易于追踪
warnings.filterwarnings("ignore")  # 初始化和配置警告过滤，忽略不必要的警告

def convert_video_to_25fps(input_video_path, output_video_path):
    command = [
        "ffmpeg",
        "-i", input_video_path,  # 输入视频文件
        "-r", "25",  # 设置输出视频帧率为 25 fps
        output_video_path  # 输出视频文件
    ]
    subprocess.run(command, check=True)

def generate_video_with_audio(video_path, 
                              audio_path, 
                            #   deepspeech_model_path='../asserts/output_graph.pb',
                              lawdnet_model_path='../output/training_model_weight/288-mouth-CrossAttention-插值coarse-to-fine-shengshu/clip_training_256/checkpoint_epoch_120.pth',
                              output_dir='./output_video',
                              BatchSize = 20,
                              mouthsize = '288' ,
                              gpu_index = 0,
                              output_name = '5kp-60standard—epoch120-720P-复现',
                              start_time_sec = 0,
                              max_frames = None,
                              dp2_path = './dp2_models/LibriSpeech_Pretrained_v3.ckpt',
                              ):
    
    args = ['--mouth_region_size', mouthsize]
    opt = DINetTrainingOptions().parse_args(args)
    
    if torch.cuda.is_available() and gpu_index >= 0 and torch.cuda.device_count() > gpu_index:
        device = f'cuda:{gpu_index}'
        print(f"Using GPU: {gpu_index}")
    else:
        device = 'cpu'
        print("Using CPU")
    
    random.seed(opt.seed + gpu_index)
    np.random.seed(opt.seed + gpu_index)
    torch.cuda.manual_seed_all(opt.seed + gpu_index)
    torch.manual_seed(opt.seed + gpu_index)
    
    out_W = int(opt.mouth_region_size * 1.25)
    print("opt.mouth_region_size: ", opt.mouth_region_size)
    print(f"out_W: {out_W}")
    B = BatchSize
    
    # 如果是从音频文件提取DeepSpeech特征
    print("Extracting DeepSpeech features from audio file...")
    start_time = time.time()
    # deepspeech_tensor, _ = extract_deepspeech(audio_path, deepspeech_model_path)
    # print("deepspeech_tensor shape: ", deepspeech_tensor.shape)
    # import pdb; pdb.set_trace()

    model_path = dp2_path
    precision = 16

    deepspeech_tensor, _ = transcribe_and_process_audio(
        audio_path=audio_path,
        model_path=model_path,
        device=device,
        precision=precision
    )


    end_time = time.time()
    print(f"Running time: {end_time - start_time} seconds for extracting DeepSpeech features.")
    # torch.save(deepspeech_tensor, './template/template_audio_deepspeech.pt')

    ## 测试时：直接读取本地deepspeech_tensor
    # print("Loading DeepSpeech features from local file...")
    # deepspeech_tensor = torch.load('./template/template_audio_deepspeech.pt')

    # import pdb; pdb.set_trace()

    
    def read_video_np(video_path, start_time_sec, max_frames):
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
    
    # video_frames = read_video_np(video_path, max_frames=deepspeech_tensor.shape[0]+10)
    max_frames = max_frames if max_frames is not None else deepspeech_tensor.shape[0]
    max_frames_to_read = min(max_frames, deepspeech_tensor.shape[0]) + 50 # 读取视频帧的最大数量


    video_frames = read_video_np(video_path, start_time_sec, max_frames_to_read)
    print("video_frames length: ", len(video_frames))
    video_frames = np.array(video_frames, dtype=np.float32)
    video_frames = video_frames[..., ::-1]
    
    len_out = min(len(video_frames), deepspeech_tensor.shape[0]) // B * B
    video_frames = video_frames[:len_out]
    deepspeech_tensor = deepspeech_tensor[:len_out].to(device)
    
    net_g = LawDNet(opt.source_channel, opt.ref_channel, opt.audio_channel, 
                    opt.warp_layer_num, opt.num_kpoints, opt.coarse_grid_size).to(device)
    print(f"Loading LawDNet model from: {lawdnet_model_path}")

    start_time = time.time()
    checkpoint = torch.load(lawdnet_model_path)
    end_time = time.time()
    print(f"Lawdnet Model loaded in {end_time - start_time:.6f} seconds")

    state_dict = checkpoint['state_dict']['net_g']
    new_state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())
    net_g.load_state_dict(new_state_dict)
    net_g.eval()
    
    facealigner = FaceAlign(ratio=1.6, device=device)
    sqmasker = SmoothSqMask(device=device).to(device)
    
    torchlm.runtime.bind(faceboxesv2(device=device))
    torchlm.runtime.bind(pipnet(backbone="resnet18", pretrained=True, num_nb=10, num_lms=68, net_stride=32,
                                input_size=256, meanface_type="300w", map_location=device.__str__()))
    
    landmarks, _ = torchlm.runtime.forward(video_frames[0])
    landmarks_list = []

    for frame in tqdm(video_frames, desc='Processing video frames, extracting landmarks...'):
        landmarks, bboxes = torchlm.runtime.forward(frame)
        highest_trust_index = np.argmax(bboxes[:, 4])
        most_trusted_landmarks = landmarks[highest_trust_index]
        landmarks_list.append(most_trusted_landmarks)

    landmarks_list = np.array(landmarks_list)
    
    reference_index = torch.randint(0, len(video_frames), (5,)).tolist()
    reference_tensor = torch.tensor(video_frames[reference_index], dtype=torch.float).to(device)
    reference_tensor = reference_tensor.permute(0, 3, 1, 2)
    reference_landmarks = torch.tensor(landmarks_list[reference_index], dtype=torch.float).to(device)
    reference_tensor, _, _ = facealigner(reference_tensor, reference_landmarks, out_W=out_W)
    reference_tensor = reference_tensor / 255.0
    
    def reference(model, masked_source, reference, audio_tensor):
        with torch.no_grad():
            return model(masked_source, reference, audio_tensor)

    outframes = np.zeros_like(video_frames)
    for i in tqdm(range(len_out // B), desc="Processing batches"):
        source_tensor = torch.tensor(video_frames[i * B:(i + 1) * B].copy(), dtype=torch.float).to(device).permute(0, 3, 1, 2)
        landmarks_tensor = torch.tensor(landmarks_list[i * B:(i + 1) * B], dtype=torch.float).to(device)
        feed_tensor, _, affine_matrix = facealigner(source_tensor, landmarks_tensor, out_W=out_W)
        feed_tensor_masked = sqmasker(feed_tensor / 255.0)
        reference_tensor_B = reference_tensor.unsqueeze(0).expand(B, -1, -1, -1, -1).reshape(B, 5 * 3, feed_tensor.shape[2], feed_tensor.shape[3])
        audio_tensor = deepspeech_tensor[i * B:(i + 1) * B].to(device)
        start_time = time.time()
        output_B = reference(net_g, feed_tensor_masked, reference_tensor_B, audio_tensor).float().clamp_(0, 1)
        end_time = time.time()
        print(f"lawdnet 模型纯推理时间 {i} Reference tensor expanded in {end_time - start_time:.6f} seconds")

        outframes_B = facealigner.recover(output_B * 255.0, source_tensor, affine_matrix).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        outframes[i * B:(i + 1) * B] = outframes_B

    outframes = outframes.astype(np.uint8)
    
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    # def save_video_with_audio(outframes, output_video_path, audio_path, output_dir, fps=25):

    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)

    #     processed_clip = ImageSequenceClip([frame for frame in outframes], fps=fps)

    #     audio_clip = AudioFileClip(audio_path)

    #     final_clip = processed_clip.set_audio(audio_clip)
    #     # import pdb; pdb.set_trace()

    #     final_clip.write_videofile(output_video_path, fps=fps, codec='libx264')

    #     return output_video_path

    def save_video_with_audio(outframes, output_video_path, audio_path, output_dir, fps=25):
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 设置临时文件路径
        temp_video_path = os.path.join(output_dir, "temp_video.mp4")
        temp_frames_path = os.path.join(output_dir, "frames")

        # 确保临时帧目录存在
        if not os.path.exists(temp_frames_path):
            os.makedirs(temp_frames_path)

        # 保存帧为图像文件
        for i, frame in enumerate(outframes):
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 修改这里，确保颜色是BGR
            frame_filename = os.path.join(temp_frames_path, f"frame_{i:05d}.png")
            cv2.imwrite(frame_filename, frame_bgr)

        # 使用 ffmpeg 将图像序列转换为视频
        command = [
            "ffmpeg",
            "-y",  # 覆盖输出文件
            "-framerate", str(fps),  # 设置帧率
            "-i", os.path.join(temp_frames_path, "frame_%05d.png"),  # 输入图像序列
            "-c:v", "libx264",  # 编码器
            "-pix_fmt", "yuv420p",  # 像素格式
            temp_video_path  # 输出视频文件
        ]
        subprocess.run(command, check=True)

        # 使用 ffmpeg 将音频和视频合成
        command = [
            "ffmpeg",
            "-y",  # 覆盖输出文件
            "-i", temp_video_path,  # 输入视频文件
            "-i", audio_path,  # 输入音频文件
            "-c:v", "copy",  # 视频流直接复制
            "-c:a", "aac",  # 音频编码器
            "-strict", "experimental",  # 使用实验性编码器
            output_video_path  # 输出视频文件
        ]
        subprocess.run(command, check=True)

        # 删除临时文件
        os.remove(temp_video_path)
        for filename in os.listdir(temp_frames_path):
            file_path = os.path.join(temp_frames_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(temp_frames_path)

        return output_video_path
    
    video_basename_without_ext = os.path.splitext(os.path.basename(video_path))[0]
    output_video_path = os.path.join(output_dir, f"{timestamp_str}_{video_basename_without_ext}_{output_name}.mp4")
    return save_video_with_audio(outframes, output_video_path, audio_path, output_dir, fps=25)


if __name__ == "__main__":
    video_path = '/pfs/mt-1oY5F7/luoyihao/project/DJL/LawDNet_2024/Exp-of-Junli/template/tedliu_远景_25fps.mp4'
    # video_path = '/pfs/mt-1oY5F7/luoyihao/project/DJL/LawDNet_2024/asserts/training_data_HDTF_25fps_2/split_video_25fps/RD_Radio1_000_gfzcyh.mp4'
    # audio_path = '/pfs/mt-1oY5F7/luoyihao/project/DJL/LawDNet_2024/asserts/training_data_HDTF_25fps_2/split_video_25fps_audio/RD_Radio1_000_gfzcyh.wav'
    # video_path = '/pfs/mt-1oY5F7/luoyihao/project/DJL/LawDNet_2024/asserts/training_data/split_video_25fps/3坐_1_25fps_ntpx5u.mp4'
    # video_path = './template/109刘锎宇一棵开花的树25fps_wz94b3.mp4'
    # video_path = '/pfs/mt-1oY5F7/luoyihao/project/DJL/LawDNet_2024/asserts/training_data/split_video_25fps/人物52_25fps_nmlcsh.mp4'

    audio_path = "./template/青岛3.wav" #'./test_dp2_audio/taylor-20s.wav'
    output_dir = './output_video'
    # 设置模型文件路径
    # deepspeech_model_path = "../asserts/output_graph.pb"
    # lawdnet_model_path = "/pfs/mt-1oY5F7/luoyihao/project/DJL/LawDNet_2024/output/training_model_weight/288-mouth-CrossAttention-HDTF-jinpeng/clip_training_256/checkpoint_epoch_170.pth"
    lawdnet_model_path = "/pfs/mt-1oY5F7/luoyihao/project/DJL/LawDNet_2024/output/training_model_weight/288-mouth-CrossAttention-HDTF-jinpeng-dp2-删除静音-测试/clip_training_256/checkpoint_epoch_170.pth"
    # lawdnet_model_path = "../output/training_model_weight/288-mouth-CrossAttention-HDTF-bilibili-xhs/clip_training_256-256无效/checkpoint_epoch_170.pth"
    # lawdnet_model_path = "../template/pretrain_model.pth"
    BatchSize = 60
    mouthsize = '288'
    gpu_index = 2
    output_name = '288-mouth-CrossAttention-HDTF-jinpeng-dp2_测试'
    dp2_path = './dp2_models/LibriSpeech_Pretrained_v3.ckpt'
    
    start_time_sec = 0 # 原视频的第几秒开始
    max_frames = None

    start_time = time.time()

    result_video_path = generate_video_with_audio(video_path, 
                                                  audio_path,
                                                #   deepspeech_model_path, 
                                                  lawdnet_model_path,
                                                  output_dir,
                                                  BatchSize,
                                                  mouthsize,
                                                  gpu_index,
                                                  output_name,
                                                  start_time_sec,
                                                  max_frames,
                                                  dp2_path
                                                  )
    print(f"Generated video with audio at: {result_video_path}")

    end_time = time.time()
    print(f"Total running time: {end_time - start_time} seconds")
