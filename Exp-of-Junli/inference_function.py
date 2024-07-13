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
from tqdm import tqdm
from config.config import DINetTrainingOptions
from torchlm.tools import faceboxesv2
from torchlm.models import pipnet
from tensor_processing import SmoothSqMask, FaceAlign
from audio_processing import extract_deepspeech
from moviepy.editor import VideoFileClip, ImageSequenceClip, AudioFileClip

# 项目特定的模块
from models.LawDNet import LawDNet

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 使CUDA错误更易于追踪
warnings.filterwarnings("ignore")  # 初始化和配置警告过滤，忽略不必要的警告

def generate_video_with_audio(video_path, 
                              audio_path, 
                              deepspeech_model_path='../asserts/output_graph.pb',
                              lawdnet_model_path='../output/training_model_weight/288-mouth-CrossAttention-插值coarse-to-fine-shengshu/clip_training_256/checkpoint_epoch_120.pth',
                              output_dir='./output_video',
                              BatchSize = 20,
                              mouthsize = '288' ,
                              gpu_index = 0
                              ):
    
    args = ['--opt.mouth_region_size', mouthsize]
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
    B = BatchSize
    output_name = '5kp-60standard—epoch120-720P-复现'
    
    # 如果是从音频文件提取DeepSpeech特征
    print("Extracting DeepSpeech features from audio file...")
    start_time = time.time()
    deepspeech_tensor, _ = extract_deepspeech(audio_path, deepspeech_model_path)
    end_time = time.time()
    print(f"Running time: {end_time - start_time} seconds for extracting DeepSpeech features.")
    torch.save(deepspeech_tensor, './template/template_audio_deepspeech.pt')

    # import pdb; pdb.set_trace()
    
    def read_video_np(video_path, max_frames=None):
        assert os.path.exists(video_path), f"Video file not found: {video_path}"
        cap = cv2.VideoCapture(video_path)
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
    
    video_frames = read_video_np(video_path, max_frames=deepspeech_tensor.shape[0]+10)
    video_frames = np.array(video_frames, dtype=np.float32)
    video_frames = video_frames[..., ::-1]
    
    len_out = min(len(video_frames), deepspeech_tensor.shape[0]) // B * B
    video_frames = video_frames[:len_out]
    deepspeech_tensor = deepspeech_tensor[:len_out].to(device)
    
    net_g = LawDNet(opt.source_channel, opt.ref_channel, opt.audio_channel, 
                    opt.warp_layer_num, opt.num_kpoints, opt.coarse_grid_size).to(device)
    checkpoint = torch.load(lawdnet_model_path)
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
        output_B = reference(net_g, feed_tensor_masked, reference_tensor_B, audio_tensor).float().clamp_(0, 1)
        outframes_B = facealigner.recover(output_B * 255.0, source_tensor, affine_matrix).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        outframes[i * B:(i + 1) * B] = outframes_B

    outframes = outframes.astype(np.uint8)
    
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    def save_video_with_audio(outframes, output_video_path, audio_path, output_dir, fps=25):

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        processed_clip = ImageSequenceClip([frame for frame in outframes], fps=fps)

        audio_clip = AudioFileClip(audio_path)

        final_clip = processed_clip.set_audio(audio_clip)

        final_clip.write_videofile(output_video_path, codec='libx264')

        return output_video_path
    
    video_basename_without_ext = os.path.splitext(os.path.basename(video_path))[0]
    output_video_path = os.path.join(output_dir, f"{timestamp_str}_{video_basename_without_ext}_{output_name}.mp4")
    return save_video_with_audio(outframes, output_video_path, audio_path, output_dir)


if __name__ == "__main__":
    video_path = './template/test.mp4'
    audio_path = './template/丽娟质检中传播音作品片段展示含中英新闻播报模拟主持25fps_prjf.wav'
    output_dir = './output_video'
    # 设置模型文件路径
    deepspeech_model_path = "../template/output_graph.pb"
    # lawdnet_model_path =  "/home/dengjunli/data/dengjunli/autodl拿过来的/DINet-update/output/training_model_weight/288-mouth-CrossAttention-插值coarse-to-fine-2/clip_training_256/checkpoint_epoch_120.pth"
    lawdnet_model_path = "../template/pretrain_model.pth"
    BatchSize = 20
    mouthsize = '288'
    gpu_index = 1
    result_video_path = generate_video_with_audio(video_path, 
                                                  audio_path,
                                                  deepspeech_model_path, 
                                                  lawdnet_model_path,
                                                  output_dir,
                                                  BatchSize,
                                                  mouthsize,
                                                  gpu_index,
                                                  )
    print(f"Generated video with audio at: {result_video_path}")
