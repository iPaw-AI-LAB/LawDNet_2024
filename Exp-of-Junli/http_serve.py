import sys
sys.path.append('../')

import os
import time
import tempfile

from collections import OrderedDict
import cv2
import numpy as np
import torchlm
from torchlm.tools import faceboxesv2
from torchlm.models import pipnet
from tqdm import tqdm
import torch
from flask import Flask, request, send_file
from moviepy.editor import AudioFileClip, VideoClip

from config.config import DINetTrainingOptions
from deepspeech_pytorch.loader.data_loader import load_audio
from deepspeech_pytorch.utils import load_model
from models.LawDNet import LawDNet
from extract_deepspeech_pytorch2 import check_and_resample_audio, transcribe_audio_data
from tensor_processing import FaceAlign, SmoothSqMask


# TODO: configurize
gpu_index = 0
lawdnet_model_path = "./pretrain_model/checkpoint_epoch_170.pth"
dp2_path = './pretrain_model/LibriSpeech_Pretrained_v3.ckpt'
video_path = './data/figure_five_two.mp4' #'./template/douyin绿幕数字人女.mp4'
batch_size = 1


class VideoTemplate:
    def __init__(self, path: str, target_fps: int, max_frames: int, device: str):
        cap = cv2.VideoCapture(path)
        frame_interval = int(1000/target_fps)
        frames = []
        for i in range(max_frames):
            cap.set(cv2.CAP_PROP_POS_MSEC, i * frame_interval)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        self.frames = frames
        self.device = device

    def preprocess(self, out_W: int):
        len_out = len(self.frames) // batch_size * batch_size
        frames = np.array(self.frames[:len_out], dtype=np.float32)
        frames = frames[..., ::-1]

        # 从视频帧转换成 landmarks
        torchlm.runtime.bind(faceboxesv2(device=self.device))
        torchlm.runtime.bind(pipnet(backbone="resnet18", pretrained=True, num_nb=10, num_lms=68, net_stride=32,
                                    input_size=256, meanface_type="300w", map_location=self.device))
        landmarks_list = []
        for frame in tqdm(frames, desc='Processing video frames, extracting landmarks...'):
            landmarks, bboxes = torchlm.runtime.forward(frame)
            highest_trust_index = np.argmax(bboxes[:, 4])
            most_trusted_landmarks = landmarks[highest_trust_index]
            landmarks_list.append(most_trusted_landmarks)
        landmarks_list = np.array(landmarks_list)

        facealigner = FaceAlign(ratio=1.6, device=self.device)
        sqmasker = SmoothSqMask(device=self.device).to(self.device)
        
        # 预处理参考帧
        reference_index = torch.randint(0, len(frames), (5,)).tolist()
        reference_tensor = torch.tensor(frames[reference_index], dtype=torch.float).to(self.device).permute(0, 3, 1, 2)
        reference_landmarks = torch.tensor(landmarks_list[reference_index], dtype=torch.float).to(self.device)
        reference_tensor, _, _ = facealigner(reference_tensor, reference_landmarks, out_W=out_W)
        self.reference_tensor = reference_tensor / 255.0

        # 创建视频帧的副本以确保连续的内存布局
        frames = np.ascontiguousarray(frames[:len_out])

        # 预处理所有帧
        self.all_source_tensor = torch.tensor(frames, dtype=torch.float).to(self.device).permute(0, 3, 1, 2)
        all_landmarks_tensor = torch.tensor(landmarks_list[:len_out], dtype=torch.float).to(self.device)
        all_feed_tensor, _, self.all_affine_matrix = facealigner(self.all_source_tensor, all_landmarks_tensor, out_W=out_W)
        self.all_feed_tensor_masked = sqmasker(all_feed_tensor / 255.0)

class InferenceServer:
    def __init__(self, opt: DINetTrainingOptions):
        self.device = f'cuda:{gpu_index}' if torch.cuda.is_available() and gpu_index >= 0 else 'cpu'
        self.lawdnet = self.load_lawdnet(lawdnet_model_path, opt)
        self.video_template = VideoTemplate(video_path, target_fps=25, max_frames=5000, device=self.device)
        self.video_template.preprocess(opt.mouth_region_size*1.25)
        self.deepspeech = self.load_deepspeech(dp2_path)
        self.facealigner = FaceAlign(ratio=1.6, device=self.device)
    
    def load_lawdnet(self, lawdnet_model_path: str, opt: DINetTrainingOptions) -> LawDNet:
        net_g = LawDNet(opt.source_channel, opt.ref_channel, opt.audio_channel, 
                    opt.warp_layer_num, opt.num_kpoints, opt.coarse_grid_size).to(self.device)
        checkpoint = torch.load(lawdnet_model_path, map_location=self.device)
        state_dict = checkpoint['state_dict']['net_g']
        new_state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())
        net_g.load_state_dict(new_state_dict)
        net_g.eval()
        return net_g

    def load_deepspeech(self, deepspeech_model_path: str):
        return load_model(device=self.device, model_path=deepspeech_model_path)

    def generate_video_with_audio(self, deepspeech_tensor: torch.Tensor) -> np.ndarray:
        B = batch_size

        len_out = min(len(self.video_template.frames), deepspeech_tensor.shape[0]) // B * B

        # 预处理参考张量
        reference_tensor_expanded = self.video_template.reference_tensor.unsqueeze(0).expand(B, -1, -1, -1, -1).reshape(B, 5 * 3, self.video_template.all_feed_tensor_masked.shape[2], self.video_template.all_feed_tensor_masked.shape[3])

        # 主循环部分
        outframes = np.zeros_like(self.video_template.frames[:len_out])
        main_loop_start = time.time()
        with torch.no_grad():
            for i in tqdm(range(len_out // B), desc="Processing batches"):
                start_idx = i * B
                end_idx = (i + 1) * B
                
                feed_tensor_masked = self.video_template.all_feed_tensor_masked[start_idx:end_idx]
                audio_tensor = deepspeech_tensor[start_idx:end_idx].to(self.device)
                
                output_B = self.lawdnet(feed_tensor_masked, reference_tensor_expanded, audio_tensor).float().clamp_(0, 1)
                
                outframes_B = self.facealigner.recover(output_B * 255.0, self.video_template.all_source_tensor[start_idx:end_idx], self.video_template.all_affine_matrix[start_idx:end_idx]).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                outframes[start_idx:end_idx] = outframes_B

        outframes = outframes.astype(np.uint8)
        main_loop_end = time.time()
        print(f"主循环耗时: {main_loop_end - main_loop_start:.2f} 秒")

        return outframes


app = Flask(__name__)

@app.route('/generate-video', methods=['POST'])
def generate_video():
    tmp_file_path = tempfile.mktemp(suffix='.wav')
    request_file = request.files['audio']
    request_file.save(tmp_file_path)

    check_and_resample_audio(tmp_file_path)
    pcm_data = load_audio(tmp_file_path)

    # 获取 DeepSpeech 特征
    deepspeech_tensor, _ = transcribe_audio_data(pcm_data, inference_server.deepspeech, device=inference_server.device)
    video_frames = inference_server.generate_video_with_audio(deepspeech_tensor)

    # 保存视频文件
    video_file_path = tempfile.mktemp(suffix='.mp4')
    make_frame = lambda t: video_frames[int(t * 25)]
    video_clip = VideoClip(make_frame=make_frame, duration=len(video_frames) / 25)
    audio_clip = AudioFileClip(tmp_file_path)
    video_clip.audio = audio_clip
    video_clip.write_videofile(video_file_path, codec='libx264', fps=25, audio=True, audio_codec='aac')

    # 删除临时文件
    os.remove(tmp_file_path)

    return send_file(video_file_path, mimetype='video/mp4')

if __name__ == '__main__':
    args = ['--opt.mouth_region_size','288']
    opt = DINetTrainingOptions().parse_args(args)
    inference_server = InferenceServer(opt)

    app.run(host='0.0.0.0', port=5051)