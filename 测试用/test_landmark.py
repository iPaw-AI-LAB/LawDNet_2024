from moviepy.editor import AudioFileClip, VideoFileClip
import cv2

#!/usr/bin/python
#-*- coding: utf-8 -*-
# Video 25 FPS, Audio 16000HZ

import torch
import numpy
import time, pdb, argparse, subprocess, os, math, glob
import cv2
import python_speech_features

from scipy import signal
from scipy.io import wavfile
# from SyncNetModel import *
from shutil import rmtree

from moviepy.editor import AudioFileClip, VideoFileClip
import logging

import torchlm
from torchlm.tools import faceboxesv2
from torchlm.models import pipnet
import numpy as np
from tqdm import tqdm


import tensorflow as tf


import torch

print(torch.version.cuda)  # 打印CUDA版本 11.7
print(torch.backends.cudnn.version())  # 打印cuDNN版本 8902


# 检查 GPU 是否可用
if tf.config.list_physical_devices('GPU'):
    # print("tensorflow.compat.v1.config:",tensorflow.compat.v1.config)
    print('GPU 可用')
else:
    print('GPU 不可用')




# logging.basicConfig(filename='./deng_align_videos.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
# # ==================== dengjunli ====================

# def landmark_csv(video_path, num_frames):
#     ############  
#     cap = cv2.VideoCapture(video_path)

#     # 初始化结果数组
#     # num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 用前面拿到的num_frames
#     # print("num_frames:",num_frames)
#     video_landmark_data = np.zeros((num_frames, 68, 2))

#     # 加载模型
#     torchlm.runtime.bind(faceboxesv2(device="cuda"))  # set device="cuda" if you want to run with CUDA
#     # set map_location="cuda" if you want to run with CUDA
#     torchlm.runtime.bind(
#     pipnet(backbone="resnet101", pretrained=True,
#             num_nb=10, num_lms=68, net_stride=32, input_size=256,
#             meanface_type="300w", map_location="cuda", checkpoint=None)
#     ) # will auto download pretrained weights from latest release if pretrained=True

#     # 逐帧处理视频
#     frame_index = 0
#     # 初始化结果数组
#     # num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     # 显示进度条
#     for frame_index in tqdm(range(num_frames)):
#         # print(frame_index,"帧")
#         ret, frame = cap.read()
#         if not ret:
#             import pdb
#             pdb.set_trace()
#             break

#         # 进行人脸关键点检测
#         landmarks, bboxes = torchlm.runtime.forward(frame)

#         # 将关键点结果保存到数组中
#         for i in range(len(bboxes)):
#             video_landmark_data[frame_index] = landmarks[i]

#         frame_index += 1

#     # 释放资源
#     cap.release()

#     # 创建一个形状为 (num_frames, 138) 的 numpy 数组 tensor
#     one_line = np.zeros((num_frames, 141))

#     # 遍历每一帧数据
#     for i in range(num_frames):
#         # 创建一个新的列表 row，并将当前帧的数据添加到其中
#         row = [i+1, 0, i*0.04, 0.98, 1] + video_landmark_data[i].flatten('F').tolist()
#         # 将 row 中的元素依次添加到 tensor 中
#         one_line[i] = np.array(row)

#     # 将 tensor 保存为 csv 文件
#     np.savetxt(video_path.replace('.mp4','.csv'), one_line, delimiter=',', 
#             header='frame,face_id,timestamp,confidence,success,'+
#             ','.join([f'x_{i}' for i in range(68)])+','+','.join([f'y_{i}' for i in range(68)]), comments='',fmt='%.1f')

# path = ('./RD_Radio52_000_corrected.mp4')

# video = VideoFileClip(path)

# # 计算视频的总帧数
# num_frames = int(video.duration * video.fps)

# # 输出总帧数
# print("视频的总帧数为:", num_frames)

# videoCapture = cv2.VideoCapture(path)
# frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)

# print("opencv视频总帧数:",frames)

# landmark_csv(path,num_frames)


