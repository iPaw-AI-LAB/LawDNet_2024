import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from sync_batchnorm import SynchronizedBatchNorm1d as BatchNorm1d
from torch.cuda.amp import autocast as autocast
import sys
sys.path.append("..")
from torch_affine_ops import standard_grid

# 定义嘴唇图像的CNN模型
# class LipContentModel(nn.Module):
#     def __init__(self):
#         super(LipContentModel, self).__init__()
#         self.conv1 = nn.Conv2d(15, 32, kernel_size=3, stride=1, padding=1)
#         # self.bn1 = BatchNorm2d(32)
#         self.relu1 = nn.LeakyReLU(inplace=True)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         # self.bn2 = BatchNorm2d(64)
#         self.relu2 = nn.LeakyReLU(inplace=True)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc = nn.Linear(64 * 64 * 64, 32)
#         # self.bn_fc = BatchNorm1d(32)  # 添加全连接层后的批归一化层

#     def forward(self, x):
#         with autocast():
#             x = self.conv1(x)
#             # x = self.bn1(x)
#             x = self.relu1(x)
#             x = self.pool1(x)
#             x = self.conv2(x)
#             # x = self.bn2(x)
#             x = self.relu2(x)
#             x = self.pool2(x)
#             x = x.view(x.size(0), -1)
#             x = self.fc(x)
#             # x = self.bn_fc(x)  # 全连接层后的批归一化
#             return x

# 定义嘴唇图像的CNN模型
class LipContentModel(nn.Module):
    def __init__(self):
        super(LipContentModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(15, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 32 * 32, 512),
            nn.ReLU(),
            nn.Linear(512, 29 * 5),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        with autocast():
            batch_size = x.size(0)
            x = self.conv_layers(x)
            x = x.view(batch_size, -1)
            x = self.fc_layers(x)
            x = x.view(batch_size, 29, 5)
            return x


# 定义DeepSpeech音频特征的CNN模型
class AudioContentModel(nn.Module):
    def __init__(self):
        super(AudioContentModel, self).__init__()
        self.conv1 = nn.Conv1d(29, 64, kernel_size=3, stride=1, padding=1)
        # self.bn1 = BatchNorm1d(64)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        # self.bn2 = BatchNorm1d(128)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = nn.Linear(128 * 2, 32)
        # self.bn_fc = BatchNorm1d(32)  # 添加全连接层后的批归一化层

    def forward(self, x):
        with autocast():
            x = self.conv1(x)
            # x = self.bn1(x)
            x = self.relu1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            # x = self.bn2(x)
            x = self.relu2(x)
            x = self.pool2(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            # x = self.bn_fc(x)  # 全连接层后的批归一化
            return x