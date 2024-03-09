import torch
import numpy as np
import json
import random
import cv2
import os
from tqdm import tqdm
from torch.utils.data import Dataset
from models.Gaussian_blur import Gaussian_bluring
from utils.data_processing import load_landmark_openface_origin
from tensor_processing import SmoothMask

def get_data(json_name, augment_num):
    """
    从指定的JSON文件中加载数据，并根据增强次数扩展数据集。
    
    :param json_name: JSON文件名，包含数据集的详细信息。
    :param augment_num: 数据增强的次数。
    :return: 数据集的名称列表和数据字典。
    """
    print('开始加载数据')
    with open(json_name, 'r') as f:
        data_dic = json.load(f)
    data_dic_name_list = []
    for augment_index in tqdm(range(augment_num)):  # Wrapped with tqdm for progress tracking
        for video_name in data_dic.keys():
            data_dic_name_list.append(video_name)
    random.shuffle(data_dic_name_list)
    print('完成加载')
    return data_dic_name_list, data_dic

class DINetDataset(Dataset):
    def __init__(self, path_json, augment_num, mouth_region_size):
        """
        初始化DINet数据集。
        
        :param path_json: 包含训练数据信息的JSON文件路径。
        :param augment_num: 数据增强次数。
        :param mouth_region_size: 嘴部区域的大小。
        """
        self.data_dic_name_list, self.data_dic = get_data(path_json, augment_num)
        self.mouth_region_size = mouth_region_size
        self.radius = mouth_region_size // 2
        self.radius_1_4 = self.radius // 4
        self.img_h = self.radius * 3 + self.radius_1_4
        self.img_w = self.radius * 2 + self.radius_1_4 * 2
        self.length = len(self.data_dic_name_list)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.smoothmask = SmoothMask()

    def __getitem__(self, index):
        """
        根据索引获取数据集中的单个项。
        
        :param index: 数据项的索引。
        :return: 元组(source_image_data, reference_clip_data, deepspeech_feature, flag)
        """
        flag = torch.ones(1, device=self.device)
        video_name = self.data_dic_name_list[index]
        video_clip_num = len(self.data_dic[video_name]['clip_data_list'])

        # 验证视频片段数量是否足够
        if video_clip_num < 6:
            print(f"数据问题：{video_name}, 视频片段数量：{video_clip_num}")
            return self.zeros_sample()

        random_anchor = random.sample(range(video_clip_num), 6)
        source_anchor, reference_anchor_list = random_anchor[0], random_anchor[1:]

        source_image_path_list = self.data_dic[video_name]['clip_data_list'][source_anchor]['frame_path_list']
        source_random_index = random.choice(range(2, 7))

        # 验证源图像路径是否有效
        if len(source_image_path_list) < 5 or not os.path.exists(source_image_path_list[source_random_index]):
            print(f"数据问题：{video_name}, source_image_path_list长度：{len(source_image_path_list)}")
            return self.zeros_sample()

        source_image_data = self.preprocess_image(source_image_path_list[source_random_index])
        # source_image_mask = source_image_data.copy()  # 使用smoothmask进行处理的情况可以在这里添加

        deepspeech_feature = np.array(self.data_dic[video_name]['clip_data_list'][source_anchor]['deep_speech_list'][source_random_index - 2:source_random_index + 3])
        reference_frame_data_list = [
            self.preprocess_image(self.data_dic[video_name]['clip_data_list'][anchor]['frame_path_list'][random.choice(range(9))]) 
            for anchor in reference_anchor_list
        ]

        # 检查参考帧数据有效性
        if not self.check_data_validity(reference_frame_data_list, (self.img_h, self.img_w, 3)):
            print(f"数据问题：{video_name}, 参考帧大小不一致")
            return self.zeros_sample()

        reference_clip_data = np.concatenate(reference_frame_data_list, axis=2)

        # 检查深度语音特征形状
        if not self.check_data_validity(deepspeech_feature, (5, 29)):
            print(f"数据问题：{video_name}, deepspeech_feature形状：{deepspeech_feature.shape}")
            return self.zeros_sample()

        source_image_data_tensor = torch.tensor(source_image_data).float().permute(2, 0, 1).to(self.device)
        # source_image_mask_tensor = torch.tensor(source_image_mask).float().permute(2, 0, 1).to(self.device)
        reference_clip_data_tensor = torch.tensor(reference_clip_data).float().permute(2, 0, 1).to(self.device)
        deepspeech_feature_tensor = torch.tensor(deepspeech_feature).float().permute(1, 0).to(self.device)

        return source_image_data_tensor, reference_clip_data_tensor, deepspeech_feature_tensor, flag


    def __len__(self):
        """
        返回数据集大小。
        """
        return self.length

    def zeros_sample(self):
        """
        当数据有误时，返回随机样本和标志位0。
        """
        source_image_data = torch.zeros((3, self.img_h, self.img_w), device=self.device)
        source_image_mask = torch.zeros((3, self.img_h, self.img_w), device=self.device)
        reference_clip_data = torch.zeros((15, self.img_h, self.img_w), device=self.device)  # 假设有5个参考帧，每帧3个通道
        deep_speech_full = torch.zeros((29, 5), device=self.device)
        return source_image_data, reference_clip_data, deep_speech_full, torch.zeros(1, device=self.device)

    def preprocess_image(self, image_path):
        """
        预处理图像数据。
        
        :param image_path: 图像文件的路径。
        :return: 预处理后的图像数据。
        """
        image_data = cv2.imread(image_path)[:, :, ::-1] / 255.0  # 读取图像并转换为RGB
        if image_data.shape != (self.img_h, self.img_w, 3):
            image_data = cv2.resize(image_data, (self.img_w, self.img_h))  # 调整图像尺寸
        return image_data
    

    def check_data_validity(self, data_list, expected_shape):
        """
        检查数据有效性。
        
        :param data_list: 待检查的数据列表。
        :param expected_shape: 期望的数据形状。
        :return: 数据是否有效的布尔值。
        """
        # 如果送入的是list
        if isinstance(data_list, list):
            for data in data_list:
                if not np.array_equal(data.shape, expected_shape):
                    print("数据形状不符合期望：", data.shape, "期望：", expected_shape)
                    return False  # 数据无效
            return True  # 数据有效
        else:
            # deepspeech_feature_tensor
            if not np.array_equal(data_list.shape, expected_shape):
                print("数据形状不符合期望：", data_list.shape, "期望：", expected_shape)
                return False
            return True
