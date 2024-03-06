import torch
import numpy as np
import json
import random
import cv2
import os
import sys
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from models.Gaussian_blur import Gaussian_bluring
from utils.data_processing import load_landmark_openface_origin
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tensor_processing import SmoothMask


import cv2
import os
import random
from pathlib import Path
import numpy as np



def load_reference_frames(folder_path, img_h=416, img_w=320):
    """
    在给定的文件夹路径中随机选取以'p', 's', 'e', 'f', 'w'开头的jpg文件各一张，
    读取它们为参考帧，并返回处理后的参考帧列表。
    
    :param folder_path: 文件夹路径，其中包含以特定字符开头的jpg图像文件。
    :param img_h: 目标图像高度
    :param img_w: 目标图像宽度
    :return: 包含处理后的参考帧数据的列表。
    """
    reference_frame_list = []
    target_chars = ['p', 's', 'e', 'f', 'w']
    
    for char in target_chars:
        # 在文件夹中找到所有以当前字符开头的jpg文件
        char_files = list(Path(folder_path).glob(f'{char}*.jpg'))
        if char_files:
            # 从符合条件的文件中随机选择一个
            selected_file = random.choice(char_files)
            # 读取并处理图像
            reference_frame_data = cv2.imread(str(selected_file))[:, :, ::-1] / 255.0
            if reference_frame_data.shape != (img_h, img_w, 3):
                reference_frame_data = cv2.resize(reference_frame_data, (img_w, img_h))
            reference_frame_list.append(reference_frame_data)
    
    return reference_frame_list


def get_data(json_name,augment_num):
    print('start loading data')
    with open(json_name,'r') as f:
        data_dic = json.load(f)
    data_dic_name_list = []
    for augment_index in range(augment_num):
        for video_name in data_dic.keys():
            data_dic_name_list.append(video_name)
    random.shuffle(data_dic_name_list)
    print('finish loading')
    return data_dic_name_list,data_dic


class DINetDataset(Dataset):
    def __init__(self,path_json,augment_num,mouth_region_size):
        super(DINetDataset, self).__init__()
        self.data_dic_name_list,self.data_dic = get_data(path_json,augment_num)
        # self.top_left_dict = np.load('./asserts/training_data/left_top_dir.npy',allow_pickle=True).item()
        # self.landmark_crop_dic = np.load('./asserts/training_data/landmark_crop_dic.npy',allow_pickle=True).item()
        self.mouth_region_size = mouth_region_size
        self.radius = mouth_region_size//2
        self.radius_1_4 = self.radius//4
        self.img_h = self.radius * 3 + self.radius_1_4
        self.img_w = self.radius * 2 + self.radius_1_4 * 2
        self.length = len(self.data_dic_name_list)
        self.smoothmask = SmoothMask()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __getitem__(self, index):
        flag = torch.ones(1,device='cuda')
        video_name = self.data_dic_name_list[index]
        video_clip_num = len(self.data_dic[video_name]['clip_data_list'])

        ######################################### 所有检查视频 脏数据/空数据的操作 ###################################
        if video_clip_num < 6 :
            print("有问题的{},为{}:".format(video_name,video_clip_num))
            source_clip,source_clip_mask, reference_clip,deep_speech_clip,deep_speech_full,flag = self.random_sample_with_batch()
            return source_clip,source_clip_mask, reference_clip,deep_speech_clip,deep_speech_full , flag
        ######################################### 所有检查视频 脏数据/空数据的操作 ###################################
        
        source_anchor = random.sample(range(video_clip_num), 1)[0]

        source_image_path_list = self.data_dic[video_name]['clip_data_list'][source_anchor]['frame_path_list']
        reference_for_dV_path_list = self.data_dic[video_name]['clip_data_list'][source_anchor-1]['frame_path_list']

        source_clip_list = []
        source_clip_mask_list = []
        deep_speech_list = []
        reference_clip_list = []
        reference_for_dV = []

        # 获取参考帧，也放进Dv里训练
        for reference_for_dV_frame_index in range(2, 2 + 5):
######################################### 所有检查视频 脏数据/空数据的操作 ###################################
            if len(reference_for_dV_path_list) < 5:
                print("有问题的{},reference_frame_path_list为{}:".format(video_name,len(reference_for_dV_path_list)))
                return self.random_sample_with_batch()

            if not os.path.isfile(reference_for_dV_path_list[reference_for_dV_frame_index]):
                print("脏数据已经删掉：",reference_for_dV_path_list[reference_for_dV_frame_index])
                # flag = torch.zeros(1,device='cuda:0')
                return self.random_sample_with_batch()

            if video_clip_num <= 5 :
                print("有问题的video_name,小于等于5,无法取到参考帧:",video_name)
                # flag = torch.zeros(1,device='cuda:0')
                return self.random_sample_with_batch()
######################################### 所有检查 视频 脏数据/空数据的操作 ###################################    

            reference_for_dV_data = cv2.imread(reference_for_dV_path_list[reference_for_dV_frame_index])[:, :, ::-1]/255.0
            if reference_for_dV_data.shape != (self.img_h, self.img_w, 3):
                reference_for_dV_data = cv2.resize(reference_for_dV_data, (self.img_w, self.img_h))
            reference_for_dV.append(reference_for_dV_data)
        # import pdb;pdb.set_trace()
        reference_for_dV = np.stack(reference_for_dV, 0)

        
        for source_frame_index in range(2, 2 + 5):
######################################### 所有检查视频 脏数据/空数据的操作 ###################################
            if len(source_image_path_list) < 5:
                print("有问题的{},reference_frame_path_list为{}:".format(video_name,len(source_image_path_list)))
                return self.random_sample_with_batch()

            if not os.path.isfile(source_image_path_list[source_frame_index]):
                print("脏数据已经删掉：",source_image_path_list[source_frame_index])
                # flag = torch.zeros(1,device='cuda:0')
                return self.random_sample_with_batch()

            if video_clip_num <= 5 :
                print("有问题的video_name,小于等于5,无法取到参考帧:",video_name)
                # flag = torch.zeros(1,device='cuda:0')
                return self.random_sample_with_batch() 
######################################### 所有检查 视频 脏数据/空数据的操作 ###################################      
            
            source_image_data = cv2.imread(source_image_path_list[source_frame_index])[:, :, ::-1]/255.0
            if source_image_data.shape != (self.img_h, self.img_w, 3):
                source_image_data = cv2.resize(source_image_data, (self.img_w, self.img_h))

            source_clip_list.append(source_image_data)
            source_image_mask = source_image_data.copy()        
            source_clip_mask_list.append(source_image_mask)

            ## load deep speech feature
            deepspeech_array = np.array(self.data_dic[video_name]['clip_data_list'][source_anchor]['deep_speech_list'][
                                       source_frame_index - 2:source_frame_index + 3])
            deep_speech_list.append(deepspeech_array)

            ## ## load reference images
            reference_frame_list = []
            reference_anchor_list = random.sample(range(video_clip_num), 5)
            for reference_anchor in reference_anchor_list:
                reference_frame_path_list = self.data_dic[video_name]['clip_data_list'][reference_anchor][
                    'frame_path_list']
        ######################################### 所有检查视频 脏数据/空数据的操作 ###################################           
                if len(reference_frame_path_list) < 9 :
                    print("有问题的{},reference_frame_path_list为{}:".format(video_name,len(reference_frame_path_list)))
                    return self.random_sample_with_batch()
        ######################################### 所有检查视频 脏数据/空数据的操作 ###################################           
                reference_random_index = random.sample(range(9), 1)[0]
                reference_frame_path = reference_frame_path_list[reference_random_index]
                reference_frame_data = cv2.imread(reference_frame_path)[:, :, ::-1]/ 255.0
                if reference_frame_data.shape != (self.img_h, self.img_w, 3):
                    reference_frame_data = cv2.resize(reference_frame_data, (self.img_w, self.img_h)) 
                reference_frame_list.append(reference_frame_data)

                # import pdb; pdb.set_trace()
                reference_frame_list_two =  load_reference_frames("./"+str(Path(reference_frame_path_list[0]).parents[1]), self.img_h, self.img_w)

###################################### 检查reference_frame_data_list ########################################
            if all(np.array_equal(x.shape, reference_frame_list[0].shape) for x in reference_frame_list):
                pass
            else:
                print("有问题的video_name:",video_name)
                print("reference各个图片imagesize不相同")
                return self.random_sample_with_batch()

##################################################### 检查reference_frame_data_list ########################################

            reference_clip_list.append(np.concatenate(reference_frame_list, 2))
        ##### 获取了五张图片，五张图片的GT，五张图片的音频，五张图片的参考帧

        source_clip = np.stack(source_clip_list, 0)
        source_clip_mask = np.stack(source_clip_mask_list, 0)
        deep_speech_full = np.array(self.data_dic[video_name]['clip_data_list'][source_anchor]['deep_speech_list'])

######################################### 所有检查 音频 脏数据/空数据的操作 ###################################     

        if all(np.array_equal(x.shape, deep_speech_list[0].shape) for x in deep_speech_list):
            pass
        else:
            print("有问题的video_name:",video_name)
            print("deepspeech不相等")
            return self.random_sample_with_batch()
        
        if deep_speech_full.shape != (9,29):
            print("音频数据有问题；",source_image_path_list[source_frame_index])
            print("deepspeech_feature.shape:",deep_speech_full.shape)
            return self.random_sample_with_batch()

######################################### 所有检查 音频 脏数据/空数据的操作 ###################################  

        deep_speech_clip = np.stack(deep_speech_list, 0)
        reference_clip = np.stack(reference_clip_list, 0)

        source_clip = torch.from_numpy(source_clip).float().permute(0, 3, 1, 2).cuda()
        source_clip_mask = torch.from_numpy(source_clip_mask).float().permute(0, 3, 1, 2).cuda()
        reference_clip = torch.from_numpy(reference_clip).float().permute(0, 3, 1, 2).cuda()
        deep_speech_clip = torch.from_numpy(deep_speech_clip).float().permute(0, 2, 1).cuda()
        deep_speech_full = torch.from_numpy(deep_speech_full).permute(1, 0).cuda() 
        reference_for_dV = torch.from_numpy(reference_for_dV).float().permute(0, 3, 1, 2).cuda()

        return source_clip,source_clip_mask, reference_clip,deep_speech_clip,deep_speech_full,reference_for_dV,flag

    def __len__(self):
        return self.length
    
    def random_sample(self):
        '''
        当数据有误时返回随机数,flag=0
        '''
        source_clip_shape = torch.Size([5, 3, self.img_h, self.img_w])
        source_clip = torch.randn(source_clip_shape,device=self.device)

        source_clip_mask_shape = torch.Size([5, 3, self.img_h, self.img_w])
        source_clip_mask = torch.randn(source_clip_mask_shape,device=self.device)

        reference_clip_shape = torch.Size([5, 15, self.img_h, self.img_w])
        reference_clip = torch.randn(reference_clip_shape,device=self.device)

        deep_speech_clip_shape = torch.Size([5, 29, 5])
        deep_speech_clip = torch.randn(deep_speech_clip_shape,device=self.device)

        deep_speech_full_shape = torch.Size([29, 9])
        deep_speech_full = torch.randn(deep_speech_full_shape,device=self.device)

        return source_clip,source_clip_mask, reference_clip,deep_speech_clip,deep_speech_full,torch.zeros(1,device='cuda:0')
    
    def random_sample_with_batch(self):
        '''
        当数据有误时返回随机数,flag=0
        '''
        source_clip_shape = torch.Size([5, 3, self.img_h, self.img_w])
        source_clip = torch.randn(source_clip_shape,device=self.device)

        reference_for_dV_shape = torch.Size([5, 3, self.img_h, self.img_w])
        reference_for_dV = torch.randn(reference_for_dV_shape,device=self.device)

        source_clip_mask_shape = torch.Size([5, 3, self.img_h, self.img_w])
        source_clip_mask = torch.randn(source_clip_mask_shape,device=self.device)

        reference_clip_shape = torch.Size([5, 15, self.img_h, self.img_w])
        reference_clip = torch.randn(reference_clip_shape,device=self.device)

        deep_speech_clip_shape = torch.Size([5, 29, 5])
        deep_speech_clip = torch.randn(deep_speech_clip_shape,device=self.device)

        deep_speech_full_shape = torch.Size([29, 9])
        deep_speech_full = torch.randn(deep_speech_full_shape,device=self.device)

        return source_clip,source_clip_mask, reference_clip,deep_speech_clip,deep_speech_full, reference_for_dV, torch.zeros(1,device=self.device)
        