import torch
import numpy as np
import json
import random
import cv2
import os
import sys
from torch.utils.data import Dataset
from models.Gaussian_blur import Gaussian_bluring
from utils.data_processing import load_landmark_openface_origin
import torch.nn.functional as F
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tensor_processing import SmoothMask


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
        self.mouth_region_size = mouth_region_size
        self.radius = mouth_region_size//2
        self.radius_1_4 = self.radius//4
        self.img_h = self.radius * 3 + self.radius_1_4
        self.img_w = self.radius * 2 + self.radius_1_4 * 2
        self.length = len(self.data_dic_name_list)
        # self.top_left_dict = np.load('./asserts/training_data/left_top_dir.npy',allow_pickle=True).item()
        # self.landmark_crop_dic = np.load('./asserts/training_data/landmark_crop_dic.npy',allow_pickle=True).item()
        # self.heatmap_layer = Gaussian_bluring(radius=60,sigma=20,padding='same')######## 做高斯模糊
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.smoothmask = SmoothMask()

    def __getitem__(self, index):
        flag = torch.ones(1,device='cuda')
        video_name = self.data_dic_name_list[index]
        video_clip_num = len(self.data_dic[video_name]['clip_data_list'])

        ######################################### 所有检查视频 脏数据/空数据的操作 ###################################
        if video_clip_num < 6 :
            print("有问题的{},为{}:".format(video_name,video_clip_num))
            source_image_data,source_image_mask, reference_clip_data,deepspeech_feature,flag = self.random_sample()
            return source_image_data,source_image_mask, reference_clip_data,deepspeech_feature,flag  
        ######################################### 所有检查视频 脏数据/空数据的操作 ###################################    

        random_anchor = random.sample(range(video_clip_num), 6)
        source_anchor, reference_anchor_list = random_anchor[0],random_anchor[1:]
        ## load source image
        source_image_path_list = self.data_dic[video_name]['clip_data_list'][source_anchor]['frame_path_list']
        source_random_index = random.sample(range(2, 7), 1)[0]

        ######################################### 所有检查视频 脏数据/空数据的操作 ###################################
        if len(source_image_path_list) < 5:
            print("有问题的{},reference_frame_path_list为{}:".format(video_name,len(source_image_path_list)))
            source_image_data,source_image_mask, reference_clip_data,deepspeech_feature,flag = self.random_sample()
            return source_image_data,source_image_mask, reference_clip_data,deepspeech_feature,flag     
          
        if not os.path.exists(source_image_path_list[source_random_index]):
            print("数据不存在；",source_image_path_list[source_random_index])
            source_image_data,source_image_mask, reference_clip_data,deepspeech_feature,flag = self.random_sample()
            return source_image_data,source_image_mask, reference_clip_data,deepspeech_feature,flag
        ######################################### 所有检查视频 脏数据/空数据的操作 ###################################

        source_image_data = cv2.imread(source_image_path_list[source_random_index])[:, :, ::-1]/ 255.0
        if source_image_data.shape != (self.img_h, self.img_w, 3):
            source_image_data = cv2.resize(source_image_data, (self.img_w, self.img_h))

        source_image_mask = source_image_data.copy() # origin  #origin original 采用smoothmask时注释掉

        ## load deep speech feature
        deepspeech_feature = np.array(self.data_dic[video_name]['clip_data_list'][source_anchor]['deep_speech_list'][source_random_index - 2:source_random_index + 3])

        ## load reference images
        reference_frame_data_list = []
        for reference_anchor in reference_anchor_list:
            reference_frame_path_list = self.data_dic[video_name]['clip_data_list'][reference_anchor]['frame_path_list']

        ######################################### 所有检查视频 脏数据/空数据的操作 ###################################           
            if len(reference_frame_path_list) < 9 :
                print("有问题的{},reference_frame_path_list为{}:".format(video_name,len(reference_frame_path_list)))
                source_image_data,source_image_mask, reference_clip_data,deepspeech_feature,flag = self.random_sample()
                return source_image_data,source_image_mask, reference_clip_data,deepspeech_feature,flag  
        ######################################### 所有检查视频 脏数据/空数据的操作 ###################################

            reference_random_index = random.sample(range(9), 1)[0]
            reference_frame_path = reference_frame_path_list[reference_random_index]
            reference_frame_data = cv2.imread(reference_frame_path)[:, :, ::-1]/255.0
            if reference_frame_data.shape != (self.img_h, self.img_w, 3):
                reference_frame_data = cv2.resize(reference_frame_data, (self.img_w, self.img_h))
            reference_frame_data_list.append(reference_frame_data)
##################################################### 检查reference_frame_data_list ########################################
        
        if all(np.array_equal(x.shape, reference_frame_data_list[0].shape) for x in reference_frame_data_list):
            pass
        else:
            print("有问题的video_name:",video_name)
            print("reference各个图片imagesize不相同")
            
            source_image_data,source_image_mask, reference_clip_data,deepspeech_feature,flag = self.random_sample()
            return source_image_data,source_image_mask, reference_clip_data,deepspeech_feature,flag

##################################################### 检查reference_frame_data_list ########################################

        reference_clip_data = np.concatenate(reference_frame_data_list, 2)

        # to tensor
        source_image_data = torch.from_numpy(source_image_data).float().permute(2,0,1).cuda()
        source_image_mask = torch.from_numpy(source_image_mask).float().permute(2,0,1).cuda()
        reference_clip_data = torch.from_numpy(reference_clip_data).float().permute(2,0,1).cuda()

######################################### 所有检查 音频 脏数据/空数据的操作 ###################################  
        if deepspeech_feature.shape != (5,29):
            print("音频数据有问题；",source_image_path_list[source_random_index])
            print("deepspeech_feature.shape:",deepspeech_feature.shape)
            source_image_data,source_image_mask, reference_clip_data,deepspeech_feature,flag = self.random_sample()
            return source_image_data,source_image_mask, reference_clip_data,deepspeech_feature,flag
######################################### 所有检查 音频 脏数据/空数据的操作 ###################################  
        deepspeech_feature = torch.from_numpy(deepspeech_feature).float().permute(1,0).cuda()



        return source_image_data,source_image_mask, reference_clip_data,deepspeech_feature,flag

    def __len__(self):
        return self.length
    
    def random_sample(self):
        '''
        当数据有误时返回随机数,flag=0
        '''       
        source_clip_shape = torch.Size([3, self.img_h, self.img_w])
        source_image_data = torch.randn(source_clip_shape,device=self.device)

        source_clip_mask_shape = torch.Size([3, self.img_h, self.img_w])
        source_image_mask = torch.randn(source_clip_mask_shape,device=self.device)

        reference_clip_shape = torch.Size([15, self.img_h, self.img_w])
        reference_clip = torch.randn(reference_clip_shape,device=self.device)

        deep_speech_full_shape = torch.Size([29, 5])
        deep_speech_full = torch.randn(deep_speech_full_shape,device=self.device)

        return source_image_data,source_image_mask, reference_clip, deep_speech_full , torch.zeros(1,device=self.device)