'''
说明：制作数据集的代码，包括提取视频帧、提取音频、提取deep speech特征、根据openface的landmark裁剪人脸、生成训练json文件
'''
import glob
import os
import subprocess
import cv2
import numpy as np
import json
import shutil
import torch
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import ThreadPool

import sys
sys.path.append('./')
from tensor_processing import *

from utils.data_processing import load_landmark_openface_origin,compute_crop_radius
from utils.deep_speech import DeepSpeech
from config.config import DataProcessingOptions
from tqdm import tqdm


def extract_audio(source_video_dir,res_audio_dir):
    '''
    extract audio files from videos
    '''
    if not os.path.exists(source_video_dir):
        raise ('wrong path of video dir')
    if not os.path.exists(res_audio_dir):
        os.mkdir(res_audio_dir)
    video_path_list = glob.glob(os.path.join(source_video_dir, '*.mp4'))
    for video_path in video_path_list:
        print('extract audio from video: {}'.format(os.path.basename(video_path)))
        audio_path = os.path.join(res_audio_dir, os.path.basename(video_path).replace('.mp4', '.wav'))
        cmd = 'ffmpeg -i {} -f wav -ar 16000 {}'.format(video_path, audio_path)
        subprocess.call(cmd, shell=True)

def extract_deep_speech(audio_dir,res_deep_speech_dir,deep_speech_model_path):
    '''
    extract deep speech feature
    '''
    if not os.path.exists(res_deep_speech_dir):
        os.mkdir(res_deep_speech_dir)
    DSModel = DeepSpeech(deep_speech_model_path)
    wav_path_list = glob.glob(os.path.join(audio_dir, '*.wav'))
    
    for wav_path in wav_path_list:
        print('video_name 正在处理：',wav_path)
        video_name = os.path.basename(wav_path).replace('.wav', '')
        res_dp_path = os.path.join(res_deep_speech_dir, video_name + '_deepspeech.txt')
        if os.path.exists(res_dp_path):
            os.remove(res_dp_path)
        print('extract deep speech feature from audio:{}'.format(video_name))
        ds_feature = DSModel.compute_audio_feature(wav_path)
        np.savetxt(res_dp_path, ds_feature)

def extract_video_frame(source_video_dir,res_video_frame_dir):
    '''
        extract video frames from videos
    '''
    if not os.path.exists(source_video_dir):
        raise ('wrong path of video dir')
    if not os.path.exists(res_video_frame_dir):
        os.mkdir(res_video_frame_dir)
    video_path_list = glob.glob(os.path.join(source_video_dir, '*.mp4'))
    for video_path in video_path_list:
        video_name = os.path.basename(video_path)
        frame_dir = os.path.join(res_video_frame_dir, video_name.replace('.mp4', ''))
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)
        print('extracting frames from {} ...'.format(video_name))
        videoCapture = cv2.VideoCapture(video_path)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        if int(fps) != 25:
            raise ('{} video is not in 25 fps'.format(video_path))
        frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        for i in tqdm(range(int(frames))):
            ret, frame = videoCapture.read()
            result_path = os.path.join(frame_dir, str(i).zfill(6) + '.jpg')
            cv2.imwrite(result_path, frame)


def crop_face_according_openfaceLM(openface_landmark_dir,video_frame_dir,res_crop_face_dir,clip_length):
    '''
      crop face according to openface landmark
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    landmark_crop_dic = {} # 定义一个左上标角的字典
    facealigner = FaceAlign(device=device,ratio=1.6)

    if not os.path.exists(openface_landmark_dir):
        raise ('wrong path of openface landmark dir')
    if not os.path.exists(video_frame_dir):
        raise ('wrong path of video frame dir')
    if not os.path.exists(res_crop_face_dir):
        os.mkdir(res_crop_face_dir)
    landmark_openface_path_list = glob.glob(os.path.join(openface_landmark_dir, '*.csv'))
    for landmark_openface_path in tqdm(landmark_openface_path_list):
        video_name = os.path.basename(landmark_openface_path).replace('.csv', '')
        crop_face_video_dir = os.path.join(res_crop_face_dir, video_name)
        if not os.path.exists(crop_face_video_dir):
            os.makedirs(crop_face_video_dir)
        print('cropping face from video: {} ...'.format(video_name))
        landmark_openface_data = load_landmark_openface_origin(landmark_openface_path).astype(np.int32)
        frame_dir = os.path.join(video_frame_dir, video_name)
        if not os.path.exists(frame_dir):
            raise ('run last step to extract video frame')
        if len(glob.glob(os.path.join(frame_dir, '*.jpg'))) != landmark_openface_data.shape[0]:
            # raise ('landmark length is different from frame length')
            print('landmark length is different from frame length',video_name)
        frame_length = min(len(glob.glob(os.path.join(frame_dir, '*.jpg'))), landmark_openface_data.shape[0])
        end_frame_index = list(range(clip_length, frame_length, clip_length))
        video_clip_num = len(end_frame_index)
        for i in tqdm(range(video_clip_num)):

            res_face_clip_dir = os.path.join(crop_face_video_dir, str(i).zfill(6))
            if not os.path.exists(res_face_clip_dir):
                os.mkdir(res_face_clip_dir)
            images = []
            landmarks = []
            for frame_index in range(end_frame_index[i]- clip_length,end_frame_index[i]):
                source_frame_path = os.path.join(frame_dir,str(frame_index).zfill(6)+'.jpg')
                source_frame_data = cv2.imread(source_frame_path)
                frame_landmark = landmark_openface_data[frame_index, :, :]
                images.append(source_frame_data)
                landmarks.append(frame_landmark)   

            images_np = np.stack(images,axis=0)
            images_torch = torch.from_numpy(images_np.copy()).float().permute(0,3,1,2).to(device)
            landmark_np = np.stack(landmarks,axis=0)
            landmarks_tensor = torch.from_numpy(landmark_np).float().to(device)  

            # 批量正脸化并保存
            # out_W：输出图像的宽度, 高度为宽度的1.3倍
            face_align_img, lmrks_align, _ = facealigner(images_torch, landmarks_tensor, out_W=640) 
            index = 0

            for frame_index_2 in range(end_frame_index[i]- clip_length,end_frame_index[i]):
                face_align_img_np = face_align_img[index].permute(1,2,0).cpu().numpy().astype(np.uint8)
                res_crop_face_frame_path = os.path.join(res_face_clip_dir, str(frame_index_2).zfill(6) + '.jpg')
                if os.path.exists(res_crop_face_frame_path):
                   os.remove(res_crop_face_frame_path)
                cv2.imwrite(res_crop_face_frame_path, face_align_img_np)
                landmark_crop_dic[res_crop_face_frame_path] = lmrks_align[index]   
                index +=1 

    print("save landmark_crop_dic.npy")
    np.save('./asserts/training_data/landmark_crop_dic.npy',landmark_crop_dic)


def generate_training_json(crop_face_dir,deep_speech_dir,clip_length,res_json_path):
    video_name_list = os.listdir(crop_face_dir)
    video_name_list.sort()
    res_data_dic = {}
    for video_index, video_name in enumerate(video_name_list):
        print('generate training json file :{} {}/{}'.format(video_name,video_index,len(video_name_list)))
        tem_dic = {}
        deep_speech_feature_path = os.path.join(deep_speech_dir, video_name + '_deepspeech.txt')
        if not os.path.exists(deep_speech_feature_path):
            # raise ('wrong path of deep speech')
            print('wrong path of deep speech',"delete video_name:",video_name)
            
            file_path = os.path.join(crop_face_dir, video_name)
            if os.path.exists(file_path):
                shutil.rmtree(file_path)
            file_path = os.path.join(opt.openface_landmark_dir, video_name+'.csv')
            if os.path.exists(file_path):    
                os.remove(file_path)
            file_path = os.path.join(opt.video_frame_dir, video_name)
            if os.path.exists(file_path):
                shutil.rmtree(file_path)
            file_path = os.path.join(opt.source_video_dir, video_name+'.mp4')
            if os.path.exists(file_path):  
                os.remove(file_path)
            # if the file doesn't exist, skip this iteration of the loop
            continue
            
        deep_speech_feature = np.loadtxt(deep_speech_feature_path)
        video_clip_dir = os.path.join(crop_face_dir, video_name)
        clip_name_list = os.listdir(video_clip_dir)
        clip_name_list.sort()
        video_clip_num = len(clip_name_list)
        clip_data_list = []
        for clip_index, clip_name in enumerate(clip_name_list):
            tem_tem_dic = {}
            clip_frame_dir = os.path.join(video_clip_dir, clip_name)
            frame_path_list = glob.glob(os.path.join(clip_frame_dir, '*.jpg'))
            frame_path_list.sort()
            assert len(frame_path_list) == clip_length
            start_index = int(float(clip_name) * clip_length)
            assert int(float(os.path.basename(frame_path_list[0]).replace('.jpg', ''))) == start_index
            frame_name_list = [video_name + '/' + clip_name + '/' + os.path.basename(item) for item in frame_path_list]
            deep_speech_list = deep_speech_feature[start_index:start_index + clip_length, :].tolist()
            if len(frame_name_list) != len(deep_speech_list):
                print(' skip video: {}:{}/{}  clip:{}:{}/{} because of different length: {} {}'.format(
                    video_name,video_index,len(video_name_list),clip_name,clip_index,len(clip_name_list),
                     len(frame_name_list),len(deep_speech_list)))
            tem_tem_dic['frame_name_list'] = frame_name_list
            tem_tem_dic['frame_path_list'] = frame_path_list
            tem_tem_dic['deep_speech_list'] = deep_speech_list
            clip_data_list.append(tem_tem_dic)
        tem_dic['video_clip_num'] = video_clip_num
        tem_dic['clip_data_list'] = clip_data_list
        res_data_dic[video_name] = tem_dic
    if os.path.exists(res_json_path):
        os.remove(res_json_path)
    with open(res_json_path,'w') as f:
        json.dump(res_data_dic,f)


def extract_video_frame_multithreading(source_video_dir, res_video_frame_dir):
    '''
        extract video frames from videos 倒序处理并跳过已处理的视频，用来加速拆帧的过程
    '''

    def process_video(video_path):
        video_name = os.path.basename(video_path)
        frame_dir = os.path.join(res_video_frame_dir, video_name.replace('.mp4', ''))
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)
        print('extracting frames from {} ...'.format(video_name))
        videoCapture = cv2.VideoCapture(video_path)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        if int(fps) != 25:
            raise ValueError('{} video is not in 25 fps'.format(video_path))
        frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        for i in tqdm(range(int(frames))):
            ret, frame = videoCapture.read()
            result_path = os.path.join(frame_dir, str(i).zfill(6) + '.jpg')
            cv2.imwrite(result_path, frame)

    if not os.path.exists(source_video_dir):
        raise ValueError('wrong path of video dir')
    if not os.path.exists(res_video_frame_dir):
        os.makedirs(res_video_frame_dir)
    video_path_list = glob.glob(os.path.join(source_video_dir, '*.mp4'))

    with ThreadPoolExecutor() as executor:
        executor.map(process_video, video_path_list)

def crop_face_according_openfaceLM_multithreading(openface_landmark_dir, video_frame_dir, res_crop_face_dir, clip_length):
    '''
    crop face according to openface landmark
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    landmark_crop_dic = {}  # 定义一个左上标角的字典
    facealigner = FaceAlign(device=device, ratio=1.6)

    if not os.path.exists(openface_landmark_dir):
        raise ('wrong path of openface landmark dir')
    if not os.path.exists(video_frame_dir):
        raise ('wrong path of video frame dir')
    if not os.path.exists(res_crop_face_dir):
        os.mkdir(res_crop_face_dir)
    landmark_openface_path_list = glob.glob(os.path.join(openface_landmark_dir, '*.csv'))

    def process_video(landmark_openface_path):
        video_name = os.path.basename(landmark_openface_path).replace('.csv', '')
        crop_face_video_dir = os.path.join(res_crop_face_dir, video_name)
        if not os.path.exists(crop_face_video_dir):
            os.makedirs(crop_face_video_dir)
        print('cropping face from video: {} ...'.format(video_name))
        landmark_openface_data = load_landmark_openface_origin(landmark_openface_path).astype(np.int32)
        frame_dir = os.path.join(video_frame_dir, video_name)
        if not os.path.exists(frame_dir):
            raise ('run last step to extract video frame')
        if len(glob.glob(os.path.join(frame_dir, '*.jpg'))) != landmark_openface_data.shape[0]:
            # raise ('landmark length is different from frame length')
            print('landmark length is different from frame length', video_name)
        frame_length = min(len(glob.glob(os.path.join(frame_dir, '*.jpg'))), landmark_openface_data.shape[0])
        end_frame_index = list(range(clip_length, frame_length, clip_length))
        video_clip_num = len(end_frame_index)
        for i in tqdm(range(video_clip_num)):

            res_face_clip_dir = os.path.join(crop_face_video_dir, str(i).zfill(6))
            if not os.path.exists(res_face_clip_dir):
                os.mkdir(res_face_clip_dir)
            images = []
            landmarks = []
            for frame_index in range(end_frame_index[i] - clip_length, end_frame_index[i]):
                source_frame_path = os.path.join(frame_dir, str(frame_index).zfill(6) + '.jpg')
                source_frame_data = cv2.imread(source_frame_path)
                frame_landmark = landmark_openface_data[frame_index, :, :]
                images.append(source_frame_data)
                landmarks.append(frame_landmark)

            images_np = np.stack(images, axis=0)
            images_torch = torch.from_numpy(images_np.copy()).float().permute(0, 3, 1, 2).to(device)
            landmark_np = np.stack(landmarks, axis=0)
            landmarks_tensor = torch.from_numpy(landmark_np).float().to(device)

            # 批量正脸化并保存
            # out_W：输出图像的宽度, 高度为宽度的1.3倍
            face_align_img, lmrks_align, _ = facealigner(images_torch, landmarks_tensor, out_W=640)
            index = 0

            for frame_index_2 in range(end_frame_index[i] - clip_length, end_frame_index[i]):
                face_align_img_np = face_align_img[index].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                res_crop_face_frame_path = os.path.join(res_face_clip_dir, str(frame_index_2).zfill(6) + '.jpg')
                if os.path.exists(res_crop_face_frame_path):
                    os.remove(res_crop_face_frame_path)
                cv2.imwrite(res_crop_face_frame_path, face_align_img_np)
                landmark_crop_dic[res_crop_face_frame_path] = lmrks_align[index]
                index += 1

    # 使用线程池进行并行处理
    with ThreadPool(processes=22) as pool:
        pool.map(process_video, landmark_openface_path_list)

    print("save landmark_crop_dic.npy")
    np.save('./asserts/training_data/landmark_crop_dic.npy', landmark_crop_dic)



if __name__ == '__main__':
    opt = DataProcessingOptions().parse_args()
    ##########  step1: extract video frames
    if opt.extract_video_frame:
        extract_video_frame(opt.source_video_dir, opt.video_frame_dir)
    ##########  step2: extract audio files
    if opt.extract_audio:
        extract_audio(opt.source_video_dir,opt.audio_dir)
    ##########  step3: extract deep speech features
    if opt.extract_deep_speech:
        extract_deep_speech(opt.audio_dir, opt.deep_speech_dir,opt.deep_speech_model)
    ##########  step4: crop face images
    if opt.crop_face:
        crop_face_according_openfaceLM(opt.openface_landmark_dir,opt.video_frame_dir,opt.crop_face_dir,opt.clip_length)
    ##########  step5: generate training json file
    if opt.generate_training_json:
        generate_training_json(opt.crop_face_dir,opt.deep_speech_dir,opt.clip_length,opt.json_path)

    ################# 多线程版本的
    ##########  step6: extract video frames reverse
    if opt.extract_video_frame_multithreading:
        extract_video_frame_multithreading(opt.source_video_dir, opt.video_frame_dir)
    ##########  step7: crop face images reverse
    if opt.crop_face_multithreading:
        crop_face_according_openfaceLM_multithreading(opt.openface_landmark_dir, opt.video_frame_dir, opt.crop_face_dir,
                                             opt.clip_length)


