import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip


def split_videos(folder_path, output_folder):
    # 获取文件夹内所有视频文件
    videos = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.mp4')]
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    for video in videos:
        video_path = os.path.join(folder_path, video)
        clip = VideoFileClip(video_path)
        duration = clip.duration

        if duration > 60:
            # 计算需要拆分为多少段
            num_parts = int(duration // 60) + 1

            for i in range(num_parts):
                start_time = i * 60
                end_time = (i + 1) * 60

                # 拆分每一段视频
                part_video_path = os.path.join(output_folder, f'{video[:-4]}_part{i+1}.mp4')
                ffmpeg_extract_subclip(video_path, start_time, end_time, targetname=part_video_path)
                print(f'Successfully split video {video} into part {i+1}')

        else:
            print(f'Video {video} is already within 1 minute')


# 指定原始视频文件夹路径和输出文件夹路径
input_folder = r'D:\dengjunli\bilibili_dataset_25fps_FFoutput'
output_folder = r'D:\dengjunli\HDTF_bilibili_25fps_1min'

split_videos(input_folder, output_folder)
