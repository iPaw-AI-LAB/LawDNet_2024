import os
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import os
import cv2
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip

input_folder = r'D:\dengjunli\test'
output_folder = r'D:\dengjunli\test_output'



if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for file in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file)

    # 检查文件是否是视频文件
    if os.path.isfile(file_path) and file_path.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
        
        # 使用 OpenCV 获取视频的帧率
        video_capture = cv2.VideoCapture(file_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        video_capture.release()

        # 检查视频的帧率是否为 25fps
        if fps != 25:
            output_path = os.path.join(output_folder, file)

            # 加载视频文件
            video = VideoFileClip(file_path)

            # 保存原始音频
            audio_path = os.path.join(output_folder, f'{os.path.splitext(file)[0]}.wav')
            video.audio.write_audiofile(audio_path)

            audio = AudioFileClip(audio_path)

            # 将原始音频拼接到视频中
            video = video.set_audio(audio).set_fps(25)

            # 设置视频的帧率为 25fps，并保存
            video.write_videofile(output_path, fps=25, codec='mpeg4', audio_codec='aac')

            print(f'{file} converted to 25fps and merged with original audio')
        else:
            print(f'Skipped {file}, already 25fps')

print('Conversion completed.')
