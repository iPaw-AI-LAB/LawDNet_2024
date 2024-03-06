from moviepy.editor import VideoFileClip
from moviepy.editor import *

def change_fps(video, new_fps):
    # # 计算播放速度的倍数（等于新帧率除以原帧率）
    speed_factor = new_fps / video.fps

    
    # # 创建 speedx 效果对象
    # video = video.speedx(speed_factor)

    # 将视频的帧率设置为 25fps
    video = video.set_fps(25)

    # 将视频的播放速度设置为 1.0（不变）
    video = video.fx(vfx.speedx, factor=speed_factor)

    
    return video

# 读取视频
video = VideoFileClip(r'D:\dengjunli\test\1.mp4')

audio_origin = video.audio

# 改变帧率为25fps，同时保持播放速度不变
video = change_fps(video, 25)

# 把原音频添加上去
video = video.set_audio(audio_origin)


# 输出新视频
video.write_videofile(r'D:\dengjunli\test\1——output.mp4')