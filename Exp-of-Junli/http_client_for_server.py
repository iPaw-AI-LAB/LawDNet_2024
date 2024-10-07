import requests
import os

def generate_video(audio_file_path, server_url='http://localhost:5051/generate-video'):
    """
    向服务器发送音频文件并接收生成的视频。

    参数:
    audio_file_path (str): 音频文件的路径
    server_url (str): 服务器的URL

    返回:
    str: 生成的视频文件的路径
    """
    # 检查音频文件是否存在
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"音频文件 '{audio_file_path}' 不存在")

    # 准备文件上传
    files = {'audio': open(audio_file_path, 'rb')}

    try:
        # 发送POST请求
        response = requests.post(server_url, files=files)
        
        # 检查响应状态
        response.raise_for_status()

        # 保存接收到的视频文件
        video_file_path = 'out_video_http_server_hn.mp4'
        with open(video_file_path, 'wb') as f:
            f.write(response.content)

        print(f"视频已生成并保存为 '{video_file_path}'")
        return video_file_path

    except requests.RequestException as e:
        print(f"请求失败: {e}")
        return None
    finally:
        # 确保文件被关闭
        files['audio'].close()

if __name__ == '__main__':
    audio_file = './青岛3.wav'  # 替换为你的音频文件路径
    generate_video(audio_file)