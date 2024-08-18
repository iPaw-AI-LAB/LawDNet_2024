import requests

def send_audio_file(audio_file_path):
    url = 'http://localhost:5828/upload-audio'
    files = {'file': open(audio_file_path, 'rb')}
    
    try:
        response = requests.post(url, files=files)
        if response.status_code == 200:
            print("视频生成成功，下载链接:", response.url)
            with open("output_video.mp4", "wb") as f:
                f.write(response.content)
            print("视频已保存为 output_video.mp4")
        else:
            print("视频生成失败，状态码:", response.status_code)
            print("错误信息:", response.text)
    except Exception as e:
        print("请求失败:", e)

if __name__ == '__main__':
    audio_file_path = "./template/青岛3.wav"  # 替换为你的音频文件路径
    print("音频文件路径:", audio_file_path)
    print("开始上传音频文件...")
    send_audio_file(audio_file_path)
