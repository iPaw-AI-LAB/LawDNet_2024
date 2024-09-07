import requests
import time

def send_audio_for_video_generation(audio_file_path, server_url):
    with open(audio_file_path, 'rb') as audio_file:
        files = {'audio': audio_file}
        start_time = time.time()
        response = requests.post(f"{server_url}/generate_video", files=files)
        end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"服务器响应时间: {elapsed_time:.2f} 秒")

    if response.status_code == 200:
        # 保存返回的视频
        with open('generated_video.mp4', 'wb') as video_file:
            video_file.write(response.content)
        print("视频生成成功并已保存为 ./generated_video.mp4")
    else:
        print(f"错误: {response.text}")

if __name__ == "__main__":
    audio_file_path = "./data/青岛3.wav"
    server_url = "http://localhost:5000"  # 根据实际情况修改服务器地址
    send_audio_for_video_generation(audio_file_path, server_url)