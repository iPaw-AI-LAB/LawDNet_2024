# from flask import Flask, request, send_file
# import os

# app = Flask(__name__)

# @app.route('/get-video', methods=['GET'])
# def get_video():
#     video_path = '/home/dengjunli/data/dengjunli/autodl拿过来的/DINet-update/Exp-of-Junli/RD_Radio34_000.mp4'
#     if os.path.exists(video_path):
#         return send_file(video_path, as_attachment=True)
#     else:
#         return "Video not found", 404

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5823)

from flask import Flask, request, send_file
import os
from inference_function-DP2 import generate_video_with_audio

app = Flask(__name__)

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    print("file.filename: ", file.filename)
    if file.filename == '':
        return "No selected file", 400
    
    audio_path = os.path.join('./', file.filename)
    file.save(audio_path)
    
    # 处理音频文件并生成视频文件的逻辑
    video_path = process_audio_and_generate_video(audio_path)
    
    if os.path.exists(video_path):
        return send_file(video_path, as_attachment=True)
    else:
        return "Video not found", 404

def process_audio_and_generate_video(audio_path):
    # 在这里添加处理音频文件并生成视频文件的逻辑
    # 例如，将生成的视频文件路径返回
    print("audio_path: ", audio_path)

    video_path = './template/丽娟质检中传播音作品片段展示含中英新闻播报模拟主持25fps_prjf.mp4'
    output_dir = './output_video'
    # 设置模型文件路径
    deepspeech_model_path = "../asserts/output_graph.pb"
    # lawdnet_model_path =  "/home/dengjunli/data/dengjunli/autodl拿过来的/DINet-update/output/training_model_weight/288-mouth-CrossAttention-插值coarse-to-fine-2/clip_training_256/checkpoint_epoch_120.pth"
    lawdnet_model_path = "../output/training_model_weight/288-mouth-CrossAttention-插值coarse-to-fine-shengshu/clip_training_256/checkpoint_epoch_599.pth"
    BatchSize = 20
    mouthsize = '288'
    gpu_index = 1
    result_video_path = generate_video_with_audio(video_path, 
                                                  audio_path,
                                                  deepspeech_model_path, 
                                                  lawdnet_model_path,
                                                  output_dir,
                                                  BatchSize,
                                                  mouthsize,
                                                  gpu_index,
                                                  )
    
    return result_video_path

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5828)

