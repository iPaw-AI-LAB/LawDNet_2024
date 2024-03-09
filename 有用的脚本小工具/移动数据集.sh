# !/bin/bash

# 定义变量
DATA_DIR="./training_data_金鹏高清棚拍数据集汇总"
OUTPUT_DIR="./training_data"

# 复制文件
cp -r "$DATA_DIR"/split_video_25fps_crop_face/* "$OUTPUT_DIR"/split_video_25fps_crop_face/
echo '1 split_video_25fps_crop_face'
cp "$DATA_DIR"/split_video_25fps/*.mp4 "$OUTPUT_DIR"/split_video_25fps/
echo '2 split_video_25fps'
cp "$DATA_DIR"/split_video_25fps_audio/*.wav "$OUTPUT_DIR"/split_video_25fps_audio/
echo '3 split_video_25fps_audio'
cp "$DATA_DIR"/split_video_25fps_deepspeech/*.txt "$OUTPUT_DIR"/split_video_25fps_deepspeech/
echo '4 split_video_25fps_deepspeech'
cp "$DATA_DIR"/split_video_25fps_landmark_openface/*.csv "$OUTPUT_DIR"/split_video_25fps_landmark_openface/
echo '5 split_video_25fps_landmark_openface'
cp -r "$DATA_DIR"/split_video_25fps_frame/* "$OUTPUT_DIR"/split_video_25fps_frame/
echo '6 split_video_25fps_frame'

# 生成训练数据
# ############# 修改training_data 文件夹名字之后 ##########
cd ..
python data_processing_正脸化.py --generate_training_json 
echo 'finally generate_training_json'