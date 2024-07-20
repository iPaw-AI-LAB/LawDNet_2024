# !/bin/bash

# 该脚本用于分批处理数据集后，将多个数据集文件夹中的内容合并到一个目标文件夹中，为后续的训练做准备。
# 进入“有用的脚本小工具文件夹”
# 定义变量
DATA_DIR="../asserts/training_data_HDTF_25fps_2"
OUTPUT_DIR="../asserts/training_data_HDTF_25fps"

# 复制文件
cp -r "$DATA_DIR"/split_video_25fps_crop_face/* "$OUTPUT_DIR"/split_video_25fps_crop_face/
echo '1 split_video_25fps_crop_face'
# cp "$DATA_DIR"/split_video_25fps/*.mp4 "$OUTPUT_DIR"/split_video_25fps/
# echo '2 split_video_25fps'
# cp "$DATA_DIR"/split_video_25fps_audio/*.wav "$OUTPUT_DIR"/split_video_25fps_audio/
# echo '3 split_video_25fps_audio'
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


# #!/bin/bash

# # 该脚本用于分批处理数据集后，将多个数据集文件夹中的内容合并到一个目标文件夹中，为后续的训练做准备。
# # 进入“有用的脚本小工具文件夹”
# # 定义变量
# DATA_DIR="../asserts/training_data_HDTF_25fps_2"
# OUTPUT_DIR="../asserts/training_data_HDTF_25fps"

# # 使用rsync命令复制文件并显示进度
# rsync -av --info=progress2 "$DATA_DIR/split_video_25fps_crop_face/" "$OUTPUT_DIR/split_video_25fps_crop_face/"
# echo '1 split_video_25fps_crop_face'

# # Uncomment the following lines if you want to copy these files as well
# # rsync -av --info=progress2 "$DATA_DIR/split_video_25fps/*.mp4" "$OUTPUT_DIR/split_video_25fps/"
# # echo '2 split_video_25fps'
# # rsync -av --info=progress2 "$DATA_DIR/split_video_25fps_audio/*.wav" "$OUTPUT_DIR/split_video_25fps_audio/"
# # echo '3 split_video_25fps_audio'

# rsync -av --info=progress2 "$DATA_DIR/split_video_25fps_deepspeech/" "$OUTPUT_DIR/split_video_25fps_deepspeech/"
# echo '4 split_video_25fps_deepspeech'

# rsync -av --info=progress2 "$DATA_DIR/split_video_25fps_landmark_openface/" "$OUTPUT_DIR/split_video_25fps_landmark_openface/"
# echo '5 split_video_25fps_landmark_openface'

# rsync -av --info=progress2 "$DATA_DIR/split_video_25fps_frame/" "$OUTPUT_DIR/split_video_25fps_frame/"
# echo '6 split_video_25fps_frame'

# # 生成训练数据
# # ############# 修改training_data 文件夹名字之后 ##########
# cd ..
# python data_processing_正脸化.py --generate_training_json 
# echo 'finally generate_training_json'
