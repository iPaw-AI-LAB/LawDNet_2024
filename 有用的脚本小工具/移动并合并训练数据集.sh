# !/bin/bash

# 该脚本用于分批处理数据集后，将多个数据集文件夹中的内容合并到一个目标文件夹中，为后续的训练做准备。
# 进入“有用的脚本小工具文件夹”
# 定义变量
BASE_DIR="../asserts"

DATA_DIR="${BASE_DIR}/training_data_HDTF_25fps_2"
OUTPUT_DIR="${BASE_DIR}/training_data"

echo 'DATA_DIR: ' $DATA_DIR
echo 'OUTPUT_DIR: ' $OUTPUT_DIR

# 复制文件
# cp -r "$DATA_DIR"/split_video_25fps_crop_face/* "$OUTPUT_DIR"/split_video_25fps_crop_face/
# echo '1 split_video_25fps_crop_face'
# cp "$DATA_DIR"/split_video_25fps/*.mp4 "$OUTPUT_DIR"/split_video_25fps/
# echo '2 split_video_25fps'
# cp "$DATA_DIR"/split_video_25fps_audio/*.wav "$OUTPUT_DIR"/split_video_25fps_audio/
# echo '3 split_video_25fps_audio'
# cp "$DATA_DIR"/split_video_25fps_deepspeech/*.txt "$OUTPUT_DIR"/split_video_25fps_deepspeech/
# echo '4 split_video_25fps_deepspeech'
# cp "$DATA_DIR"/split_video_25fps_landmark_openface/*.csv "$OUTPUT_DIR"/split_video_25fps_landmark_openface/
# echo '5 split_video_25fps_landmark_openface'
# cp -r "$DATA_DIR"/split_video_25fps_frame/* "$OUTPUT_DIR"/split_video_25fps_frame/
# echo '6 split_video_25fps_frame'


# Count the number of .wav files in the split_video_25fps_audio directory.
# echo "Counting .wav files..."
num_wav_files=$(find "${OUTPUT_DIR}/split_video_25fps_audio" -type f -name "*.wav" | wc -l)
echo "Number of .wav files: $num_wav_files"

# Count the number of .mp4 files in the split_video_25fps directory.
# echo "Counting .mp4 files..."
num_mp4_files=$(find "${OUTPUT_DIR}/split_video_25fps" -type f -name "*.mp4" | wc -l)
echo "Number of .mp4 files: $num_mp4_files"

# Count the number of directories at the first level in split_video_25fps_crop_face.
# echo "Counting directories in split_video_25fps_crop_face..."
num_dirs_crop_face=$(find "${OUTPUT_DIR}/split_video_25fps_crop_face" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "Number of directories in split_video_25fps_crop_face: $num_dirs_crop_face"

# Count the number of directories at the first level in split_video_25fps_frame.
# echo "Counting directories in split_video_25fps_frame..."
num_dirs_frame=$(find "${OUTPUT_DIR}/split_video_25fps_frame" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "Number of directories in split_video_25fps_frame: $num_dirs_frame"

# Count the number of .txt files in the split_video_25fps_deepspeech directory.
# echo "Counting .txt files..."
num_txt_files=$(find "${OUTPUT_DIR}/split_video_25fps_deepspeech" -type f -name "*.txt" | wc -l)
echo "Number of .txt files: $num_txt_files"



# 生成训练数据
# ############# 修改training_data 文件夹名字之后 ##########
cd ..
python data_processing_正脸化.py --generate_training_json 
echo 'finally generate_training_json'