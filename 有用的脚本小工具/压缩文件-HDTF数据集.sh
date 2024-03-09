#!/bin/bash

# # 要压缩的文件夹路径
# folder_path="/root/autodl-tmp/DINet-gitee/DINet-update/asserts/training_data/split_video_25fps_crop_face"
# # 压缩后的 zip 文件路径和名称
# zip_path="/root/autodl-tmp/DINet-gitee/DINet-update/asserts/training_data/split_video_25fps_crop_face.zip"

# # 要压缩的文件夹路径
folder_path="./training_data-正脸化的crop"
# # 压缩后的 zip 文件路径和名称
compressed_path="./HDTF数据集training_data目录.tar.gz"

extracted_path="./解压"


# 使用 tar 命令将文件夹打包成 tar 文件，并使用 pv 命令显示进度
tar -cf - "${folder_path}" | pv -p -s $(du -sb "${folder_path}" | awk '{print $1}') | \
    gzip > "${compressed_path}"



# 显示压缩完成消息
echo "压缩完成"



# # 创建目标文件夹
# mkdir -p "${extracted_path}"

# # 解压缩 tar.gz 文件，并使用 pv 命令显示进度
# pv "${compressed_path}" | tar -xzf - -C "${extracted_path}"

# echo "解压完成"
