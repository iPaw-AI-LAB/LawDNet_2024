import os
import tarfile

# 定义文件夹路径
folder_path = r"E:\MEAD情绪人脸多视角talkinghead数据集"

def extract_tar_files(folder_path):
    # 遍历文件夹中的子文件夹及文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件名以 ".tar" 为后缀，并且只解压根目录下的文件
            if file.endswith("video.tar") and root == folder_path:
                # print(f"开始解压：{file}")
                file_path = os.path.join(root, file)
                print("正在解压",file_path)
                # 提取文件名（去除后缀），作为解压后的文件夹名
                output_folder = os.path.splitext(file)[0]
                output_folder_path = os.path.join(root, output_folder)

                # 创建解压后的文件夹
                if not os.path.exists(output_folder_path):
                    os.makedirs(output_folder_path)

                # 打开tar文件并解压
                with tarfile.open(file_path, "r") as tar:
                    tar.extractall(output_folder_path)

                print(f"解压完成：{file_path}")

        # 递归调用，遍历下一级子文件夹
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            extract_tar_files(subdir_path)

# 调用函数开始解压
extract_tar_files(folder_path)
