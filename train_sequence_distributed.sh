#!/bin/bash

# 手动移动frame训练好的模型

# 设置环境变量
export OMP_NUM_THREADS=1

conda activate lawdnet

# 定义实验名称
EXPERIMENT_NAME="288-mouth-CrossAttention-插值coarse-to-fine"

## 记得迁移文件夹到对应的位置

# 设置使用的GPU编号
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 0,1,2,3 表示使用编号为 0,1,2,3 的 GPU
GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
echo "Using $GPU_COUNT GPUs: $CUDA_VISIBLE_DEVICES"

# 定义主节点地址和端口
MASTER_ADDR="localhost"
MASTER_PORT="29456"

## debug 选项
export TORCH_DISTRIBUTED_DEBUG=DETAIL

## 训练单帧模型，帧分辨率为64x64
# pkill -f torchrun ; torchrun --nproc_per_node=$GPU_COUNT train_LawDNet_frame_distributed.py --config_path "./config/experiment/config_experiment_frame_64.py" --name "$EXPERIMENT_NAME" --master_addr $MASTER_ADDR --master_port $MASTER_PORT
# echo "finish training 64x64"

# # ####### 训练单帧模型，帧分辨率为128x128
# pkill -f torchrun ; torchrun --nproc_per_node=$GPU_COUNT train_LawDNet_frame_distributed.py --config_path "./config/experiment/config_experiment_frame_128.py" --name "$EXPERIMENT_NAME" --master_addr $MASTER_ADDR --master_port $MASTER_PORT
# echo "finish training 128x128"

# ####### 训练单帧模型，帧分辨率为256x256
# pkill -f torchrun ; torchrun --nproc_per_node=$GPU_COUNT train_LawDNet_frame_distributed.py --config_path "./config/experiment/config_experiment_frame_256.py" --name "$EXPERIMENT_NAME" --master_addr $MASTER_ADDR --master_port $MASTER_PORT
# echo "finish training 256x256"

######### 训练多帧模型，帧分辨率为256x256
pkill -f torchrun ; torchrun --nproc_per_node=$GPU_COUNT train_LawDNet_clip_distributed.py --config_path "./config/experiment/config_experiment_clip_256.py" --name "$EXPERIMENT_NAME" --master_addr $MASTER_ADDR --master_port $MASTER_PORT
echo "finish training, the experiment name is $EXPERIMENT_NAME"