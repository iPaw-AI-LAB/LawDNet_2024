#!/bin/bash

# 定义实验名称
EXPERIMENT_NAME="test_2024—多卡"

# 设置使用的GPU编号
export CUDA_VISIBLE_DEVICES=2,3

# 定义主节点地址和端口
MASTER_ADDR="localhost"
MASTER_PORT="47877"

# 执行训练任务
# 注意：根据您的需求调整 MASTER_ADDR 和 MASTER_PORT 的值

# 训练单帧模型，帧分辨率为64x64
torchrun --nproc_per_node=2 train_LawDNet_frame_distributed.py --config_path "./config/experiment/config_experiment_frame_64.py" --name "$EXPERIMENT_NAME" --master_addr $MASTER_ADDR --master_port $MASTER_PORT

# 训练单帧模型，帧分辨率为128x128
torchrun --nproc_per_node=2 train_LawDNet_frame_distributed.py --config_path "./config/experiment/config_experiment_frame_128.py" --name "$EXPERIMENT_NAME" --master_addr $MASTER_ADDR --master_port $MASTER_PORT

# 训练单帧模型，帧分辨率为256x256
torchrun --nproc_per_node=2 train_LawDNet_frame_distributed.py --config_path "./config/experiment/config_experiment_frame_256.py" --name "$EXPERIMENT_NAME" --master_addr $MASTER_ADDR --master_port $MASTER_PORT

# 训练多帧模型，帧分辨率为256x256
torchrun --nproc_per_node=2 train_LawDNet_clip_distributed.py --config_path "./config/experiment/config_experiment_clip_256.py" --name "$EXPERIMENT_NAME" --master_addr $MASTER_ADDR --master_port $MASTER_PORT
