#!/bin/bash

# 定义实验名称
EXPERIMENT_NAME="test_jinpeng_hdtf_dp2_288"

# 执行第一个训练任务
python train_LawDNet_frame.py --config_path "./config/experiment/config_experiment_frame_64.py" --name "$EXPERIMENT_NAME"

# # 执行第二个训练任务
# python train_LawDNet_frame.py --config_path "./config/experiment/config_experiment_frame_128.py" --name "$EXPERIMENT_NAME"

# # 执行第三个训练任务
# python train_LawDNet_frame.py --config_path "./config/experiment/config_experiment_frame_256.py" --name "$EXPERIMENT_NAME"

# # 执行第四个训练任务
# python train_LawDNet_clip.py --config_path "./config/experiment/config_experiment_clip_256.py" --name "$EXPERIMENT_NAME"
