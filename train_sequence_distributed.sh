# #!/bin/bash

# # 手动移动frame训练好的模型

# # 设置环境变量
# export OMP_NUM_THREADS=1

# # conda activate lawdnet

# # 定义实验名称
# EXPERIMENT_NAME="288-mouth-CrossAttention-HDTF-bilibili-1"

# ## 记得迁移文件夹到对应的位置

# # 设置使用的GPU编号
# export CUDA_VISIBLE_DEVICES=0,1,2,3  # 0,1,2,3 表示使用编号为 0,1,2,3 的 GPU
# GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
# echo "Using $GPU_COUNT GPUs: $CUDA_VISIBLE_DEVICES"

# # 定义主节点地址和端口
# MASTER_ADDR="localhost"
# MASTER_PORT="29481"

# ## debug 选项
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# ######## 训练单帧模型，帧分辨率为64x64
# pkill -f torchrun ; torchrun --nproc_per_node=$GPU_COUNT train_LawDNet_frame_distributed.py --config_path "./config/experiment/config_experiment_frame_64.py" --name "$EXPERIMENT_NAME" --master_addr $MASTER_ADDR --master_port $MASTER_PORT
# echo "finish training 64x64"

# # ####### 训练单帧模型，帧分辨率为128x128
# pkill -f torchrun ; torchrun --nproc_per_node=$GPU_COUNT train_LawDNet_frame_distributed.py --config_path "./config/experiment/config_experiment_frame_128.py" --name "$EXPERIMENT_NAME" --master_addr $MASTER_ADDR --master_port $MASTER_PORT
# echo "finish training 128x128"

# ####### 训练单帧模型，帧分辨率为256x256
# pkill -f torchrun ; torchrun --nproc_per_node=$GPU_COUNT train_LawDNet_frame_distributed.py --config_path "./config/experiment/config_experiment_frame_256.py" --name "$EXPERIMENT_NAME" --master_addr $MASTER_ADDR --master_port $MASTER_PORT
# echo "finish training 256x256"

# ######### 训练多帧模型，帧分辨率为256x256
# pkill -f torchrun ; torchrun --nproc_per_node=$GPU_COUNT train_LawDNet_clip_distributed.py --config_path "./config/experiment/config_experiment_clip_256.py" --name "$EXPERIMENT_NAME" --master_addr $MASTER_ADDR --master_port $MASTER_PORT
# echo "finish training, the experiment name is $EXPERIMENT_NAME"



######## shengshu

#!/bin/bash

# 手动移动frame训练好的模型

# 设置环境变量
export OMP_NUM_THREADS=1

# conda activate lawdnet

# 定义实验名称
EXPERIMENT_NAME="288-mouth-CrossAttention-HDTF-bilibili-xhs"

## 记得迁移文件夹到对应的位置

# 设置使用的GPU编号
export CUDA_VISIBLE_DEVICES=2,3  # 0,1,2,3 表示使用编号为 0,1,2,3 的 GPU
GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
echo "Using $GPU_COUNT GPUs: $CUDA_VISIBLE_DEVICES"

# 定义主节点地址
MASTER_ADDR="localhost"

## debug 选项
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# 检查并等待端口释放函数
wait_for_port_release() {
  local port=$1
  while lsof -i :$port; do
    echo "Port $port is still in use, waiting..."
    sleep 5
  done
}

# 记录 torchrun 进程的 PID 并终止
run_torchrun_and_wait() {
  local config_path=$1
  local experiment_name=$2
  local master_port=$3

  wait_for_port_release $master_port
  torchrun --nproc_per_node=$GPU_COUNT train_LawDNet_frame_distributed.py --config_path "$config_path" --name "$experiment_name" --master_addr $MASTER_ADDR --master_port $master_port &
  local torchrun_pid=$!
  wait $torchrun_pid
}

run_torchrun_and_wait_clip() {
  local config_path=$1
  local experiment_name=$2
  local master_port=$3

  wait_for_port_release $master_port
  torchrun --nproc_per_node=$GPU_COUNT train_LawDNet_clip_distributed.py --config_path "$config_path" --name "$experiment_name" --master_addr $MASTER_ADDR --master_port $master_port &
  local torchrun_pid=$!
  wait $torchrun_pid
}

# ######## 训练单帧模型，帧分辨率为64x64
# MASTER_PORT="29481"
# run_torchrun_and_wait "./config/experiment/config_experiment_frame_64.py" "$EXPERIMENT_NAME" $MASTER_PORT
# echo "finish training 64x64"

# # ####### 训练单帧模型，帧分辨率为128x128
# MASTER_PORT="29482"
# run_torchrun_and_wait "./config/experiment/config_experiment_frame_128.py" "$EXPERIMENT_NAME" $MASTER_PORT
# echo "finish training 128x128"

# ####### 训练单帧模型，帧分辨率为256x256
# MASTER_PORT="29483"
# run_torchrun_and_wait "./config/experiment/config_experiment_frame_256.py" "$EXPERIMENT_NAME" $MASTER_PORT
# echo "finish training 256x256"

######### 训练多帧模型，帧分辨率为256x256
MASTER_PORT="29484"
run_torchrun_and_wait_clip "./config/experiment/config_experiment_clip_256.py" "$EXPERIMENT_NAME" $MASTER_PORT
echo "finish training, the experiment name is $EXPERIMENT_NAME"

