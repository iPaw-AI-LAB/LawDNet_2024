# This file is used to set the experiment configuration for training the frame-level model with mouth region size 64x64.
experiment_config = {
    'augment_num': 32,
    'mouth_region_size': 64,
    'batch_size': 16,
    'coarse2fine': False,  # 从随机初始化开始训练
    'result_path': './output/training_model_weight/frame_training_64', # 会自动根据实验名字添加子文件夹
    'non_decay': 5,
    'decay': 5
}
