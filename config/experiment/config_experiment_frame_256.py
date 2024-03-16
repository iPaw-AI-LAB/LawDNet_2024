# config_experiment.py

experiment_config = {
    'augment_num': 20,
    'mouth_region_size': 256,
    'batch_size': 8,
    'coarse2fine': True,  # 从预训练模型开始训练
    'coarse_model_path': './output/training_model_weight/frame_training_128/netG_model_epoch_1.pth',# 会自动根据实验名字添加子文件夹
    'result_path': './output/training_model_weight/frame_training_256', # 会自动根据实验名字添加子文件夹
    'non_decay': 2,
    'decay': 2
}
