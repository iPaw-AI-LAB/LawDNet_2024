# config_experiment.py

experiment_config = {
    'augment_num': 3,
    'mouth_region_size': 288,  # 256
    'batch_size': 4,
    'coarse2fine': True,  # 从预训练模型开始训练
    'pretrained_frame_DINet_path': './output/training_model_weight/frame_training_256/netG_model_epoch_1.pth',
    'result_path': './output/training_model_weight/clip_training_256'
}
