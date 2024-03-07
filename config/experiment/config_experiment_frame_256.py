# config_experiment.py

experiment_config = {
    'augment_num': 20,
    'mouth_region_size': 256,
    'batch_size': 4,
    'coarse2fine': True,  # 由于是一个flag, 如果存在，设为True
    'coarse_model_path': './asserts/training_model_weight/frame_training_128/netG_model_epoch_1.pth',
    'result_path': './asserts/training_model_weight/frame_training_256'
}
