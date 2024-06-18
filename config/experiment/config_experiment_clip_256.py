# config_experiment.py

experiment_config = {
    'augment_num': 3,
    'mouth_region_size': 288,  # 256
    'batch_size': 2,
    'pretrained_frame_DINet_path': './output/training_model_weight/clip_training_256/checkpoint_epoch_119.pth',# 会自动根据实验名字添加子文件夹
    'result_path': './output/training_model_weight/clip_training_256', # 会自动根据实验名字添加子文件夹
    'pretrained_syncnet_path': './asserts/syncnet_256mouth.pth',
    'non_decay': 60, # 从start epoch开始算起
    'decay': 60,
    'start_epoch': 120,
    'resume': True   # 是否是断点训练
}


'''
frame_training_256 ： netG_model_epoch_3.pth
clip_training_256 ：checkpoint_epoch_1.pth + 'resume': True

'''