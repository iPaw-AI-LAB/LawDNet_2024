# This file is used to set the experiment configuration for clip training
experiment_config = {
    'augment_num': 3,
    'mouth_region_size': 288,  # 256
    'batch_size': 16,
    'pretrained_frame_DINet_path': './output/training_model_weight/frame_training_256/netG_model_epoch_5.pth', # 填写上一轮coarse训练的模型路径
    'result_path': './output/training_model_weight/clip_training_256', # 会自动根据实验名字添加子文件夹
    'pretrained_syncnet_path': './asserts/syncnet_256mouth.pth',
    'non_decay': 60, # 从start epoch开始算起
    'decay': 60,
    'start_epoch': 1,
    'resume': False   # 是否是clip的断点训练
}




''' Example
本次clip训练的断点训练：'resume': True ，对应的start_epoch: 断点训练的epoch
coarse to fine 训练，clip 阶段从零开始：'resume': False ， 对应的start_epoch: 1

会自动根据实验名称完整名字：
pretrained_frame_DINet_path: "./output/training_model_weight/{experiment_name}/frame_training_256/netG_model_epoch_6.pth"

最终的result_path：
LawDNet_2024/output/training_model_weight/{experiment_name}/clip_training_256

frame_training_256 ： netG_model_epoch_3.pth
clip_training_256 ：checkpoint_epoch_1.pth with 'resume': True
'''