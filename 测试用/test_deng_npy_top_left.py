import numpy as np


# Load
load_dict = np.load('./asserts/training_data/left_top_dir.npy',allow_pickle=True).item()
# print(load_dict['./asserts/training_data/split_video_25fps_frame/WDA_JoeNeguse_001_corrected/000757.jpg'])
# print(type(load_dict['./asserts/training_data/split_video_25fps_frame/WDA_JoeNeguse_001_corrected/000758.jpg']))
print(len(load_dict))
print(type(load_dict))

search_string = "WDA_JoeNeguse_001_corrected"

matching_keys = [key for key in load_dict.keys() if search_string in key]

print(matching_keys)

import torch
print(torch.version.cuda)


import torch
print(torch.__version__)
print(torch.cuda.is_available())

print(torch.backends.cudnn.version())