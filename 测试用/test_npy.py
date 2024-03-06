import numpy as np

load_dict = np.load('./asserts/training_data/landmark_crop_dic.npy',allow_pickle=True).item()
landmark_test = load_dict['./asserts/training_data/split_video_25fps_crop_face/RD_Radio52_000_corrected/000000/000000.jpg']
# print(type(load_dict['./asserts/training_data/split_video_25fps_frame/WDA_JoeNeguse_001_corrected/000758.jpg']))
print(len(load_dict))
print(type(load_dict))
# print(load_dict.keys())
import pdb
pdb.set_trace()
print(type(landmark_test))
print(landmark_test)
print(len(landmark_test[0]))