import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt



def concat_ref_and_src(src, ref):
    '''
    input: B*1,15,h,w ; B*1,15,h,w  （原图 ，参考帧）
    output: B*1,30,h,w
    '''

    # 原图和参考帧的张量维度
    original_frames = src
    reference_frames = ref

    # 提取A人物的原图（1，15，h，w）
    A_original = original_frames[0:1, :, :, :]
    # 提取A人物的参考帧（1，15，h，w）
    A_reference = reference_frames[0:1, :, :, :]

    # 提取B人物的原图（1，15，h，w）
    B_original = original_frames[1:2, :, :, :]
    # 提取B人物的参考帧（1，15，h，w）
    B_reference = reference_frames[1:2, :, :, :]

    # 将A人物的原图和参考帧按通道拼接为（1，30，h，w）
    # import pdb; pdb.set_trace()
    try:
        A_combined = torch.cat((A_original, A_reference), dim=1)
    except Exception as e:
        import pdb; pdb.set_trace()

    # 将B人物的原图和参考帧按通道拼接为（1，30，h，w）
    B_combined = torch.cat((B_original, B_reference), dim=1)

    combined = torch.cat((A_combined, B_combined), dim=0)

    return combined

def gamma_correction(img, gamma=2.0):
    # 使用Gamma值校正图像
    corrected = np.power(img, 1 / gamma)
    return np.clip(corrected, 0, 1)

def save_feature_map(ref_in_feature,filename_prefix):

    # plt.imshow(feature_image, cmap='viridis')  # 使用viridis色图，您也可以选择其他色图如'gray'
    # plt.axis('off')
    # plt.colorbar()  # 添加颜色条以显示数值范围
    # plt.savefig('feature_visualization.png', bbox_inches='tight', pad_inches=0)

    # # 归一化到0-255范围
    # feature_image_normalized = ((feature_image - feature_image.min()) / (feature_image.max() - feature_image.min()) * 255).astype(np.uint8)
    # cv2.imwrite('/home/dengjunli/data/dengjunli/autodl拿过来的/DINet-update/Exp-of-Junli/特征图可视化-feature_visualization.png', feature_image_normalized)
    # print('特征图可视化-feature_visualization.png已保存')

    # 可视化并保存ref_in_feature的前几个特征通道
    num_channels_to_save = 10  # 您可以修改这个值以保存更多或更少的通道
    image_path = "/home/dengjunli/data/dengjunli/autodl拿过来的/DINet-update/Exp-of-Junli/"

    for channel in range(num_channels_to_save):
        feature_image = ref_in_feature[0, channel].cpu().detach().numpy()

        # 使用Gamma校正提亮特征图像
        feature_image = gamma_correction(feature_image)

        # 使用matplotlib的viridis colormap进行彩色映射
        feature_colored = plt.get_cmap('viridis')((feature_image - feature_image.min()) / (feature_image.max() - feature_image.min()))
        feature_colored = (feature_colored[:, :, :3] * 255).astype(np.uint8)  # 去除alpha通道并转换为0-255


        # 为每个通道保存一个唯一的文件名
        filename = f'{filename_prefix}_channel_{channel}_colored.png'
        cv2.imwrite(image_path + filename, cv2.cvtColor(feature_colored, cv2.COLOR_RGB2BGR))

        # 归一化到0-255范围
        feature_image_normalized = ((feature_image - feature_image.min()) / (feature_image.max() - feature_image.min()) * 255).astype(np.uint8)
        
        # 为每个通道保存一个唯一的文件名
        filename = f'{filename_prefix}_channel_{channel}_gray.png'
        cv2.imwrite(image_path + filename, feature_image_normalized)
        print(f'{filename_prefix}_channel_{channel}_colored.png', '已保存')

    return None

