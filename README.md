# 改进 DINet
## 安装说明
1. 需要用到的模型
    - 包含 deepspeech 和 换脸
链接: https://pan.baidu.com/s/1bNJT409wNlcJgkiAGHcONA?pwd=ipaw 提取码: ipaw 


## 代码说明
#### 有用的小工具
1. 压缩文件并显示进度 `DINet-gitee/DINet-update/asserts/压缩文件.sh` 
2. 合并不同数据集的landmark字典 `/asserts/training_data/邓-处理大数据集.ipynb`
3. 提供给yihao的函数 `./Give_Yihao.py`



## 做数据集的代码
[基于syncnet-python修改](https://github.com/iPaw-AI-LAB/syncnet)

功能：将视频转为25fps，并检测得到landmark.csv；若原视频就为25fps，在源文件夹。若原视频不为25fps，存放在目标文件夹

1. 运行make_dataset_multi.py
    1. 修改里面的源文件夹和目标文件夹（保存25fps视频和csv的位置）
2. `test_syncnet_dengjunli.ipynb`用于移动数据集

## 做数据集步骤
1. 用做数据集的代码处理视频
2. 将视频移动到本实验的 `./assert/training_data/split_video_25fps`
3. 将视频对应的csv移动到本实验代码的`./assert/training_data/split_video_25fps_landmark_openface`
- 移动视频和对应csv的代码可以用`autodl-tmp/syncnet_python_origin/test_syncnet_dengjunli.ipynb` -
4. 运行`data_processing_正脸化.py`,得到crop后的图像
    - crop后的图像大小统一为 **416 320**，比例为1.3:1，通过`FaceAlign`类的参数`out_W`进行设置
    - 得到正脸化的crop landmark 字典 `./asserts/training_data/landmark_crop_dic.npy`
5. 若需要合并不同数据集的字典`landmark_crop_dic.npy`,可运行`autodl-tmp/DINet-gitee/DINet-update/asserts/training_data/邓-处理大数据集.ipynb`
5. 重新生成完整的json文件 `python data_processing_正脸化.py --generate_training_json` 

## 常用的训练命令/测试命令
[【腾讯文档】DINet常用命令-dengjunli](
https://docs.qq.com/doc/DTENSWFlpTVFvSkhn)


## 实验记录demo
[飞书云文档](https://y5ucgsxnni.feishu.cn/docx/QSxadxHp0o6bgLxiiEbc0nvNnZd)


## 评价指标


## 资源
[原仓库DINet](https://fuxivirtualhuman.github.io/pdf/AAAI2023_FaceDubbing.pdf)




