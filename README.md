# 改进 DINet
## 安装说明
1. 需要用到的模型
    - 包含 deepspeech 和 换脸
链接: https://pan.baidu.com/s/1bNJT409wNlcJgkiAGHcONA?pwd=ipaw 提取码: ipaw 


# 代码使用说明

## 训练 
采用coarse to fine 的训练策略，每个阶段有自己的config文件，父文件是```./config/config.py```

### DDP 并行训练方式 - 快
打开```train_sequence_distributed.sh``` 修改NAME(实验名称)
直接执行脚本：```sh train_sequence_distributed.sh```



### DP并行方式训练-慢
或直接执行脚本：```sh ./train_sequence.sh```



### 模型保存路径
```autodl拿过来的/DINet-update/output/training_model_weight/NAME(实验名称)```

[基于codebase仓库DINet重构](https://fuxivirtualhuman.github.io/pdf/AAAI2023_FaceDubbing.pdf)

## 测试：
```./Exp-of-Junli/optimized-prediction-deng.ipynb```

## 查看wandb 训练日志
```https://wandb.ai/ai-zhua``

#### 有用的小工具
1. 压缩文件并显示进度 `./有用的脚本小工具/压缩文件.sh` 
2. 合并不同数据集的landmark字典 `/有用的脚本小工具/邓-处理大数据集.ipynb`
3. 音频处理的函数 `./audio_processing.py`


# 做数据集的代码
[基于syncnet-python修改](https://github.com/iPaw-AI-LAB/syncnet)

功能：将视频转为25fps，并检测得到landmark.csv；若原视频就为25fps，在源文件夹。若原视频不为25fps，存放在目标文件夹

1. 运行make_dataset_multi.py
    1. 修改里面的源文件夹和目标文件夹（保存25fps视频和csv的位置）
2. `test_syncnet_dengjunli.ipynb`用于移动数据集

## 做数据集步骤
主要代码```data_processing_正脸化.py```

## 使用方法

本脚本支持多种命令行参数，以控制其操作。以下是可用选项及其描述：

```bash
python data_processing_正脸化.py [OPTIONS]
```

### 可用选项

- `--extract_video_frame`：启用从源视频中提取视频帧。
- `--extract_audio`：启用从源视频中提取音频。
- `--extract_deep_speech`：启用从音频文件中提取DeepSpeech特征。
- `--crop_face`：启用根据OpenFace地标裁剪人脸。
- `--generate_training_json`：启用生成训练JSON文件。
- `--extract_video_frame_multithreading`：启用多线程提取视频帧，以提高效率。
- `--crop_face_multithreading`：启用多线程裁剪人脸，以提高效率。

### 步骤指南

1. **提取视频帧：**
   从指定目录中的视频文件提取帧。
   ```bash
   python data_processing_正脸化.py --extract_video_frame
   ```
   
2. **提取音频：**
   将视频文件中的音轨提取为WAV格式。
   ```bash
   python data_processing_正脸化.py --extract_audio
   ```

3. **提取DeepSpeech特征：**
   处理提取的音频文件，计算并保存DeepSpeech特征。
   ```bash
   python data_processing_正脸化.py --extract_deep_speech
   ```

4. **根据landmark裁剪人脸：**
   使用landmark从视频帧中裁剪人脸。
   ```bash
   python data_processing_正脸化.py --crop_face
   ```

5. **生成训练JSON文件：**
   生成一个JSON文件，列出视频帧、音频特征及元数据路径，用于训练。
   ```bash
   python data_processing_正脸化.py --generate_training_json
   ```

6. **多线程提取视频帧（可选）：**
   使用多线程更高效地提取视频帧。
   ```bash
   python data_processing_正脸化.py --extract_video_frame_multithreading
   ```

7. **多线程裁剪人脸（可选）：**
   使用多线程更高效地从视频帧中裁剪人脸。
   ```bash
   python data_processing_正脸化.py --crop_face_multithreading
   ```

## 注意事项

- 在运行脚本之前，请确保所有所需的输入目录存在且包含必要的文件。

#### 数据处理脚本

1. 用做数据集的代码处理视频
2. 将视频移动到本实验的 `./assert/training_data/split_video_25fps`
3. 将视频对应的csv移动到本实验代码的`./assert/training_data/split_video_25fps_landmark_openface`
- 移动视频和对应csv的代码可以用`autodl-tmp/syncnet_python_origin/test_syncnet_dengjunli.ipynb` -
4. 运行`data_processing_正脸化.py`,得到crop后的图像
    - crop后的图像大小统一为 **416 320**，比例为1.3:1，通过`FaceAlign`类的参数`out_W`进行设置
    - 得到正脸化的crop landmark 字典 `./asserts/training_data/landmark_crop_dic.npy`
5. 若需要合并不同数据集的字典`landmark_crop_dic.npy`,可运行`autodl-tmp/DINet-gitee/DINet-update/asserts/training_data/邓-处理大数据集.ipynb`
5. 重新生成完整的json文件 `python data_processing_正脸化.py --generate_training_json` 

- 若要更换数据集，请将数据集命名为training_data
- 测试用的小数据集 training_data-一个中国人

## 常用的训练命令/测试命令
[【腾讯文档】DINet常用命令-dengjunli](
https://docs.qq.com/doc/DTENSWFlpTVFvSkhn)

### 实验记录+demo
[飞书云文档实验记录和demo](https://y5ucgsxnni.feishu.cn/docx/QSxadxHp0o6bgLxiiEbc0nvNnZd)

### 论文地址
[overleaf 需审批](https://www.overleaf.com/read/vkhhnxrvwbdw#3778eb)

### 宣传网页
[LawDNet主页]https://github.com/Fannovel16/ComfyUI-MotionDiff

### 评价指标
[数据集评测指标代码](https://gitee.com/dengjunli/evaluation_wav2lip)

### 资源
[原仓库DINet](https://fuxivirtualhuman.github.io/pdf/AAAI2023_FaceDubbing.pdf)


### bug
1. 处理数据集的时候，金鹏数据集的crop face的长度和deepspeech长度不一致
2. 音频的deepspeech帧数总比视频帧多3帧







