# LawDnet 2024

### BUG
1. 处理数据集的时候，jinpeng数据集的crop face的长度和deepspeech长度不一致
2. 音频的deepspeech帧数总比视频帧多3帧
3. 对loss的权重敏感，尤其是syncnet_loss,导致震荡严重，但是不影响训练结果

### 改进
1. 根据音频的长度来确定拆帧的数量，推理时

## 安装说明
先安装tensorflow_gpu = 1.15, 模型whl在[百度网盘](https://pan.baidu.com/s/1bNJT409wNlcJgkiAGHcONA?pwd=ipaw)， 提取码：ipaw 

#### conda安装
```bash
conda install -c conda-forge ffmpeg
pip install -r requirements.txt
```

- 需要用到的模型
    - 包含 换脸 ，deepspeech(tensorflow), vgg
链接: https://pan.baidu.com/s/1bNJT409wNlcJgkiAGHcONA?pwd=ipaw 提取码: ipaw 

##### 模型文件放置位置：
- output_graph.pb : pretrained deepspeech model of tensorflow 1.15，放在```./asserts/```
- syncnet_256mouth.pth: 用于唇形同步损失，放在```./asserts/```



# 代码使用说明

## 训练 
采用 coarse to fine 的训练策略，每个阶段有自己的config文件，位于```./config/experiment ``` 
包括：


基础配置文件是```./config/config.py```

### DDP 并行训练方式 - 快
```python
sh train_sequence_distributed.sh
# 对应参数在config.py 和 train_sequence_distributed.sh 中修改
```
- 模型保存位置： ```./output/training_model_weight/NAME(实验名称)```

打开```train_sequence_distributed.sh``` 修改NAME(实验名称)
直接执行脚本：```sh train_sequence_distributed.sh```

### DP并行方式训练-慢
直接执行脚本：```sh ./train_sequence.sh```


## 测试：
``` cd ./Exp-of-Junli/ ```
``` python inference_function.py```
```./Exp-of-Junli/optimized-prediction-deng.ipynb # 方便单步调试看中间结果```
```./Exp-of-Junli/inference_function.py # 修改模型路径，数字人视频，音频```
```./Exp-of-Junli/server_LawDNet.py # 提供server服务```

## server服务
下载预训练模型，测试视频：[百度网盘](https://pan.baidu.com/s/1FFINqyyz2to96_-A7QhhHA?pwd=ipaw) 提取码: ipaw 

```sh
cd Exp-of-Junli;
python server_LawDNet.py
```

# 视频生成器脚本说明

本脚本用于生成视频，并与音频同步。以下是脚本中使用的参数说明：

## 参数列表

| 参数名称               | 描述                                                         | 默认值                   |
|------------------------|--------------------------------------------------------------|--------------------------|
| `video_path`          | 输入视频文件的路径。                                           | `./template/...mp4`     |
| `audio_path`          | 输入音频文件的路径。                                           | `./template/...wav`     |
| `output_dir`          | 输出视频文件的目录。                                           | `./output_video`        |
| `deepspeech_model_path`| DeepSpeech 模型文件的路径。                                    | `../asserts/output_graph.pb` |
| `lawdnet_model_path`  | LawdNet 模型文件的路径。                                       | `../output/...pth`      |
| `BatchSize`           | 处理批次大小。                                                | `20`                     |
| `mouthsize`           | 嘴部模型的大小。                                              | `288`                    |
| `gpu_index`           | 使用的GPU索引号。                                             | `1`                      |
| `result_video_path`   | 生成视频文件的完整路径，由函数生成并返回。                | （由函数生成）           |

## 使用方法

1. 确保所有必要的文件路径正确无误。
2. 根据需要调整批次大小、嘴部模型大小和GPU索引。
3. 运行脚本，它将自动处理视频和音频，并生成输出文件。


## 查看wandb 训练日志
```https://wandb.ai/ai-zhua``

#### 有用的小工具
1. 压缩文件并显示进度 `./有用的脚本小工具/压缩文件.sh` 
2. 合并不同数据集的landmark字典 `/有用的脚本小工具/邓-处理大数据集.ipynb`
3. 音频处理的函数 `./audio_processing.py`


# 处理数据集的代码
[基于syncnet-python修改](https://github.com/iPaw-AI-LAB/syncnet)

功能：将视频转为25fps，并检测得到landmark.csv；若原视频就为25fps，在源文件夹。若原视频不为25fps，存放在目标文件夹

## 训练数据集准备
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
- 移动视频和对应csv的代码可以用`./syncnet/test_syncnet_dengjunli.ipynb` -
4. 运行`data_processing_正脸化.py`,得到crop后的图像
    - crop后的图像大小统一为 **416 320**，比例为1.3:1，通过`FaceAlign`类的参数`out_W`进行设置
    - 得到正脸化的crop landmark 字典 `./asserts/training_data/landmark_crop_dic.npy`
5. 若需要合并不同数据集的字典`landmark_crop_dic.npy`,可运行`./有用的脚本小工具/邓-处理大数据集.ipynb`
5. 重新生成完整的json文件 `python data_processing_正脸化.py --generate_training_json` 

- 若要更换数据集，请将数据集命名为training_data



## 常用的训练命令/测试命令
[【腾讯文档】Lawdnet常用命令-dengjunli](https://docs.qq.com/doc/DTENSWFlpTVFvSkhn)


### 实验记录+demo
[飞书云文档实验记录和demo](https://y5ucgsxnni.feishu.cn/docx/QSxadxHp0o6bgLxiiEbc0nvNnZd)

### 论文地址
[overleaf 需审批](https://www.overleaf.com/read/vkhhnxrvwbdw#3778eb)

### 宣传网页
[LawDNet主页](https://cucdengjunli.github.io/idf/)


### 评价指标
[数据集评测指标代码](https://gitee.com/dengjunli/evaluation_wav2lip)

### 资源
[原仓库DINet](https://fuxivirtualhuman.github.io/pdf/AAAI2023_FaceDubbing.pdf)

### 工程化文档
[将其工程化的记录](https://kdocs.cn/l/cinrYOJIsclj)

### 鸣谢

[codebase-DINet](https://fuxivirtualhuman.github.io/pdf/AAAI2023_FaceDubbing.pdf)









