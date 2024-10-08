# LawDNet 2024



### BUG

- [ ] 处理数据集的时候，Jinpeng 数据集的 crop face 的长度和 DeepSpeech 长度不一致
- [ ] 音频的 DeepSpeech 帧数总比视频帧多 3 帧
- [ ] 对 loss 的权重敏感，尤其是 syncnet_loss，导致震荡严重，但是不影响训练结果
- [ ] 人头占画面全部时有网格现象，可能是 affine grid 的 align corner搞的鬼，有待实验解决
- [ ] DP2 版本的cuda 只能使用0号，待改进

### 改进

- [x] 根据音频的长度来确定拆帧的数量，推理时
- [ ] 用 DeepSpeech PyTorch 来做数据集
- [ ] 用hubert来提取音频特征，以适应中文场景
- [ ] 训练数据很重要，目前HDTF训练效果保证稳定，其它数据集不一定，应该是deepspeech音频特征不适应中文


## 环境配置
~~先安装tensorflow_gpu = 1.15, 模型whl在[百度网盘](https://pan.baidu.com/s/1bNJT409wNlcJgkiAGHcONA?pwd=ipaw)， 提取码：ipaw~~

~~下载完成后，存放于`./tensorflow_gpu-1.15.0-cp37-cp37m-manylinux2010_x86_64.whl`~~

~~**先安装tensorflow, 再安装torch**~~

```bash
conda create --name LawDNet python=3.9
conda activate LawDNet
~~pip install tensorflow_gpu-1.15.0-cp37-cp37m-manylinux2010_x86_64.whl~~
conda install -c conda-forge ffmpeg
pip install -r requirements.txt
```

~~- 需要用到的模型: 换脸 ，deepspeech(tensorflow), vgg 等。 [百度网盘](https://pan.baidu.com/s/1bNJT409wNlcJgkiAGHcONA?pwd=ipaw)， 提取码：ipaw~~

~~- output_graph.pb : pretrained deepspeech model of tensorflow 1.15，用于提取音频特征，放在```./asserts/```~~
- syncnet_256mouth.pth: 用于训练时计算唇形同步损失，放在```./asserts/```

---

#### 有用的小工具： `./有用的脚本小工具/`
1. 压缩文件并显示进度 `sh ./压缩文件.sh` 
2. 合并不同数据集的landmark字典 `邓-处理大数据集.ipynb`
3. 合并两组数据集`sh ./移动数据集.sh`，新处理的数据集和老数据集合并时用到

---
# 数据集筛选，对齐，提取landmark
用**[数据集筛选，对齐，提取landmark](https://github.com/iPaw-AI-LAB/syncnet)代码**处理视频 
- 基于 [syncnet-python](https://github.com/joonson/syncnet_python) 重构

**功能：** 将视频转为25fps，并检测得到 `landmark.csv` 文件。若原视频即为25fps，保存在源文件夹 (`root_folder`)；若原视频不为25fps，存放在目标文件夹 (`output_folder`)。

**最佳实践：**
- 通常将 `root_folder` 源文件夹和 `output_folder` 目标文件夹设置为相同的文件夹。处理完之后，将视频移动到 `training_data` 文件夹。
- 使用 `./syncnet/move_dataset.ipynb` 可以只移动同时存在 `.mp4` 和 `.csv` 文件的数据，从而起到过滤的作用。

## 训练数据集准备
将音视频对齐的数据集进行预处理，并正脸化。

#### 数据处理脚本


- 将视频移动到本实验的 `./assert/training_data/split_video_25fps`
- 将视频对应的csv移动到本实验代码的`./assert/training_data/split_video_25fps_landmark_openface`
   - 移动视频和对应csv的代码可以用`./syncnet/move_dataset.ipynb`, 仅移动视频和csv同时存在的数据
- 运行`data_processing_正脸化.py`, 进行训练数据集准备
    - crop后的图像大小统一为 **416 320**，比例为1.3:1，通过`FaceAlign`类的参数`out_W`进行设置
      - 得到正脸化的crop landmark 字典 `./asserts/training_data/landmark_crop_dic.npy`
      - 若需要合并不同数据集的landmark字典`landmark_crop_dic.npy`,可运行`./有用的脚本小工具/邓-处理大数据集.ipynb`
      - 重新生成完整的json文件 `python data_processing_正脸化.py --generate_training_json` 

- 训练数据集存放于`./training_data`



### 一句命令处理训练数据
```sh
python data_processing_正脸化.py --extract_video_frame_multithreading && \
python data_processing_正脸化.py --extract_audio && \
python data_processing_正脸化.py --extract_deep_speech_multithreading && \
python data_processing_正脸化.py --crop_face_multithreading && \
python data_processing_正脸化.py --generate_training_json
```

### 处理训练数据步骤

- `--extract_video_frame`：启用从源视频中提取视频帧。:
  ```sh
  python data_processing_正脸化.py --extract_video_frame
  ```
  - `--extract_video_frame_multithreading`：启用多线程提取视频帧，以提高效率。:
    ```sh
    python data_processing_正脸化.py --extract_video_frame_multithreading
    ```

- `--extract_audio`：启用从源视频中提取音频。
  ```sh
  python data_processing_正脸化.py --extract_audio
  ```

- `--extract_deep_speech`：启用从音频文件中提取DeepSpeech特征。
  ```sh
  python data_processing_正脸化.py --extract_deep_speech
  ```

- `--crop_face`：启用根据landmark裁剪人脸。:
  ```sh
  python data_processing_正脸化.py --crop_face
  ```
  - `--crop_face_multithreading`：启用多线程裁剪人脸，以提高效率。:
    ```sh
    python data_processing_正脸化.py --crop_face_multithreading
    ```

- `--generate_training_json`：启用生成训练JSON文件。
  ```sh
  python data_processing_正脸化.py --generate_training_json
  ```


# 代码使用说明

## 训练 
采用 coarse to fine 的训练策略，每个阶段有自己的config文件，位于```./config/experiment ``` 

基础配置文件是```./config/config.py```

### DDP 并行训练方式 - 快
打开```train_sequence_distributed.sh``` 修改NAME(实验名称)
直接执行脚本：```sh train_sequence_distributed.sh```
```python
sh train_sequence_distributed.sh
# 对应参数在config.py 和 train_sequence_distributed.sh 中修改
```


| 训练配置项名称               | 描述                                                         | 示例值                   |
|--------------------------|--------------------------------------------------------------|--------------------------|
| `augment_num`            | 数据增强的次数。                                               | `3`                      |
| `mouth_region_size`      | 嘴部区域的大小。                                              | `288`（或`256`）        |
| `batch_size`             | 训练时每个批次的样本数量。                                   | `8`                      |
| `pretrained_frame_DINet_path` | 上一轮coarse训练的模型路径。                               | `./output/..._epoch_119.pth` |
| `result_path`            | 结果和模型保存的路径。                                       | `./output/.../clip_training_256` |
| `pretrained_syncnet_path`| SyncNet模型的预训练路径。                                    | `./asserts/syncnet_256mouth.pth` |
| `non_decay`              | 学习率开始衰减之前的epoch数量。                                    | `300`                    |
| `decay`                  | 学习率衰减的的epoch数量。                                           | `300`                    |
| `start_epoch`            | 训练开始的epoch，用于断点续练或从零开始训练。             | `1`                      |
| `resume`                 | 是否从断点恢复训练。                                          | `False` 或 `True`       |


如果您的此轮训练中断了，希望从之前的训练断点继续训练，您可以设置如下：

```python
{
    'resume': True,
    'start_epoch': 断点训练的epoch
}
```

- 模型保存位置： ```./output/training_model_weight/NAME(实验名称)```


### DP并行方式训练-慢
直接执行脚本：```sh ./train_sequence.sh```

### [wandb 查看训练日志](https://wandb.ai/ai-zhua)

### 训练容易出错的地方
1. 请仔细检查各个训练阶段的config文件
2. 务必保证coarse to fine 训练，直接训第四步得到的嘴部是模糊的，模型没有办法一步登天
3. 请检查torchrun的端口号是否被占用
4. 若训练效果嘴巴模糊，则增加最后一步的epoch到200以上
5. decay epoch 和 non decay epoch 数量必须相同，否则会导致学习率为负数
6. 训练数据很重要，数据集越高清，音频无杂音最好。目前HDTF训练效果保证稳定，其它数据集不一定，应该是deepspeech音频特征不适应中文，不要盲目增加中文数据集，不好的数据集甚至有😡反效果
7. 本项目取消了BatchNorm层，训练时请把batchsize调小，本实验设为1
8. 使用deepspeech2 pytorch训练时，注意deepspeech2 pytorch无法处理静音数据，遇到静音视频请手动删除
   1. [deepspeech pytorch 2](https://github.com/iPaw-AI-LAB/deepspeech_pytorch)
   2. 提取没有经过softmax的deepspeech特征: `python extract_deepspeech_pytorch2.py`


## 测试：
下载预训练模型，测试视频：[百度网盘](https://pan.baidu.com/s/1FFINqyyz2to96_-A7QhhHA?pwd=ipaw) 提取码: ipaw 

#### 新版速度快

优化中间过程，提前保存特定人的 landmark，frame， renference_image_tensor到本地，使用时直接读取

如果是新送入的视频，第一次执行会保存以上内容到本地，第二次之后则直接从本地读取，以加快执行速度

4090 上，视频推理速度和音频 1:1，拼接成视频的过程占 9s 左右


#### LawDNet 视频生成server使用说明

```
cd ./Exp-of-Junli # 进入实验文件夹
```

本系统包含三个主要Python文件:server_faster.py、inference_function_DP2.py和client_faster.py。以下是每个文件的功能和使用方法:

##### 1. inference_function_DP2.py

这个文件包含了视频生成的核心功能和辅助函数。

功能:
- 提供视频帧处理、特征提取、模型推理等功能
- 包含视频生成的主要逻辑

使用方法:
- 这个文件通常不需要直接运行,它被server_faster.py调用
- 如果需要单独测试,可以修改文件末尾的main函数,然后运行:
  ```
  python inference_function_DP2.py
  ```

##### 2. server_faster.py

这是系统的服务器端程序。

功能:
- 初始化并加载所需的模型和数据
- 提供HTTP接口,接收客户端的音频文件并生成对应的视频

使用方法:
1. 修改文件中的相关路径,如模型路径、输出目录等
2. 运行命令:
   ```
   python server_faster.py
   ```
3. 等待所有模型和模特数据加载完成后，服务器将在本地`5000`端口启动

##### 3. client_faster.py

这是客户端程序,用于向服务器`server_faster.py`发送音频文件并接收生成的视频。

功能:
- 向服务器发送音频文件
- 接收并保存服务器生成的视频

使用方法:
1. 确保服务器(server_faster.py)已经运行
2. 修改audio_file_path变量,指向您要使用的音频文件
3. 如果需要,修改server_url变量以匹配服务器地址
4. 运行命令:
   ```
   python client_faster.py
   ```
5. 程序将发送音频文件,并将接收到的视频保存为`./generated_video.mp4`

注意事项:

1. 请确保已安装所有必要的Python库,如Flask、torch、numpy等
2. 运行服务器前,请检查并修改所有相关的文件路径
3. 生成的视频质量和处理时间可能会因硬件配置而异
4. 建议在GPU环境下运行以获得更好的性能






#### 旧版速度慢

~~修改``` inference_function.py ``` 里面的参数：~~
修改``` inference_function_dp2.py ``` 里面的参数：

- **video_path**: 输入视频文件的路径。
- **audio_path**: 输入音频文件的路径。
- ~~**deepspeech_model_path**: DeepSpeech 模型文件的路径（`output_graph.pb`）。~~
- **lawdnet_model_path**: 预训练 LawDNet 模型文件的路径（`checkpoint_epoch_170.pth`）。
- **output_dir**: 保存输出视频的目录。
- **BatchSize**: 处理视频帧的批处理大小。
- **mouthsize**: 处理的嘴部区域大小（例如 `288`）。
- **gpu_index**: 使用的GPU索引（设置为 `-1` 表示使用CPU）。
- **output_name**: 输出视频文件的自定义名称。
- **dp2_model_path**: deepspeech2 pytorch 模型 `LibriSpeech_Pretrained_v3.ckpt`

``` sh
cd ./Exp-of-Junli/ 
python inference_function_DP2.py
```

```./Exp-of-Junli/optimized-prediction-deng.ipynb # 方便单步调试看中间结果```

```./Exp-of-Junli/inference_function.py # 修改模型路径，数字人视频，音频```

```./Exp-of-Junli/server_LawDNet.py # 提供server服务```

```./Exp-of-Junli/client_LawDNet.py # 提供发送音频的 client 服务```

## server服务
生成视频并发送到指定端口

下载预训练模型，测试视频：[百度网盘](https://pan.baidu.com/s/1FFINqyyz2to96_-A7QhhHA?pwd=ipaw) 提取码: ipaw 

```sh
cd Exp-of-Junli;
python server_LawDNet.py # 执行 server
python client_LawDNet.py # 发送音频文件到指定端口，触发 server
```

## 参数列表

| 参数名称               | 描述                                                         | 默认值                   |
|------------------------|--------------------------------------------------------------|--------------------------|
| `video_path`          | 模版视频文件的路径。                                           | `./template/...mp4`     |
| `audio_path`          | 输入音频文件的路径。（由chattts提供）                            | `./template/...wav`     |
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

## 资源

[常用的训练命令/测试命令](https://docs.qq.com/doc/DTENSWFlpTVFvSkhn) - 腾讯文档

[实验记录和demo](https://y5ucgsxnni.feishu.cn/docx/QSxadxHp0o6bgLxiiEbc0nvNnZd) - 飞书云文档

[论文地址（需审批）](https://www.overleaf.com/read/vkhhnxrvwbdw#3778eb) - Overleaf

[LawDNet主页](https://cucdengjunli.github.io/idf/) - 宣传网页

[数据集评测指标代码](https://gitee.com/dengjunli/evaluation_wav2lip) - 评价指标

[参考论文](https://fuxivirtualhuman.github.io/pdf/AAAI2023_FaceDubbing.pdf) - 资源

[将其工程化的记录](https://kdocs.cn/l/cinrYOJIsclj) - 工程化文档

[codebase-DINet](https://github.com/MRzzm/DINet) - 鸣谢









