import os
import numpy as np
import torch
from torch.amp import autocast
from deepspeech_pytorch.decoder import Decoder
from deepspeech_pytorch.loader.data_loader import ChunkSpectrogramParser, ChunkSpectrogramParserOfAudioData
from deepspeech_pytorch.model import DeepSpeech
from deepspeech_pytorch.utils import load_decoder, load_model
from deepspeech_pytorch.configs.inference_config import TranscribeConfig, LMConfig
from scipy.io import wavfile
import resampy
import warnings
import time
import json
from typing import List
from tqdm import tqdm

import hydra
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store(name="config", node=TranscribeConfig)

def check_and_resample_audio(audio_path: str, target_sample_rate: int = 16000):
    audio_sample_rate, audio = wavfile.read(audio_path)
    if audio.ndim != 1:
        warnings.warn("Audio has multiple channels, the first channel is used")
        audio = audio[:, 0]
    if audio_sample_rate != target_sample_rate:
        audio = resampy.resample(audio, sr_orig=audio_sample_rate, sr_new=target_sample_rate)
        wavfile.write(audio_path, target_sample_rate, audio.astype(np.int16))
    return audio_path

def decode_results(decoded_output: List):
    results = []
    for b in range(len(decoded_output)):
        for pi in range(len(decoded_output[b])):
            result = {'transcription': decoded_output[b][pi]}
            results.append(result)
    return results

def run_transcribe(audio_path: str,
                   spect_parser: ChunkSpectrogramParser,
                   model: DeepSpeech,
                   decoder: Decoder,
                   device: torch.device,
                   precision: int,
                   chunk_size_seconds: float):
    model.to(device)
    hs = None  # means that the initial RNN hidden states are set to zeros
    all_outs = []
    with torch.no_grad():
        for spect in spect_parser.parse_audio(audio_path, chunk_size_seconds):
            spect = spect.contiguous()
            spect = spect.view(1, 1, spect.size(0), spect.size(1))
            spect = spect.to(device)
            input_sizes = torch.IntTensor([spect.size(3)]).to(device).int()
            with autocast(device_type=str(device), enabled=(precision == 16)):
                out_before_softmax, out, output_sizes, hs = model(spect, input_sizes, hs)
                ###！！
                # print("需要没经过softmax的deepspeech输出")
            all_outs.append(out_before_softmax.cpu())
    all_outs = torch.cat(all_outs, axis=1)  # combine outputs of chunks in one tensor
    decoded_output, decoded_offsets = decoder.decode(all_outs)
    return all_outs, decoded_output

def run_transcribe_tensor(audio: np.ndarray,
                          spect_parser: ChunkSpectrogramParserOfAudioData,
                          model: DeepSpeech,
                          device: torch.device,
                          precision: int,
                          chunk_size_seconds: float):
    all_outs = []
    hs = None  # means that the initial RNN hidden states are set to zeros
    with torch.no_grad():
        for spect in spect_parser.parse_audio(audio, chunk_size_seconds):
            spect = spect.contiguous()
            spect = spect.view(1, 1, spect.size(0), spect.size(1))
            spect = spect.to(device)
            input_sizes = torch.IntTensor([spect.size(3)]).to(device).int()
            with autocast(device_type=str(device), enabled=(precision == 16)):
                out_before_softmax, out, output_sizes, hs = model(spect, input_sizes, hs)
            all_outs.append(out_before_softmax.cpu())
    all_outs = torch.cat(all_outs, axis=1)  # combine outputs of chunks in one tensor
    return all_outs

def process_audio_folder(audio_folder: str, model_path: str, output_folder: str, log_file_path: str, device: torch.device, precision: int):
    print('Processing audio folder:', audio_folder)
    
    # Load model and decoder
    model = load_model(device=device, model_path=model_path)
    
    cfg = TranscribeConfig()

    spect_parser = ChunkSpectrogramParser(audio_conf=model.spect_cfg, normalize=True)

    decoder = load_decoder(
        labels=model.labels,
        cfg=cfg.lm
    )

    model.eval()

    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    log_data = []

    wav_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]

    for file_name in tqdm(wav_files, desc="Processing audio files"):
        audio_path = os.path.join(audio_folder, file_name)
        output_txt_path = os.path.join(output_folder, file_name.replace(".wav", "_deepspeech_dp2.txt"))

        print('Processing audio:', audio_path)

        # Check and resample audio if necessary
        audio_path = check_and_resample_audio(audio_path, target_sample_rate=16000)

        # Transcribe audio
        ds_features_all, decoded_output = run_transcribe(
            audio_path=audio_path,
            spect_parser=spect_parser,
            model=model,
            decoder=decoder,
            device=device,
            precision=precision,
            chunk_size_seconds=-1
        )

        # Decode results
        transcriptions = decode_results(decoded_output)

        # Append results to log data
        log_data.append({"file": file_name, "results": {"output": transcriptions}})

        # Extract every second frame
        ds_features = ds_features_all[0][::2]  # Assuming ds_features shape is (1, frames, 29)
        # print("Extracted ds_features shape: ", ds_features.shape)

        # Save to txt file
        np.savetxt(output_txt_path, ds_features.numpy())
        print(f"Features saved to {output_txt_path}")

    # Save log data to file
    with open(log_file_path, 'w') as log_file:
        json.dump(log_data, log_file, indent=4, ensure_ascii=False)
    print(f"Log saved to {log_file_path}")


# 这个函数是用来处理音频的，包括预处理和特征提取
def transcribe_and_process_audio(audio_path: str, 
                                 model_path: str, 
                                 device: torch.device, 
                                 precision: int = 16,
                                 model: DeepSpeech = None,
                                 cfg: TranscribeConfig = None,
                                 spect_parser: ChunkSpectrogramParser = None,
                                 decoder: Decoder = None
                                 ):
    """
    input: 
        Str: audio_path - Path to the audio file
        Str: model_path - Path to the DeepSpeech model file
        torch.device: device - Device to run the model on
        Int: precision - Precision to use for model inference (default is 16)
    output: 
        Tensor: deepspeech_tensor_all - Processed DeepSpeech tensor
        Int: audio_frame_length - Length of the audio frame
    """
    # print('Transcribing and processing audio from:', audio_path)
    # print('Using DeepSpeech model at:', model_path)

    # if not os.path.exists(model_path):
    #     raise FileNotFoundError('Please download the pretrained model of DeepSpeech.')

    # if not os.path.exists(audio_path):
    #     raise FileNotFoundError('Wrong audio path: {}'.format(audio_path))

    # # Load model and decoder
    # start_time = time.time()
    # model = load_model(device=device, model_path=model_path)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(f"deepspeech Model loaded in {execution_time:.6f} seconds")
    
    # cfg = TranscribeConfig()
    # spect_parser = ChunkSpectrogramParser(audio_conf=model.spect_cfg, normalize=True)
    # decoder = load_decoder(
    #     labels=model.labels,
    #     cfg=cfg.lm
    # )

    # model.eval()

    # Check and resample audio if necessary
    audio_path = check_and_resample_audio(audio_path, target_sample_rate=16000)

    # Transcribe audio
    ds_features_all, decoded_output = run_transcribe(
        audio_path=audio_path,
        spect_parser=spect_parser,
        model=model,
        decoder=decoder,
        device=device,
        precision=precision,
        chunk_size_seconds=-1
    )

    ds_feature = ds_features_all[0][::2].numpy()


    # print('ds_feature:', ds_feature.shape)
    audio_frame_length = ds_feature.shape[0]
    
    # Post-processing
    ds_feature_padding = np.pad(ds_feature, ((2, 2), (0, 0)), mode='edge')
    # print('ds_feature_padding:', ds_feature_padding.shape)

    deepspeech_tensor_all = torch.zeros(ds_feature.shape[0], ds_feature.shape[1], 5)
    for i in tqdm(range(ds_feature.shape[0]), desc='Processing Audio batches'):
        deepspeech_tensor = torch.from_numpy(ds_feature_padding[i : i + 5, :]).permute(1, 0).float()
        deepspeech_tensor_all[i] = deepspeech_tensor

    # print('deepspeech_tensor_all:', deepspeech_tensor_all.shape)
    # print('audio_frame_length:', audio_frame_length)

    return deepspeech_tensor_all, audio_frame_length

def transcribe_audio_data(audio: torch.Tensor, model: DeepSpeech, precision: int = 16, device: torch.device = None):
    spect_parser = ChunkSpectrogramParserOfAudioData(audio_conf=model.spect_cfg, normalize=True)

    ds_features_all = run_transcribe_tensor(
        audio=audio,
        spect_parser=spect_parser,
        model=model,
        device=device, 
        precision=precision,
        chunk_size_seconds=-1)

    ds_feature = ds_features_all[0][::2].numpy()

    # print('ds_feature:', ds_feature.shape)
    audio_frame_length = ds_feature.shape[0]
    
    # Post-processing
    ds_feature_padding = np.pad(ds_feature, ((2, 2), (0, 0)), mode='edge')
    # print('ds_feature_padding:', ds_feature_padding.shape)

    deepspeech_tensor_all = torch.zeros(ds_feature.shape[0], ds_feature.shape[1], 5)
    for i in tqdm(range(ds_feature.shape[0]), desc='Processing Audio batches'):
        deepspeech_tensor = torch.from_numpy(ds_feature_padding[i : i + 5, :]).permute(1, 0).float()
        deepspeech_tensor_all[i] = deepspeech_tensor

    # print('deepspeech_tensor_all:', deepspeech_tensor_all.shape)
    # print('audio_frame_length:', audio_frame_length)

    return deepspeech_tensor_all, audio_frame_length

if __name__ == "__main__":
    # Manually specify the parameters
    # audio_folder = './test_data'

    # audio_folder = './test_dp2_audio'
    # model_path = './dp2_models/LibriSpeech_Pretrained_v3.ckpt'

    # # output_folder = './output_features'
    # output_folder = './dp2_results'
    # log_file_path = output_folder + '/deepspeech_log.json'
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # precision = 16

    # start_time = time.time()  # 记录开始时间

    # process_audio_folder(
    #     audio_folder=audio_folder,
    #     model_path=model_path,
    #     output_folder=output_folder,
    #     log_file_path=log_file_path,
    #     device=device,
    #     precision=precision
    # )

    # end_time = time.time()  # 记录结束时间
    # execution_time = end_time - start_time  # 计算执行时间

    # print(f"Execution time: {execution_time:.6f} seconds")  # 输出执行时间

############################################################################################################

    audio_file_path = './test_dp2_audio/taylor-20s.wav'
    model_path = './dp2_models/LibriSpeech_Pretrained_v3.ckpt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = 16

    deepspeech_tensor_all, audio_frame_length = transcribe_and_process_audio(
        audio_path=audio_file_path,
        model_path=model_path,
        device=device,
        precision=precision
    )

    print("Processed DeepSpeech tensor (deepspeech_tensor_all):", deepspeech_tensor_all.shape)
    print("Audio frame length (audio_frame_length):", audio_frame_length)
