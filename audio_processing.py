import os
import numpy as np
import torch    
from utils.deep_speech import DeepSpeech
import wave
from timeit import default_timer as timer
import sys
from tqdm import tqdm

def extract_deepspeech(driving_audio_path,deepspeech_model_path):
    '''
    input: 
        Str:driving_audio_path
        Str: deepspeech_model_path
    output: 
        Tensor:deepspeech_tensor_all
        Int: audio_frame_length 音频帧数
    '''
    print('extracting deepspeech feature from : {}'.format(driving_audio_path))
    if not os.path.exists(deepspeech_model_path):
        raise ('pls download pretrained model of deepspeech')
    DSModel = DeepSpeech(deepspeech_model_path)
    if not os.path.exists(driving_audio_path):
        raise ('wrong audio path :{}'.format(driving_audio_path))
    ds_feature = DSModel.compute_audio_feature(driving_audio_path)
    audio_frame_length = ds_feature.shape[0]
    ds_feature_padding = np.pad(ds_feature, ((2, 2), (0, 0)), mode='edge')


    deepspeech_tensor_all = torch.zeros(ds_feature.shape[0], ds_feature.shape[1], 5).cuda()
    for i in tqdm(range(ds_feature.shape[0]), desc='Processing Audio batches'):
        deepspeech_tensor = torch.from_numpy(ds_feature_padding[i : i+5, :]).permute(1, 0).float().cuda()
        deepspeech_tensor_all[i] = deepspeech_tensor

    # 保存
    # torch.save(deepspeech_tensor_all, driving_audio_path.replace('.wav', 'deepspeech_tensor_all.pt'))

    return deepspeech_tensor_all, audio_frame_length

def tts(driving_audio_path, deepspeech_model_path ):

    fin = wave.open(driving_audio_path, 'rb')
    fs_orig = fin.getframerate()
    # if fs_orig != desired_sample_rate:
    #     print('Warning: original sample rate ({}) is different than {}hz. Resampling might produce erratic speech recognition.'.format(fs_orig, desired_sample_rate), file=sys.stderr)
    #     fs_new, audio = convert_samplerate(args.audio, desired_sample_rate)
    # else:
    audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

    audio_length = fin.getnframes() * (1/fs_orig)
    fin.close()

    print('Running inference.', file=sys.stderr)
    inference_start = timer()
    # sphinx-doc: python_ref_inference_start
    # if args.extended:
    #     print(metadata_to_string(ds.sttWithMetadata(audio, 1).transcripts[0]))
    # elif args.json:
    #     print(metadata_json_output(ds.sttWithMetadata(audio, args.candidate_transcripts)))
    # else:
    #     print(ds.stt(audio))
    DSModel = DeepSpeech(deepspeech_model_path)
    tts_result=DSModel.stt(audio)
    print("tts_result:",tts_result)

    # sphinx-doc: python_ref_inference_stop
    inference_end = timer() - inference_start
    print('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length), file=sys.stderr)







if __name__ == "__main__":

    #########  1. 将音频转为deepspeech特征
    
    deepspeech_model_path = "./asserts/output_graph.pb"
    driving_audio_path = "./测试用/driving_audio_1.wav"
    # deepspeech_tensor_all, audio_frame_length = extract_deepspeech(driving_audio_path,deepspeech_model_path)


    #########  1. 将音频转为deepspeech特征


    #########  2. 根据音频读文字，检查能不能根据deepspeech读文字

    # tts(driving_audio_path, deepspeech_model_path )


    ############ 3. 官方
    import deepspeech

    model_path = deepspeech_model_path
    beam_width = 500
    lm_alpha = 0.75
    lm_beta = 1.85

    model = deepspeech.Model(model_path)
    # model.enableDecoderWithLM(lm_path, trie_path, lm_alpha, lm_beta)

    audio_path = driving_audio_path
    audio_data, sample_rate = deepspeech.read_audio_file(audio_path)

    text = model.stt(audio_data)
    print(text)
