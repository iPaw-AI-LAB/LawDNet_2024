
import numpy as np
import warnings
import resampy
from scipy.io import wavfile
from python_speech_features import mfcc
import tensorflow as tf
import time
import os

import threading
import queue


class DeepSpeech():
    def __init__(self,model_path):
        self.graph, self.logits_ph, self.input_node_ph, self.input_lengths_ph \
            = self._prepare_deepspeech_net(model_path)
        self.target_sample_rate = 16000

    def _prepare_deepspeech_net(self,deepspeech_pb_path):
        with tf.io.gfile.GFile(deepspeech_pb_path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        graph = tf.compat.v1.get_default_graph()
        tf.import_graph_def(graph_def, name="deepspeech")
        logits_ph = graph.get_tensor_by_name("deepspeech/logits:0") # delete deepspeech if you want to use this py as a package
        input_node_ph = graph.get_tensor_by_name("deepspeech/input_node:0") # delete deepspeech 
        input_lengths_ph = graph.get_tensor_by_name("deepspeech/input_lengths:0") # delete deepspeech 
        # ###########~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # logits_ph = graph.get_tensor_by_name("logits:0") # delete deepspeech if you want to use this py as a package
        # input_node_ph = graph.get_tensor_by_name("input_node:0") # delete deepspeech 
        # input_lengths_ph = graph.get_tensor_by_name("input_lengths:0") # delete deepspeech 

        return graph, logits_ph, input_node_ph, input_lengths_ph

    def conv_audio_to_deepspeech_input_vector(self,audio,
                                              sample_rate,
                                              num_cepstrum,
                                              num_context):
        # Get mfcc coefficients:
        features = mfcc(
            signal=audio,
            samplerate=sample_rate,
            numcep=num_cepstrum)
        
        # print("看这个MFCC feature 是否一样: ")
        # import pdb
        # pdb.set_trace() 

        # We only keep every second feature (BiRNN stride = 2):
        features = features[::2]

        # One stride per time step in the input:
        num_strides = len(features)

        # Add empty initial and final contexts:
        empty_context = np.zeros((num_context, num_cepstrum), dtype=features.dtype)
        features = np.concatenate((empty_context, features, empty_context))

        # Create a view into the array with overlapping strides of size
        # numcontext (past) + 1 (present) + numcontext (future):
        window_size = 2 * num_context + 1
        train_inputs = np.lib.stride_tricks.as_strided(
            features,
            shape=(num_strides, window_size, num_cepstrum),
            strides=(features.strides[0],
                     features.strides[0], features.strides[1]),
            writeable=False)

        # Flatten the second and third dimensions:
        train_inputs = np.reshape(train_inputs, [num_strides, -1])

        train_inputs = np.copy(train_inputs)
        train_inputs = (train_inputs - np.mean(train_inputs)) / \
                       np.std(train_inputs)

        return train_inputs

    def compute_audio_feature(self,audio_path):   # original
        os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

        audio_sample_rate, audio = wavfile.read(audio_path)
        if audio.ndim != 1:
            warnings.warn(
                "Audio has multiple channels, the first channel is used")
            audio = audio[:, 0]
        if audio_sample_rate != self.target_sample_rate:
            resampled_audio = resampy.resample(
                x=audio.astype(np.float64),
                sr_orig=audio_sample_rate,
                sr_new=self.target_sample_rate)
        else:
            resampled_audio = audio.astype(np.float64)

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1
        with tf.compat.v1.Session(graph=self.graph, config=config) as sess:
        
        # with tf.compat.v1.Session(graph=self.graph) as sess:
            input_vector = self.conv_audio_to_deepspeech_input_vector(
                audio=resampled_audio.astype(np.int16),
                sample_rate=self.target_sample_rate,
                num_cepstrum=26,
                num_context=9)
            # 91 494

            # print("观察deepspeech的输出")
            # import pdb; pdb.set_trace() 
            
            network_output = sess.run(
                    self.logits_ph,
                    feed_dict={
                        self.input_node_ph: input_vector[np.newaxis, ...],
                        self.input_lengths_ph: [input_vector.shape[0]]})
            # （视频帧数*2-1， 1， 29
            print("看这个network_output的shape: ",network_output.shape)
            
            ds_features = network_output[::2,0,:] # 音频帧数是50fps，两帧取一次，使其与视频帧数25fps相等，
            print("看这个ds_features的shape: ",ds_features.shape)
        return ds_features

    # def compute_audio_feature(self,audio_path):   
    #     os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

    #     audio_sample_rate, audio = wavfile.read(audio_path)
    #     if audio.ndim != 1:
    #         warnings.warn(
    #             "Audio has multiple channels, the first channel is used")
    #         audio = audio[:, 0]
    #     if audio_sample_rate != self.target_sample_rate:
    #         resampled_audio = resampy.resample(
    #             x=audio.astype(np.float64),
    #             sr_orig=audio_sample_rate,
    #             sr_new=self.target_sample_rate)
    #     else:
    #         resampled_audio = audio.astype(np.float64)

    #     input_vector = self.conv_audio_to_deepspeech_input_vector(
    #         audio=resampled_audio.astype(np.int16),
    #         sample_rate=self.target_sample_rate,
    #         num_cepstrum=26,
    #         num_context=9)

    #     from multiprocessing import Process, Queue

    #     def run_in_process(input_vector, result_queue):
    #         config = tf.compat.v1.ConfigProto()
    #         config.gpu_options.allow_growth = True
    #         with tf.compat.v1.Session(graph=self.graph, config=config) as sess:
    #             result = sess.run(self.logits_ph, feed_dict={self.input_node_ph: input_vector[np.newaxis, ...], self.input_lengths_ph: [input_vector.shape[0]]})
    #             result_queue.put(result)

    #     input_vector1 = input_vector[:input_vector.shape[0]//4]
    #     input_vector2 = input_vector[input_vector.shape[0]//4:input_vector.shape[0]//2]
    #     input_vector3 = input_vector[input_vector.shape[0]//2:input_vector.shape[0]*3//4]
    #     input_vector4 = input_vector[input_vector.shape[0]*3//4:]

    #     result_queue = Queue()

    #     process1 = Process(target=run_in_process, args=(input_vector1, result_queue))
    #     process2 = Process(target=run_in_process, args=(input_vector2, result_queue))
    #     process3 = Process(target=run_in_process, args=(input_vector3, result_queue))
    #     process4 = Process(target=run_in_process, args=(input_vector4, result_queue))

    #     process1.start()
    #     process2.start()
    #     process3.start()
    #     process4.start()

    #     process1.join()
    #     process2.join()
    #     process3.join()
    #     process4.join()

    #     result1 = result_queue.get()
    #     result2 = result_queue.get()
    #     result3 = result_queue.get()
    #     result4 = result_queue.get()

    #     network_output = np.concatenate((result1, result2, result3, result4), axis=0)

    #     ds_features = network_output[::2,0,:]

    #     return ds_features



if __name__ == '__main__':
    audio_path = './Exp-of-Junli/template/英文tts.wav'
    model_path = './asserts/output_graph.pb'
    DSModel = DeepSpeech(model_path)

    start_time = time.time()
    ds_feature = DSModel.compute_audio_feature(audio_path)
    print(ds_feature)
    end_time = time.time()
    print(f"Running time: {end_time - start_time} seconds")

