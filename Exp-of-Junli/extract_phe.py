from moviepy.editor import VideoFileClip
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import soundfile as sf
import os
import numpy as np

# Load model and processor
processor = Wav2Vec2Processor.from_pretrained("./wav2vec2-xls-r-300m-timit-phoneme")
model = Wav2Vec2ForCTC.from_pretrained("./wav2vec2-xls-r-300m-timit-phoneme")

def extract_audio_from_video(video_path, audio_path):
    """
    Extracts audio from a video file.
    """
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)

def get_phoneme_timestamps(audio_path, target_phoneme):
    """
    Process audio to find phonemes and return timestamps of target phoneme.
    """
    # Read and process the input
    audio_input, sample_rate = sf.read(audio_path)
    inputs = processor(audio_input, sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

    predicted_ids = torch.argmax(logits, axis=-1)
    predicted_phonemes = processor.batch_decode(predicted_ids)[0].split()

    # Logic to extract timestamps for target_phoneme goes here
    # Placeholder for the actual implementation
    timestamps = [0]  # This should be replaced with actual timestamps of target_phoneme

    return timestamps

def timestamps_to_frames(timestamps, frame_rate):
    """
    Convert audio timestamps to corresponding frame numbers in the video.
    """
    return [int(timestamp * frame_rate) for timestamp in timestamps]

def main(video_folder, target_phoneme):
    videos = os.listdir(video_folder)
    for video in videos:
        video_path = os.path.join(video_folder, video)
        audio_path = "temp_audio.wav"  # Temporary audio file path
        extract_audio_from_video(video_path, audio_path)
        
        timestamps = get_phoneme_timestamps(audio_path, target_phoneme)
        
        video_clip = VideoFileClip(video_path)
        frame_rate = video_clip.fps
        frame_numbers = timestamps_to_frames(timestamps, frame_rate)
        
        print(f"Frames containing '{target_phoneme}' in {video}: {frame_numbers}")

        # Cleanup temporary audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)

if __name__ == "__main__":
    video_folder = "./output_video"  # Update this path to your video folder
    target_phoneme = "ɛ"  # Update this to your target phoneme
    main(video_folder, target_phoneme)
