import tensorflow as tf
from tensorflow_tts.inference import AutoProcessor, TFAutoModel, AutoConfig
import numpy as np
import soundfile as sf

# Load processor and Tacotron2 model
processor = AutoProcessor.from_pretrained("tensorspeech/tts-tacotron2-ljspeech-en")
tacotron2 = TFAutoModel.from_pretrained("tensorspeech/tts-tacotron2-ljspeech-en")

def synthesize_text(text):
    input_ids = processor.text_to_sequence(text)
    mel_outputs, _, _ = tacotron2.inference(input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0))
    mel_spectrogram = mel_outputs.numpy().squeeze()
    return mel_spectrogram

def save_waveform(mel_spectrogram, filename):
    waveform = processor.inv_mel_spectrogram(mel_spectrogram)
    sf.write(filename, waveform, 22050, 'PCM_16')

if __name__ == "__main__":
    text = "Hello, this is a machine learning based voice synthesizer."
    mel_spectrogram = synthesize_text(text)
    save_waveform(mel_spectrogram, "output.wav")
    print("Synthesized speech saved to output.wav")
