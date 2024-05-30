from flask import Flask, request, render_template, send_file
import tensorflow as tf
import numpy as np
from scipy.io.wavfile import write
import io
import tensorflow_tts
from tensorflow_tts.inference import AutoProcessor, TFAutoModel

app = Flask(__name__)

# Load pre-trained models
processor = AutoProcessor.from_pretrained("tensorspeech/tts-tacotron2-ljspeech-en")
tacotron2 = TFAutoModel.from_pretrained("tensorspeech/tts-tacotron2-ljspeech-en")
mb_melgan = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-ljspeech-en")


def synthesize_voice(text):
    # Text to mel spectrogram
    input_ids = processor.text_to_sequence(text)
    mel_outputs, _, _ = tacotron2.inference(
        input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
        input_lengths=tf.convert_to_tensor([len(input_ids)], tf.int32),
        speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
    )

    # Mel spectrogram to audio
    audio = mb_melgan.inference(mel_outputs)[0, :, 0]

    return 22050, audio.numpy()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form["text"]
        sample_rate, audio = synthesize_voice(text)
        audio_file = io.BytesIO()
        write(audio_file, sample_rate, audio)
        audio_file.seek(0)
        return send_file(
            audio_file,
            mimetype="audio/wav",
            as_attachment=True,
            attachment_filename="synthesized.wav",
        )
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
