from flask import Flask, request, render_template, send_file
import os
from voice_synthesis import synthesize_text, save_waveform

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/synthesize', methods=['POST'])
def synthesize():
    text = request.form['text']
    mel_spectrogram = synthesize_text(text)
    output_file = "static/output.wav"
    save_waveform(mel_spectrogram, output_file)
    return send_file(output_file, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
