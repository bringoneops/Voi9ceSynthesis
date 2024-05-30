from flask import Flask, request, jsonify, render_template, send_file
from gtts import gTTS
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/synthesize', methods=['POST'])
def synthesize():
    text = request.form.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    tts = gTTS(text)
    file_path = 'output.mp3'
    tts.save(file_path)

    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))