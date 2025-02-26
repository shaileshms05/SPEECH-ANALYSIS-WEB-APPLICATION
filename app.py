import os
import torch
import torchaudio
from flask import Flask, request, jsonify, render_template
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from speechbrain.inference import SpeakerRecognition
from textblob import TextBlob
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Ensure 'uploads' directory exists
os.makedirs("uploads", exist_ok=True)

# ✅ Load Speech-to-Text Model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# ✅ Load Speaker Recognition Model
speaker_recognizer = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmpdir"
)

# 🎤 Speech-to-Text (Transcription)
def transcribe_audio(audio_path):
    waveform, rate = torchaudio.load(audio_path)

    # Resample if needed
    if rate != 16000:
        transform = torchaudio.transforms.Resample(orig_freq=rate, new_freq=16000)
        waveform = transform(waveform)

    inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        logits = model(input_values=inputs.input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

# 😃 Emotion Recognition
def extract_features(audio_path):
    """Extract features from audio"""
    waveform, sample_rate = librosa.load(audio_path)
    features = librosa.feature.mfcc(y=waveform, sr=sample_rate)
    return np.mean(features, axis=1)

def analyze_emotion(audio_path):
    """Analyze emotion from the audio file"""
    try:
        features = extract_features(audio_path)
        
        # Dummy trained model for demonstration
        model = RandomForestClassifier()
        X_train, X_test, y_train, y_test = train_test_split(
            np.random.rand(100, len(features)), 
            np.random.randint(0, 4, 100), 
            test_size=0.2
        )
        model.fit(X_train, y_train)
        
        prediction = model.predict([features])
        emotions = ["neutral", "happy", "sad", "angry"]
        emotion = emotions[prediction[0]] if prediction[0] < len(emotions) else "unknown"
        
        return emotion
    except Exception as e:
        return f"Emotion detection error: {str(e)}"

# 📝 Sentiment Analysis
def evaluate_response(transcription):
    """Analyze sentiment of the transcribed text."""
    blob = TextBlob(transcription)
    return {
        "polarity": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity
    }

# 🎙️ API Endpoint for Speech Analysis
@app.route("/analyze_speech", methods=["POST"])
def analyze_speech():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    audio_path = os.path.join("uploads", audio_file.filename)
    audio_file.save(audio_path)

    try:
        transcription = transcribe_audio(audio_path)
        emotion = analyze_emotion(audio_path)
        feedback = evaluate_response(transcription)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "transcription": transcription,
        "emotion": emotion,
        "feedback": feedback
    })

# 📄 Serve the Upload Page (index.html)
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    print(app.url_map)  # Debugging: Show available endpoints
    app.run(host='0.0.0.0', debug=True)
