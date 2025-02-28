import os
import torch
import torchaudio
import librosa
import numpy as np
from flask import Flask, request, jsonify, render_template
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from textblob import TextBlob
import torch.nn as nn
import torch.nn.functional as F

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Load Pretrained Wav2Vec2 Model for Transcription
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
wav2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# ✅ Load Trained CNN Model for Emotion Detection (Ensure correct feature sizes)
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=33):  # Match trained model classes
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 10 * 12, 128)  # Adjusted feature size
        self.fc2 = nn.Linear(128, num_classes)  # Match trained model's output

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emotion_model = EmotionCNN(num_classes=33).to(device)
emotion_model.load_state_dict(torch.load("emotion_cnn.pth", map_location=device))
emotion_model.eval()

# 🎤 **Transcription Function**
def transcribe_audio(audio_path):
    waveform, rate = torchaudio.load(audio_path)
    if rate != 16000:
        transform = torchaudio.transforms.Resample(orig_freq=rate, new_freq=16000)
        waveform = transform(waveform)

    inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        logits = wav2vec_model(input_values=inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0]

# 🎭 **Emotion Detection Function**
def extract_mfcc(audio_path, max_pad_len=50):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
    pad_width = max_pad_len - mfcc.shape[1]
    if pad_width > 0:
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]

    return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 40, 50)

def analyze_emotion(audio_path):
    features = extract_mfcc(audio_path).to(device)
    with torch.no_grad():
        prediction = torch.argmax(emotion_model(features), dim=1).item()
    emotions = {
    0: "Neutral",
    1: "Calm",
    2: "Happy",
    3: "Sad",
    4: "Angry",
    5: "Fearful",
    6: "Disgust",
    7: "Surprised"
}
    
    return emotions.get(prediction, "Unknown Emotion")

    
 # Labels must match model classes
  # Labels must match model classes
   

# 😃 **Sentiment Analysis Function**
def analyze_sentiment(text):
    blob = TextBlob(text)
    return {
        "polarity": round(blob.sentiment.polarity, 2),
        "subjectivity": round(blob.sentiment.subjectivity, 2)
    }

# 🎙️ **Index Route for Upload Form**
@app.route("/")
def index():
    return render_template("index.html")

# 🎙️ **RESTful API Endpoint**
@app.route("/analyze_speech", methods=["POST"])
def analyze_speech():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    audio_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    audio_file.save(audio_path)

    try:
        transcription = transcribe_audio(audio_path)
        emotion = analyze_emotion(audio_path)
        sentiment = analyze_sentiment(transcription)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "transcription": transcription,
        "emotion": emotion,
        "sentiment": sentiment
    })

# ✅ **Run Flask Server**
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
