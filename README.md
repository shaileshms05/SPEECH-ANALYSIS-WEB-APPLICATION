Speech Analysis Web Application

Overview

This Flask-based web application provides real-time speech analysis by integrating AI models for:

Speech-to-Text Conversion using Wav2Vec2

Emotion Detection using RandomForest Classifier

Sentiment Analysis using TextBlob

The app allows users to upload audio files and receive transcriptions, detected emotions, and sentiment insights in JSON format.

Features

✅ Speech-to-Text: Converts audio to text using Wav2Vec2 (Hugging Face Transformers).✅ Emotion Recognition: Detects emotions from speech using extracted MFCC features and a RandomForest classifier.✅ Sentiment Analysis: Evaluates the polarity and subjectivity of the transcribed text using TextBlob.✅ RESTful API: Allows users to send audio files via HTTP requests and receive structured JSON responses.✅ Scalability: Designed for deployment with Flask, making it easy to extend and integrate with other services.

Tech Stack

Backend: Flask (Python)

Speech-to-Text: Wav2Vec2 (Hugging Face Transformers, PyTorch, Torchaudio)

Emotion Recognition: Librosa (Feature Extraction), Scikit-learn (RandomForest Classifier)

Sentiment Analysis: TextBlob

API Integration: Flask RESTful API

Installation & Setup

1. Clone the Repository

git clone https://github.com/your-username/speech-analysis-webapp.git
cd speech-analysis-webapp

2. Create a Virtual Environment (Optional but Recommended)

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install Dependencies

pip install -r requirements.txt

4. Run the Application

python app.py

The application will be available at http://127.0.0.1:5000/

API Endpoints

1. Analyze Speech (POST /analyze_speech)

Uploads an audio file for transcription, emotion detection, and sentiment analysis.

Request:

curl -X POST -F "audio=@sample.wav" http://127.0.0.1:5000/analyze_speech

Response:

{
    "transcription": "Hello, how are you?",
    "emotion": "happy",
    "feedback": {
        "polarity": 0.5,
        "subjectivity": 0.6
    }
}

Future Enhancements

🚀 Real-time Speech Analysis: Integrate microphone-based real-time processing.🚀 Deep Learning for Emotion Detection: Replace RandomForest with LSTM/CNN-based models.🚀 Cloud Deployment: Deploy as a microservice using Docker & AWS.

License

This project is open-source and available under the MIT License.

Author
Shailesh MS

