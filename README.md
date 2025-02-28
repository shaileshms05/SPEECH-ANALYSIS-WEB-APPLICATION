Speech Analysis Web Application

This project is a Flask-based API that processes audio files to extract speech transcription, emotion analysis, and sentiment analysis using Wav2Vec2, CNN-based emotion detection, and TextBlob for sentiment analysis.

🚀 Features

🎤 Speech-to-Text Transcription using Facebook's Wav2Vec2 model.

🎭 Emotion Detection using a pre-trained CNN model.

😃 Sentiment Analysis to determine polarity and subjectivity.

📡 REST API to handle audio file uploads and return analyzed results.

🛠️ Installation

Clone the repository:

git clone https://github.com/your-repo/speech-analysis-flask.git
cd speech-analysis-flask

Create a virtual environment and activate it:

python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install dependencies:

pip install -r requirements.txt

Download the pre-trained models:

Wav2Vec2 (Automatically downloads on first use)

Ensure emotion_cnn.pth (trained model) is available in the project directory.

🚀 Running the Application

python app.py

The API will be accessible at: http://127.0.0.1:5000/

🎙️ Usage

1️⃣ Web Interface

Open http://127.0.0.1:5000/ in your browser and upload an audio file.

2️⃣ API Endpoint

POST /analyze_speech

Request:

curl -X POST -F "audio=@sample.wav" http://127.0.0.1:5000/analyze_speech

Response:

{
  "transcription": "Hello, how are you?",
  "emotion": "Happy",
  "sentiment": {
    "polarity": 0.5,
    "subjectivity": 0.6
  }
}

📁 Project Structure

📂 speech-analysis-flask/
├── 📂 uploads/         # Directory to store uploaded files
├── 📜 app.py          # Main Flask application
├── 📜 emotion_cnn.pth # Pre-trained CNN model for emotion detection
├── 📜 requirements.txt # Required dependencies
├── 📜 README.md       # Project documentation

🏗️ Technologies Used

Python (Flask, Torch, Librosa, TextBlob)

Deep Learning Models (Wav2Vec2, CNN for emotion recognition)

API Development (Flask, JSON responses)

📌 Future Enhancements

🎙️ Improve transcription accuracy with fine-tuned models.

📊 Implement a frontend dashboard for visualization.

🔊 Support more languages for transcription.

🤝 Contributing

Feel free to contribute by creating pull requests or reporting issues.

📜 License

This project is licensed under the MIT License.

🚀 Happy Coding! 🎧
