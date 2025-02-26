// Upload and analyze audio
function uploadAudio() {
    let fileInput = document.getElementById('audioFile');
    if (!fileInput.files.length) {
        alert("Please select an audio file.");
        return;
    }

    let formData = new FormData();
    formData.append("audio", fileInput.files[0]);

    fetch("/analyze_speech", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').innerHTML = `
            <p><strong>Transcription:</strong> ${data.transcription}</p>
            <p><strong>Emotion:</strong> ${data.emotion}</p>
            <p><strong>Sentiment:</strong> Polarity: ${data.feedback.polarity}, Subjectivity: ${data.feedback.subjectivity}</p>
        `;
    })
    .catch(error => console.error('Error:', error));
}

// Rapid Fire Analogies
function startAnalogy() {
    fetch("/start_analogy")
    .then(response => response.json())
    .then(data => {
        document.getElementById('analogyPrompt').textContent = data.prompt;
        setTimeout(() => fetch("/evaluate_analogy").then(res => res.json()).then(data => {
            document.getElementById('analogyFeedback').textContent = `Score: ${data.score} - ${data.feedback}`;
        }), 5000);
    });
}

// Triple Step Exercise
function startTripleStep() {
    fetch("/start_triple_step")
    .then(response => response.json())
    .then(data => {
        document.getElementById('tripleStepTopic').textContent = data.topic;
        setTimeout(() => fetch("/evaluate_triple_step").then(res => res.json()).then(data => {
            document.getElementById('tripleStepFeedback').textContent = `Score: ${data.score} - ${data.feedback}`;
        }), 10000);
    });
}

// Conductor Exercise
function startConductor() {
    fetch("/start_conductor")
    .then(response => response.json())
    .then(data => {
        document.getElementById('conductorFeedback').textContent = `Feedback: ${data.feedback}`;
    });
}
