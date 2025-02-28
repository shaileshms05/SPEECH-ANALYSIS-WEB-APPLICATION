import torch
import torchaudio
import torchaudio.datasets
import librosa
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder

# ✅ Load RAVDESS Dataset from torchaudio
dataset = torchaudio.datasets.SPEECHCOMMANDS(root="./data", download=True)

# 🎤 Function to Extract MFCC Features
def extract_features(waveform, sample_rate, max_len=40):
    mfccs = librosa.feature.mfcc(y=waveform.numpy().squeeze(), sr=sample_rate, n_mfcc=40)
    pad_width = max_len - mfccs.shape[1]  # Padding
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfccs

# 🎯 Custom Dataset Class
class SpeechEmotionDataset(Dataset):
    def __init__(self, dataset):
        self.data = []
        self.labels = []
        self.label_encoder = LabelEncoder()

        for waveform, sample_rate, label, *_ in dataset:
            mfcc_features = extract_features(waveform, sample_rate)
            self.data.append(mfcc_features)
            self.labels.append(label)

        self.labels = self.label_encoder.fit_transform(self.labels)  # Encode labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0), torch.tensor(self.labels[idx], dtype=torch.long)

# ✅ Load Dataset
dataset = SpeechEmotionDataset(dataset)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 🧠 Define CNN Model for Emotion Recognition
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 10 * 10, 128)  # Adjust input size accordingly
        self.fc2 = nn.Linear(128, len(set(dataset.labels)))  # Number of emotion classes
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ✅ Initialize Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 🔥 Training Loop
def train(model, loader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# ✅ Train Model
train(model, train_loader, optimizer, criterion, epochs=10)

# 🎯 Evaluate Model
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")

# ✅ Evaluate on Test Data
evaluate(model, test_loader)

# ✅ Save Model
torch.save(model.state_dict(), "emotion_cnn.pth")
print("Model saved successfully!")
