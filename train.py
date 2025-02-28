import os
import torch
import torchaudio
import librosa
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ✅ Define CNN Model
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=8):  # 8 emotions
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)
        
        # ✅ Remove hardcoded input size for fc1
        self.flatten_size = None  # To be determined dynamically
        self.fc1 = None
        self.fc2 = None
       
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))

        # ✅ Print feature map size before flattening
        if self.flatten_size is None:  
            self.flatten_size = x.shape[1] * x.shape[2] * x.shape[3]
            self.fc1 = nn.Linear(self.flatten_size, 128).to(x.device)
            self.fc2 = nn.Linear(128, num_classes).to(x.device)

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ✅ Extract MFCC Features
def extract_mfcc(audio_path, max_pad_len=50):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        pad_width = max_pad_len - mfcc.shape[1]
        if pad_width > 0:
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]

        return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # (1, 40, 50)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# ✅ Custom Dataset Loader
class EmotionDataset(Dataset):
    def __init__(self, data_dir):
        self.data = []
        self.labels = []
        self.label_map = {emotion: idx for idx, emotion in enumerate(sorted(os.listdir(data_dir)))}

        for emotion in self.label_map.keys():
            emotion_folder = os.path.join(data_dir, emotion)
            for file in os.listdir(emotion_folder):
                if file.endswith(".wav"):
                    file_path = os.path.join(emotion_folder, file)
                    mfcc = extract_mfcc(file_path)
                    if mfcc is not None:
                        self.data.append(file_path)
                        self.labels.append(self.label_map[emotion])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mfcc_features = extract_mfcc(self.data[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mfcc_features, label

# ✅ Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = r"emotion_dataset"

if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Dataset directory '{data_dir}' not found!")

dataset = EmotionDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

num_classes = len(os.listdir(data_dir))  # Auto-detect number of classes
model = EmotionCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ Train Model
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        if inputs.shape[0] == 0:  # Skip empty batches
            continue
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

# ✅ Save Model
torch.save(model.state_dict(), "emotion_cnn.pth")
print("🎉 Model training complete! Saved as emotion_cnn.pth")
