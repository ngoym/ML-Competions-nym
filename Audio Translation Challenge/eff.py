import torch
import librosa
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import copy
#import pdb
import timm
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

root_dir = '/home/mutonkol/ML/Audio/train/files'
df = pd.read_csv('csv/Train.csv')
label_dict = {
    'up': 0,
    'down': 1,
    'left': 2,
    'right': 3,
    'go': 4,
    'stop': 5,
    'yes': 6,
    'no': 7
}

test_label_dict = {v: k for k, v in label_dict.items()}

TRAIN = False
if TRAIN:
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

SEED = 42
BATCH_SIZE = 48 if TRAIN else 64
LR = 0.001
NUM_EPOCHS = 20
INPUT_SIZE = 200000
MODEL_NAME = 'resnet10t' #'resnet18' resnet10t

SR = 48000           # Sample rate # 44100
n_fft = 400          # 25ms window length
hop_length = 160     # 10ms hop length
n_mels = 64          # Number of Mel bands
fmin = 20            # Minimum frequency (Hz) 85
fmax = 20000          # Maximum frequency (Hz) 8000


class AudioDataset(Dataset):
    def __init__(self, root_dir, df, mode='train'):
        self.df = df
        self.root_dir = root_dir
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get file path and load waveform
        file_path = os.path.join(self.root_dir, self.df["audio_filepath"].iloc[idx])
        waveform, sample_rate = librosa.load(file_path, sr=SR)

        # Truncate or pad the waveform to the desired input size
        if len(waveform) > INPUT_SIZE:
            waveform = waveform[:INPUT_SIZE]
        elif len(waveform) < INPUT_SIZE:
            waveform = np.pad(waveform, (0, INPUT_SIZE - len(waveform)), mode='constant')

        # Convert waveform to Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=waveform,
            sr=SR,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax
        )

        # Convert the Mel spectrogram to log scale (decibels)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Normalize Mel spectrogram (optional, depending on your model requirements)
        mel_spectrogram = (mel_spectrogram - mel_spectrogram.mean()) / mel_spectrogram.std()
        # Apply frequency or time masking with a probability of 0.3
        if np.random.rand() < 0.4:
            # Apply frequency masking
            f_mask_length = np.random.randint(3, 15)
            f_mask_start = np.random.randint(0, mel_spectrogram.shape[0] - f_mask_length)
            mel_spectrogram[f_mask_start:f_mask_start + f_mask_length, :] = 0
            # Apply frequency masking
            t_mask_length = np.random.randint(3, 15)
            t_mask_start = np.random.randint(0, mel_spectrogram.shape[1] - t_mask_length)
            mel_spectrogram[:, t_mask_start:t_mask_start + t_mask_length] = 0

        # Convert to PyTorch tensor (required for model input)
        mel_spectrogram = torch.tensor(mel_spectrogram, dtype=torch.float32)

        # If not in train mode, return just the Mel spectrogram
        if self.mode != 'train':
            return mel_spectrogram

        label = label_dict[self.df["class"].iloc[idx]]
        return mel_spectrogram, label

class AudioDatasetTest(Dataset):
    def __init__(self, root_dir, df):
        self.df = df
        self.root_dir = root_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.df["id"].iloc[idx]+".wav")
        waveform, sample_rate = librosa.load(file_path, sr=SR)
        # Truncate or pad the waveform to the desired input size
        if len(waveform) > INPUT_SIZE:
            waveform = waveform[:INPUT_SIZE]
        elif len(waveform) < INPUT_SIZE:
            waveform = np.pad(waveform, (0, INPUT_SIZE - len(waveform)), mode='constant')

        # Convert waveform to Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=waveform,
            sr=SR,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax
        )

        # Convert the Mel spectrogram to log scale (decibels)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Normalize Mel spectrogram (optional, depending on your model requirements)
        mel_spectrogram = (mel_spectrogram - mel_spectrogram.mean()) / mel_spectrogram.std()

        # Convert to PyTorch tensor (required for model input)
        mel_spectrogram = torch.tensor(mel_spectrogram, dtype=torch.float32)
        return mel_spectrogram

from sklearn.model_selection import train_test_split
from tqdm import tqdm
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['class'], random_state=SEED)

train_dataset = AudioDataset(root_dir=root_dir, df=train_df)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = AudioDataset(root_dir=root_dir, df=test_df)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

def log(message):
    print(message)
    with open('log.txt', 'a') as f:
        f.write(f"{message}\n")

class AudioClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super(AudioClassifier, self).__init__()
        self.model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes, in_chans=1)

    def forward(self, x):
        x = self.model(x)
        return x

# Create an instance of the model
model = AudioClassifier()
model = model.to(DEVICE)

if TRAIN:
    num_epochs = NUM_EPOCHS
    best_acc = 0.0

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        model.train()
        average_loss = 0.0
        for batch in tqdm(train_dataloader):
            waveforms, labels = batch
            waveforms = waveforms.to(DEVICE)
            labels = labels.to(DEVICE)
            #pdb.set_trace()
            # Forward pass
            optimizer.zero_grad()
            outputs = model(waveforms.unsqueeze(1))  # Add a channel dimension to the waveforms
            loss = criterion(outputs, labels)
            average_loss += loss.item()
            # Backward and optimize
            loss.backward()
            optimizer.step()
        scheduler.step()

        average_loss /= len(train_dataloader)

        model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                waveforms, labels = batch
                waveforms = waveforms.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(waveforms.unsqueeze(1))
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

        accuracy = total_correct / total_samples
        log(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss}")
        log(f"Validation Accuracy: {accuracy}")
        if accuracy >= best_acc:
            log(f"Saving best model with accuracy: {accuracy} - previous best accuracy: {best_acc}")
            best_acc = accuracy
            model_fp16 = copy.deepcopy(model)
            model_fp16 = model_fp16.half()
            torch.save(model_fp16.state_dict(), 'best_audio_classifier.pth')
    model_fp16 = copy.deepcopy(model)
    model_fp16 = model_fp16.half()
    torch.save(model_fp16.state_dict(), 'final_audio_classifier.pth')

else:
    model.load_state_dict(torch.load('final_audio_classifier.pth'))
    model.eval()
    test = pd.read_csv('csv/SampleSubmission_1.csv')
    test_dataset = AudioDatasetTest(root_dir=root_dir, df=test)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            waveforms = batch
            outputs = model(waveforms.unsqueeze(1))
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.numpy())
    predictions = [test_label_dict[p] for p in predictions.copy()]
    #pdb.set_trace()
    test['class'] = predictions
    test.to_csv(f'submission-{MODEL_NAME}-augs-best.csv', index=False)