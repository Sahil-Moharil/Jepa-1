import torch
from torch.utils.data import Dataset
import torchaudio
import os

class RealAudioDataset(Dataset):
    def __init__(self, audio_dir, frames=512, feature_dim=80, sample_rate=16000, extensions=(".wav", ".flac")):
        self.audio_dir = audio_dir
        self.frames = frames
        self.feature_dim = feature_dim
        self.sample_rate = sample_rate
        self.extensions = extensions

        # Recursively collect all audio files
        self.files = []
        for root, _, files in os.walk(audio_dir):
            for file in files:
                if file.lower().endswith(self.extensions):
                    self.files.append(os.path.join(root, file))

        if len(self.files) == 0:
            raise ValueError(f"No audio files found in {audio_dir} with extensions {self.extensions}")

        # Mel spectrogram + log scale
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            hop_length=160,
            n_mels=feature_dim
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load waveform
        waveform, sr = torchaudio.load(self.files[idx])

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Compute log-mel spectrogram
        mel = self.mel_transform(waveform)   # [1, n_mels, time]
        mel = self.db_transform(mel)
        mel = mel.squeeze(0).transpose(0, 1) # [time, n_mels]

        # Normalize per sample
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        # Pad or truncate
        if mel.shape[0] < self.frames:
            pad_len = self.frames - mel.shape[0]
            mel = torch.cat([mel, torch.zeros(pad_len, self.feature_dim)], dim=0)
        else:
            mel = mel[:self.frames]

        # Random mask for A-JEPA (25% masked)
        mask = torch.rand(self.frames) < 0.25
        x_context = mel.clone()
        x_context[mask] = 0

        x_target = mel.clone()
        x_target[~mask] = 0  # keep only masked positions

        return x_context, x_target
