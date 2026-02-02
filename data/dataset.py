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

        # List all files with allowed extensions
        self.files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) 
                      if f.endswith(self.extensions)]
        if len(self.files) == 0:
            raise ValueError(f"No audio files found in {audio_dir} with extensions {self.extensions}")

        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            hop_length=160,
            n_mels=feature_dim
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load waveform
        waveform, sr = torchaudio.load(self.files[idx])

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Compute mel spectrogram
        mel = self.mel_transform(waveform)  # [n_mels, time_frames]
        mel = mel.squeeze(0)                # remove channel dim if exists
        mel = mel.transpose(0, 1)           # [time_frames, n_mels]

        # Pad or truncate to fixed number of frames
        if mel.shape[0] < self.frames:
            pad_len = self.frames - mel.shape[0]
            mel = torch.cat([mel, torch.zeros(pad_len, self.feature_dim)], dim=0)
        else:
            mel = mel[:self.frames]

        # Random mask for A-JEPA
        mask = torch.rand(self.frames) > 0.5
        x_context = mel.clone()
        x_context[mask] = 0  # Masked positions set to 0

        x_target = mel.clone()
        x_target[~mask] = 0  # Only masked positions kept

        return x_context, x_target
