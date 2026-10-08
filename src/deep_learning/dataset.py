"""PyTorch dataset for the shared GTZAN song-level manifest."""
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from .mel_spectrogram import load_and_convert

GENRE_TO_INDEX = {"blues":0,"classical":1,"country":2,"disco":3,"hiphop":4,"jazz":5,"metal":6,"pop":7,"reggae":8,"rock":9}

class GTZANMelDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, audio_root: str | Path, sample_rate=22050, duration=2.0, n_mels=64, n_fft=512, hop_length=256, random_windows=True):
        self.manifest = manifest.reset_index(drop=True)
        self.audio_root = Path(audio_root)
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.random_windows = random_windows

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        row = self.manifest.iloc[index]
        path = self.audio_root / row["relative_path"]
        max_offset = max(0.0, 30.0 - self.duration)
        # Training uses a random window; validation/test use the center window for reproducibility.
        offset = float(torch.rand(1).item() * max_offset) if self.random_windows else max_offset / 2.0
        mel = load_and_convert(path, self.sample_rate, self.duration, offset, self.n_mels, self.n_fft, self.hop_length)
        x = torch.from_numpy(mel).unsqueeze(0)
        y = torch.tensor(GENRE_TO_INDEX[row["genre"]], dtype=torch.long)
        return x, y
