"""Mel-spectrogram preprocessing utilities for GTZAN."""
from pathlib import Path
from typing import Tuple
import librosa
import numpy as np

def load_audio(audio_path: str | Path, sample_rate: int = 22050, duration: float = 2.0, offset: float = 0.0) -> Tuple[np.ndarray, int]:
    audio, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True, offset=offset, duration=duration)
    target_length = int(sample_rate * duration)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]
    return audio.astype(np.float32), sr

def audio_to_log_mel(audio: np.ndarray, sample_rate: int = 22050, n_mels: int = 64, n_fft: int = 512, hop_length: int = 256) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    return log_mel.astype(np.float32)

def load_and_convert(audio_path: str | Path, sample_rate: int = 22050, duration: float = 2.0, offset: float = 0.0, n_mels: int = 64, n_fft: int = 512, hop_length: int = 256) -> np.ndarray:
    audio, sr = load_audio(audio_path, sample_rate, duration, offset)
    return audio_to_log_mel(audio, sr, n_mels, n_fft, hop_length)
