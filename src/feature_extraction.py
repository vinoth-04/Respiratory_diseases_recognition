"""
Feature extraction module for respiratory sound analysis.
"""

import numpy as np
import librosa
from typing import Tuple


def extract_mfcc_features(audio_path: str, n_mfcc: int = 13, sr: int = 22050) -> np.ndarray:
    """
    Extract MFCC features from audio file.
    
    Args:
        audio_path: Path to audio file
        n_mfcc: Number of MFCC coefficients
        sr: Sample rate
        
    Returns:
        Feature vector of shape (52,) - mean, std, min, max for 13 MFCCs
    """
    audio, _ = librosa.load(audio_path, sr=sr)
    
    # Pad or trim to 4 seconds
    if len(audio) < sr * 4:
        audio = np.pad(audio, (0, sr * 4 - len(audio)), mode='constant')
    else:
        audio = audio[:sr * 4]
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    # Compute statistics
    mean = np.mean(mfccs, axis=1)
    std = np.std(mfccs, axis=1)
    min_ = np.min(mfccs, axis=1)
    max_ = np.max(mfccs, axis=1)
    
    return np.concatenate([mean, std, min_, max_])


def extract_mfcc_sequence(audio_path: str, n_mfcc: int = 13, max_len: int = 100, sr: int = 22050) -> np.ndarray:
    """
    Extract MFCC sequence for 1D CNN.
    
    Args:
        audio_path: Path to audio file
        n_mfcc: Number of MFCC coefficients
        max_len: Maximum sequence length
        sr: Sample rate
        
    Returns:
        MFCC sequence of shape (max_len, n_mfcc)
    """
    audio, _ = librosa.load(audio_path, sr=sr)
    
    if len(audio) < sr * 4:
        audio = np.pad(audio, (0, sr * 4 - len(audio)), mode='constant')
    else:
        audio = audio[:sr * 4]
    
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc).T
    
    if mfccs.shape[0] > max_len:
        mfccs = mfccs[:max_len]
    else:
        mfccs = np.pad(mfccs, ((0, max_len - mfccs.shape[0]), (0, 0)), mode='constant')
    
    return mfccs


def extract_mel_spectrogram(audio_path: str, n_mels: int = 128, n_time: int = 128, sr: int = 22050) -> np.ndarray:
    """
    Extract Mel-spectrogram for 2D CNN.
    
    Args:
        audio_path: Path to audio file
        n_mels: Number of mel bands
        n_time: Number of time steps
        sr: Sample rate
        
    Returns:
        Mel-spectrogram of shape (128, 128, 1)
    """
    audio, _ = librosa.load(audio_path, sr=sr)
    
    if len(audio) < sr * 4:
        audio = np.pad(audio, (0, sr * 4 - len(audio)), mode='constant')
    else:
        audio = audio[:sr * 4]
    
    n_fft = 2048
    hop_length = 512
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    if mel_spec_db.shape[1] > n_time:
        mel_spec_db = mel_spec_db[:, :n_time]
    else:
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, n_time - mel_spec_db.shape[1])), mode='constant')
    
    return mel_spec_db.reshape(128, 128, 1)


def extract_mel_spectrogram_from_audio(audio: np.ndarray, sr: int, n_mels: int = 128, n_time: int = 128) -> np.ndarray:
    """
    Extract Mel-spectrogram from audio array (useful for augmented audio).
    
    Args:
        audio: Audio time series
        sr: Sample rate
        n_mels: Number of mel bands
        n_time: Number of time steps
        
    Returns:
        Mel-spectrogram of shape (128, 128, 1)
    """
    if len(audio) < sr * 4:
        audio = np.pad(audio, (0, sr * 4 - len(audio)), mode='constant')
    else:
        audio = audio[:sr * 4]
    
    n_fft = 2048
    hop_length = 512
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    if mel_spec_db.shape[1] > n_time:
        mel_spec_db = mel_spec_db[:, :n_time]
    else:
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, n_time - mel_spec_db.shape[1])), mode='constant')
    
    return mel_spec_db.reshape(128, 128, 1)
