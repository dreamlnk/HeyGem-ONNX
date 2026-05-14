"""Wav2Lip mel-spectrogram extraction — matches official preprocessing exactly.

Official pipeline (audio.py + hparams.py):
  16kHz audio -> preemphasis(0.97) -> STFT -> |·| -> mel -> dB -> normalize to [-4, 4]
"""
import numpy as np
import librosa
import librosa.filters
from scipy import signal

# ---------------------------------------------------------------------------
# Hyperparameters matching official hparams.py
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
N_FFT = 800
HOP_SIZE = 200
WIN_SIZE = 800
NUM_MELS = 80
FMIN = 55
FMAX = 7600
PREEMPHASIZE = True
PREEMPHASIS_COEF = 0.97
SIGNAL_NORMALIZATION = True
SYMMETRIC_MELS = True
MAX_ABS_VALUE = 4.0
MIN_LEVEL_DB = -100
REF_LEVEL_DB = 20

_mel_basis = None


def _build_mel_basis():
    return librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=NUM_MELS,
                               fmin=FMIN, fmax=FMAX)


def _stft(y):
    return librosa.stft(y=y, n_fft=N_FFT, hop_length=HOP_SIZE, win_length=WIN_SIZE)


def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)


def _amp_to_db(x):
    min_level = np.exp(MIN_LEVEL_DB / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _normalize(S):
    """Symmetric normalization to [-MAX_ABS_VALUE, MAX_ABS_VALUE]."""
    if SYMMETRIC_MELS:
        return np.clip((2 * MAX_ABS_VALUE) * ((S - MIN_LEVEL_DB) / (-MIN_LEVEL_DB))
                       - MAX_ABS_VALUE, -MAX_ABS_VALUE, MAX_ABS_VALUE)
    else:
        return np.clip(MAX_ABS_VALUE * ((S - MIN_LEVEL_DB) / (-MIN_LEVEL_DB)),
                       0, MAX_ABS_VALUE)


def mel_spectrogram(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Compute normalized mel spectrogram matching official Wav2Lip.

    Args:
        audio: (N,) float32 audio samples at 16kHz
        sr: sample rate (default 16000)
    Returns:
        mel: (80, T) float32 mel spectrogram, values in [-4, 4]
    """
    # Preemphasis
    if PREEMPHASIZE:
        audio = signal.lfilter([1, -PREEMPHASIS_COEF], [1], audio)

    # STFT -> mel -> dB
    D = _stft(audio.astype(np.float64))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - REF_LEVEL_DB

    if SIGNAL_NORMALIZATION:
        S = _normalize(S)

    return S.astype(np.float32)


def get_wav2lip_mel_input(mel: np.ndarray, syncnet_step: int = 16) -> np.ndarray:
    """Extract the last N mel frames as model input.

    Args:
        mel: (80, T) float32 mel spectrogram
        syncnet_step: number of mel frames per video frame (default 16)
    Returns:
        mel_input: (1, 1, 80, 16) float32 ready for Wav2Lip model
    """
    T = mel.shape[1]
    mel_slice = np.zeros((80, syncnet_step), dtype=np.float32)
    if T >= syncnet_step:
        mel_slice = mel[:, T - syncnet_step:T].copy()
    else:
        mel_slice[:, syncnet_step - T:] = mel

    return mel_slice[np.newaxis, np.newaxis, ...]  # (1, 1, 80, 16)
