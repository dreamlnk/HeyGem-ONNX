"""Wav2Lip mel-spectrogram extraction.

Matches the official Wav2Lip preprocessing:
  16kHz audio -> mel spectrogram (80 bands, hop=200, n_fft=800) -> log10
  Output: (T, 80) float32, values in approximate range [-5, 0].
"""
import numpy as np
import librosa


def mel_spectrogram(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Compute log-mel spectrogram matching Wav2Lip training.

    Args:
        audio: (N,) float32 audio samples at 16kHz
        sr: sample rate (default 16000)
    Returns:
        mel: (80, T) float32 log-mel spectrogram
    """
    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float64),
        sr=sr,
        n_fft=800,
        hop_length=200,
        win_length=800,
        n_mels=80,
        fmin=0,
        fmax=None,
        power=1.0,
    )
    # Wav2Lip uses log10, not log (ln) and not dB
    mel = np.log10(np.clip(mel, a_min=1e-5, a_max=None))
    return mel.astype(np.float32)


def get_wav2lip_mel_input(mel: np.ndarray, syncnet_step: int = 16) -> np.ndarray:
    """Extract the last N mel frames as model input.

    Args:
        mel: (80, T) float32 log-mel spectrogram (librosa format: n_mels x time)
        syncnet_step: number of mel frames per video frame (default 16)
    Returns:
        mel_input: (1, 1, 80, 16) float32 ready for Wav2Lip model
    """
    # mel is (80, T) from librosa.feature.melspectrogram
    T = mel.shape[1]
    mel_slice = np.zeros((80, syncnet_step), dtype=np.float32)
    if T >= syncnet_step:
        mel_slice = mel[:, T - syncnet_step:T].copy()
    else:
        mel_slice[:, syncnet_step - T:] = mel

    return mel_slice[np.newaxis, np.newaxis, ...]  # (1, 1, 80, 16)
