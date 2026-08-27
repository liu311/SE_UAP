import numpy as np
import soundfile as sf


def to_mono(wav_data):
    """Convert multi-channel audio to mono while preserving 1-D audio."""
    wav_data = np.asarray(wav_data, dtype=np.float32)
    if wav_data.ndim == 1:
        return wav_data
    if wav_data.ndim == 2:
        return wav_data.mean(axis=1)
    raise ValueError(f"Expected 1-D or 2-D audio, got shape {wav_data.shape}")


def max_abs_normalize(wav_data, eps=1e-8):
    """Normalize a waveform to the valid SincNet input range [-1, 1]."""
    wav_data = to_mono(wav_data)
    norm_factor = float(np.max(np.abs(wav_data)))
    if norm_factor < eps:
        return wav_data.astype(np.float32), 1.0
    return (wav_data / norm_factor).astype(np.float32), norm_factor


def load_audio(path, expected_fs=16000, normalize=True):
    """
    Load a mono waveform and optionally max-abs normalize it.

    The manuscript states that TIMIT and LibriSpeech are used at 16 kHz. Resampling is not
    performed silently here, because changing the speaker-model input rate without a known
    preprocessing recipe would make evaluation less reproducible.
    """
    wav_data, fs = sf.read(path)
    if expected_fs is not None and fs != expected_fs:
        raise ValueError(f"{path} has sample rate {fs}, expected {expected_fs}")
    if normalize:
        wav_data, norm_factor = max_abs_normalize(wav_data)
    else:
        wav_data = to_mono(wav_data).astype(np.float32)
        norm_factor = 1.0
    return wav_data, fs, norm_factor


def repeat_to_length(perturbation, length):
    """Repeat or truncate a fixed-length UAP to match an utterance length."""
    if len(perturbation) <= 0:
        raise ValueError("Perturbation must be non-empty")
    repeats = int(np.ceil(length / len(perturbation)))
    return np.tile(perturbation, repeats)[:length].astype(np.float32)
