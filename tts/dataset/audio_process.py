import librosa
import librosa.filters
import numpy as np
from scipy import signal

from scipy.io import wavfile


def load_wav(path, sample_rate=22050):
    return librosa.core.load(path, sr=sample_rate)[0]


def save_wav(wav, path, sample_rate=22050):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, sample_rate, wav.astype(np.int16))


def spectrogram(y, ref_level_db=20):
    D = _stft(_preemphasis(y))
    S = _amp_to_db(np.abs(D)) - ref_level_db
    return _normalize(S)


def inv_spectrogram(spectrogram, ref_level_db=20, power=1.5):
    S = _db_to_amp(_denormalize(spectrogram) + ref_level_db)  # Convert back to linear
    return _inv_preemphasis(_griffin_lim(S ** power))  # Reconstruct phase


def melspectrogram(y):
    D = _stft(_preemphasis(y))
    S = _amp_to_db(_linear_to_mel(np.abs(D)))
    return _normalize(S)


def inv_melspectrogram(melspectrogram, power=1.5):
    S = _mel_to_linear(_db_to_amp(_denormalize(melspectrogram)))  # Convert back to linear
    return _inv_preemphasis(_griffin_lim(S ** power))  # Reconstruct phase


# Based on https://github.com/librosa/librosa/issues/434
def _griffin_lim(S, num_iters=60):
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(complex)
    for i in range(num_iters):
        if i > 0:
            angles = np.exp(1j * np.angle(_stft(y)))
        y = _istft(S_complex * angles)
    return y


def _stft(y, n_fft=1024, hop_length=256):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)


def _istft(y, n_fft=1024, hop_length=256):
    return librosa.istft(y, hop_length=hop_length, n_fft=n_fft)


# Conversions:

_mel_basis = None
_inv_mel_basis = None


def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)


def _mel_to_linear(mel_spectrogram, floor_freq=0.01):
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(_build_mel_basis())
    return np.maximum(floor_freq, np.dot(_inv_mel_basis, mel_spectrogram))


def _build_mel_basis(sample_rate=22050, fft_size=1024, num_mels=80, min_freq=125, max_freq=7600):
    return librosa.filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=num_mels, fmin=min_freq, fmax=max_freq)


def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _preemphasis(x, alpha=0.97):
    return signal.lfilter([1, -alpha], [1], x)


def _inv_preemphasis(x, alpha=0.97):
    return signal.lfilter([1], [1, -alpha], x)


def _normalize(S, max_abs_value=4, min_level_db=-100):
    return np.clip(
        (2 * max_abs_value) * ((S - min_level_db) / (-min_level_db)) - max_abs_value,
        -max_abs_value, max_abs_value)  


def _denormalize(D, max_abs_value=4, min_level_db=-100):
    return (((np.clip(D, -max_abs_value,
                      max_abs_value) + max_abs_value) * -min_level_db / (
                     2 * max_abs_value))
            + min_level_db)


def get_hop_size():
    return 266
