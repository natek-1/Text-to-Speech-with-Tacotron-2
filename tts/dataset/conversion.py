import numpy as np

import torch

import librosa

from tts.dataset.utils import load_wav, normalize, denormalize, amp_to_db, db_to_amp


class AudioMelConversions:
    '''
    A class for converting audio waveforms to mel-spectrograms and vice versa.
    Attributes:
        sr (int): Sampling rate.
        n_fft (int): Number of FFT components.
        hop_length (int): Hop length for STFT.
        win_length (int): Window length for STFT.
        num_mels (int): Number of mel bands.
        fmin (int): Minimum frequency for mel filter bank.
        fmax (int): Maximum frequency for mel filter bank.
        min_db (float): Minimum decibel value for normalization.
        max_abs_val (float): Maximum absolute value for normalization.
    '''
    
    def __init__(self, sr=22050, n_fft=1024, hop_length=256, win_length=1024,
                 num_mels=80, fmin=0, fmax=8000, center=False, min_db=-100., max_abs_val=4):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.fmin = fmin
        self.n_mels = num_mels
        self.fmax = fmax
        self.center = center
        self.min_db = min_db
        self.max_abs_val = max_abs_val
        
        # freq_bins = n_fft/2 + 1
        self.spec2mel = self._get_spec2mel_proj() # shape (num_mels, freq_bins)
        self.mel2spec = torch.linalg.pinv(self.spec2mel)
    
    def _get_spec2mel_proj(self):
        '''
        Computes the projection matrix from spectrogram to mel-spectrogram.
        
        Outputs:
            spec2mel_proj (np.ndarray): Spectrogram to mel-spectrogram projection matrix.
        '''
        mel = librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels,
                                  fmin=self.fmin, fmax=self.fmax)
        return torch.from_numpy(mel)
    
    def audio2mel(self, audio, do_norm=False):
        '''
        Converts an audio waveform to a mel-spectrogram.
        
        Inputs:
            audio (torch.Tensor): Input audio waveform.
            do_norm (bool): Whether to normalize the mel-spectrogram.
        Outputs:
            mel_spectrogram (torch.Tensor): Mel-spectrogram.
        '''
        
        if not isinstance(audio, torch.Tensor):
            audio = torch.FloatTensor(audio)
        
        stft = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length,
                          win_length=self.win_length, window=torch.hann_window(self.win_length, periodic=True).to(audio.device),
                          center=self.center, pad_mode='reflect', normalized=False, onesided=True,
                          return_complex=True)
        spectrogram = torch.abs(stft)  # (freq_bins, time_frames)
        mel_spectrogram = torch.matmul(self.spec2mel.to(audio.device), spectrogram)  # (num_mels, time_frames)
        mel_spectrogram_db = amp_to_db(mel_spectrogram)
        
        if do_norm:
            mel_spectrogram_db = normalize(mel_spectrogram_db, min_db=self.min_db, max_abs_val=self.max_abs_val)
            
        return mel_spectrogram_db
    
    
    def mel2audio(self, mel_spectrogram, num_iters=60, do_denorm=False):
        '''
        Converts a mel-spectrogram back to an audio waveform using the Griffin-Lim algorithm.
        
        Inputs:
            mel_spectrogram (torch.Tensor): Input mel-spectrogram.
            num_iters (int): Number of iterations for Griffin-Lim.
            do_denorm (bool): Whether to denormalize the mel-spectrogram.
        Outputs:
            audio (torch.Tensor): Reconstructed audio waveform.
        '''
        
        if do_denorm:
            mel_spectrogram = denormalize(mel_spectrogram, min_db=self.min_db, max_abs_val=self.max_abs_val)
        
        mel_spectrogram_amp = db_to_amp(mel_spectrogram)
        spectrogram_approx = torch.matmul(self.mel2spec.to(mel_spectrogram.device), mel_spectrogram_amp).numpy()
        
        # Griffin-Lim algorithm
        audio = librosa.griffinlim(S=spectrogram_approx, n_iter=num_iters, hop_length=self.hop_length, win_length=self.win_length,
                                   n_fft=self.n_fft, window="hann")

        audio *= 32767 / max(0.01, np.max(np.abs(audio)))
        return audio.astype(np.int16)
    
    
            