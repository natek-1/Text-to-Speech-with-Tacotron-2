
import torch
import torchaudio

def load_wav(path_to_audio, sr=22050):
    '''
    Loads an audio file and returns the waveform as a PyTorch tensor.
    
    Inputs:
        path_to_audio (str): Path to the audio file.
        sr (int): Target sampling rate.
    Outputs:
        waveform (torch.Tensor): Loaded audio waveform of dimention (timestep).
    '''
    
    audio, og_sr = torchaudio.load(path_to_audio)
    if og_sr != sr:
        audio = torchaudio.functional.resample(audio, orig_freq=og_sr, new_freq=sr)
    
    return audio.squeeze(0)

def amp_to_db(x, min_db=-100):
    '''
    Converts amplitude to decibels.
    
    Inputs:
        x (torch.Tensor): Input amplitude tensor.
        min_db (float): Minimum decibel value.
    Outputs:
        db (torch.Tensor): Decibel tensor.
    '''
    clip_val = 10 ** (min_db / 20)
    x = torch.clamp(x, min=clip_val)
    return 20 * torch.log10(x)


def db_to_amp(x):
    '''
    Converts decibels to amplitude.
    
    Inputs:
        x (torch.Tensor): Input decibel tensor.
    Outputs:
        amp (torch.Tensor): Amplitude tensor.
    '''
    return 10 ** (x / 20)

def normalize(x, min_db=-100., max_abs_val=4):
    '''
    Normalizes a decibel tensor to the range [-4, 4] allows for stability during training.
    
    Inputs:
        x (torch.Tensor): Input decibel tensor.
        min_db (float): Minimum decibel value.
        max_abs_val (int): Maximum absolute value for normalization.
    Outputs:
        normalized (torch.Tensor): Normalized tensor.
    '''
    x = (x - min_db) / -min_db
    x = 2*max_abs_val * x - max_abs_val
    return torch.clip(x, min=-max_abs_val, max=max_abs_val)

def denormalize(x, min_db=-100., max_abs_val=4):
    '''
    Denormalizes a tensor from the range [-4, 4] back to decibels.
    
    Inputs:
        x (torch.Tensor): Input normalized tensor.
        min_db (float): Minimum decibel value.
        max_abs_val (int): Maximum absolute value for normalization.
    Outputs:
        denormalized (torch.Tensor): Denormalized decibel tensor.
    '''
    x = torch.clip(x, min= -max_abs_val, max= max_abs_val)
    x = (x + max_abs_val) / (2 * max_abs_val)
    x = x * -min_db + min_db
    return x

def build_padding_mask(lengths):
    '''
    Builds Padding Mask for Tensor of provided length
    Inputs:
        lengths (torch.Tensor): Tensor containing the length of encoded tokens
    Ouputs:
        mask (torch.Tensor): Tensor containing the padding mask
    '''
    B = lengths.shape[0]
    T = torch.max(lengths).item()
    
    mask = torch.zeros(B, T)
    for idx, length in enumerate(lengths):
        mask[idx,length:] = 1
    
    return mask.bool()
    