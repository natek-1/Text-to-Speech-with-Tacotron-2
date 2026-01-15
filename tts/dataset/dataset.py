import pandas as pd
import numpy as np


import torch
from torch.utils.data import Dataset, DataLoader

import librosa

from tts.component.tokenizer import Tokenizer
from tts.dataset.conversion import AudioMelConversions
from tts.dataset.utils import load_wav, build_padding_mask





class TTSDataset(Dataset):
    '''
    Dataset for Text-to-Speech (TTS) training.
    init:
        path_to_metadata (str): path to metadata csv file
        sample_rate (int): sample rate of audio files
        n_fft (int): number of FFT points
        n_mels (int): number of mel bands
        hop_length (int): hop length for STFT
        win_length (int): window length for STFT
        fmin (int): minimum frequency for mel spectrogram
        fmax (int): maximum frequency for mel spectrogram
        center (bool): whether to center the STFT
        min_db (float): minimum decibel value for mel spectrogram
        max_abs_val (float): maximum absolute value for mel spectrogram
        normalize (bool): whether to normalize mel spectrogram
    '''
    
    def __init__(self, path_to_metadata, sample_rate=22050, n_fft=1024, n_mels=80, hop_length=256, win_length=1024,
                 fmin=0, fmax=8000, center=False, min_db=-100., max_abs_val=4, normalize=True):
        
        self.tokenizer = Tokenizer()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length
        self.fmin = fmin
        self.fmax = fmax
        self.center = center
        self.min_db = min_db
        self.max_abs_val = max_abs_val
        self.normalize = normalize
        
        self.metadata = pd.read_csv(path_to_metadata)
        self.transcript_lengths = [len(self.tokenizer.encode(text)) for text in
                                   self.metadata['normalized_text'].tolist()]
        self.audio_proc = AudioMelConversions(sr=self.sample_rate, n_fft=self.n_fft, num_mels=self.n_mels, hop_length=self.hop_length,
                                              win_length=self.win_length, fmin=self.fmin, fmax=self.fmax, center=self.center,
                                              min_db=self.min_db, max_abs_val=self.max_abs_val)
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        
        audio_path = self.metadata.loc[idx, 'file_path']
        transcript = self.metadata.loc[idx, 'normalized_text']
        audio = load_wav(audio_path, self.sample_rate)
        mel_spectrogram = self.audio_proc.audio2mel(audio, do_norm=self.normalize)
        
        return transcript, mel_spectrogram.squeeze(0)
    
    
def TTSCollator():
    '''
    Collate function for Text-to-Speech (TTS) training.
    '''
    
    tokenizer = Tokenizer()
    
    def _collate_fn(batch):
        '''
        Collate function for Text-to-Speech (TTS) training.
        inputs:
            batch (list): list of tuples (transcript, mel_spectrogram)
        outputs:
            text_padded (torch.Tensor): padded text sequences
            mel_padded (torch.Tensor): padded mel spectrogram sequences
            gate_padded (torch.Tensor): padded gate sequences
            masked_text_padded (torch.Tensor): padded masked text sequences
            masked_mel_padded (torch.Tensor): padded masked mel spectrogram sequences
        '''
        texts = [tokenizer.encode(b[0]) for b in batch]
        mels = [b[1] for b in batch]
        
        input_lengths = torch.LongTensor([text.shape[0] for text in texts])
        output_lengths = torch.LongTensor([mel.shape[1] for mel in mels]) # each mel spectogram is n_mels, num_timestep
        
        input_lengths, sorted_idx = input_lengths.sort(descending=True)
        texts = [texts[i] for i in sorted_idx]
        mels = [mels[i] for i in sorted_idx]
        output_lengths = output_lengths[sorted_idx]
        
        # much more efficient
        text_padded = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=tokenizer.pad_token_id)

        # pad mel sequences
        max_target_len = torch.max(output_lengths).item()
        num_mels = mels[0].shape[0] # n_mels 80 by default
        
        mel_padded = torch.zeros((len(mels), num_mels, max_target_len))
        gate_padded = torch.zeros(len(mels), max_target_len) # when we should stop making predictions
        
        for i, mel in enumerate(mels):
            t = mel.shape[1]
            mel_padded[i, :, :t] = mel
            gate_padded[i, t-1:] = 1
            
        mel_padded = mel_padded.transpose(1, 2) # batch_size, num_mels, timesteps
        
        return text_padded, input_lengths,mel_padded, gate_padded, build_padding_mask(input_lengths), build_padding_mask(output_lengths)
    
    
    return _collate_fn
            

class BatchSampler:
    '''
    Docstring for BatchSampler
    '''
    def __init__(self, dataset, batch_size, drop_last=False):
        self.sampler = torch.utils.data.SequentialSampler(dataset)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.random_batches = self._make_batches()

    def _make_batches(self):

        indices = [i for i in self.sampler]

        if self.drop_last:

            total_size = (len(indices) // self.batch_size) * self.batch_size
            indices = indices[:total_size]

        batches = [indices[i:i+self.batch_size] for i in range(0, len(indices), self.batch_size)]
        random_indices = torch.randperm(len(batches))
        return [batches[i] for i in random_indices]
    
    def __iter__(self):
        for batch in self.random_batches:
            yield batch

    def __len__(self):
        return len(self.random_batches)

if __name__ == '__main__':
    dataset_path = "data/train.csv"
    dataset = TTSDataset(dataset_path)
    for text, mel in dataset:
        print(mel.shape)
        break
    collator = TTSCollator()
    train_sampler = BatchSampler(dataset, batch_size=4)
    train_loader = DataLoader(dataset, collate_fn=collator, batch_sampler=train_sampler)
    
    for text_padded, mel_padded, gate_padded, masked_text_padded, masked_mel_padded in train_loader:
        print(text_padded.shape, mel_padded.shape, gate_padded.shape, masked_text_padded.shape, masked_mel_padded.shape)
        break
        

    
            
            


            







