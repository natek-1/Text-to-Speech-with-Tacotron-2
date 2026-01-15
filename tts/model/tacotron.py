import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tts.model.config import Tacotron2Config
from tts.model.encoder import Encoder
from tts.model.decoder import Decoder



class Tacotron2(nn.Module):
    '''
    Tacotron 2 model for text-to-speech
    
    init:
        config (Tacotron2Config): configuration for the Tacotron 2 model
    '''
    def __init__(self, config: Tacotron2Config):
        super(Tacotron2, self).__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
    
    def forward(self, text, input_lengths, mel_target, encoder_mask, decoder_mask):
        '''
        forward pass
        inputs:
            text (torch.Tensor): input text of shape (batch_size, seq_len)
            input_lengths (torch.Tensor): input lengths of shape (batch_size)
            mel_target (torch.Tensor): mel-spectrogram target of shape (batch_size, mel_seq_len, num_mels)
            encoder_mask (torch.Tensor): encoder mask of shape (batch_size, seq_len)
            decoder_mask (torch.Tensor): decoder mask of shape (batch_size, seq_len)
        outputs:
            mel_outputs (torch.Tensor): mel-spectrogram outputs of shape (batch_size, mel_seq_len, num_mels)
            stop_outputs (torch.Tensor): stop outputs of shape (batch_size, mel_seq_len)
            alignment (torch.Tensor): alignment of shape (batch_size, mel_seq_len, encoder_seq_len)
        '''
        encoder_outputs = self.encoder(text, input_lengths)
        mel_outputs, mel_residual, stop_outputs, alignment = self.decoder(encoder_outputs, encoder_mask,
                                                             mel_target, decoder_mask)
        mel_outputs_postnet = mel_outputs + mel_residual
        return mel_outputs_postnet, stop_outputs, alignment
    
    @torch.no_grad()
    def inference(self, text, input_lengths):
        '''
        inference pass
        inputs:
            text (torch.Tensor): input text of shape (batch_size, seq_len)
            input_lengths (torch.Tensor): input lengths of shape (batch_size)
        outputs:
            mel_outputs (torch.Tensor): mel-spectrogram outputs of shape (batch_size, mel_seq_len, num_mels)
            alignment (torch.Tensor): alignment of shape (batch_size, mel_seq_len, encoder_seq_len)
        '''
        if text.ndim == 1:
            text = text.unsqueeze(0)
        
        assert text.shape[0] == 1, "Batch size must be 1 for inference"
        encoder_outputs = self.encoder(text, input_lengths)
        mel_outputs, mel_residual, _, alignment = self.decoder.inference(encoder_outputs)
        mel_outputs_postnet = mel_outputs + mel_residual

        return mel_outputs_postnet, alignment


if __name__ == "__main__":
    from tts.dataset.dataset import TTSDataset, TTSCollator, BatchSampler

    dataset_path = "data/train.csv"
    dataset = TTSDataset(dataset_path)
    
    config = Tacotron2Config()
    model = Tacotron2(config)
    collator = TTSCollator()
    train_sampler = BatchSampler(dataset, batch_size=4)
    train_loader = DataLoader(dataset, collate_fn=collator, batch_sampler=train_sampler)

    for batch in train_loader:
        text, input_lengths, mel_target, gate_target, encoder_mask, decoder_mask = batch
        print("Input")
        print(text.shape)
        print(input_lengths.shape)
        print(mel_target.shape)
        print("model")
        mel_outputs, stop_outputs, alignment = model(text, input_lengths, mel_target, encoder_mask, decoder_mask)
        print("Output")
        print(mel_outputs.shape)
        print(alignment.shape)
        break
        
    
        
        