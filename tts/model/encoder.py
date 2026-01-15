import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from tts.model.config import Tacotron2Config
from tts.model.layers import ConvNorm, LinearNorm



class Encoder(nn.Module):
    '''
    Encoder module for Tacotron 2
    init:
        config (Tacotron2Config): configuration for the encoder
    '''

    def __init__(self, config: Tacotron2Config):
        super(Encoder, self).__init__()
        self.config = config

        self.embedding = nn.Embedding(self.config.num_chars, self.config.character_embedding_size, padding_idx=self.config.pad_token_id)

        self.convolutions = nn.ModuleList()
        for i in range(self.config.encoder_num_convolutions):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(self.config.encoder_embed_dim if i != 0 else self.config.character_embedding_size,
                    self.config.encoder_embed_dim,
                    kernel_size=self.config.encoder_kernel_size, padding="same", w_init_gain="relu"),
                    nn.BatchNorm1d(self.config.encoder_embed_dim),
                    nn.ReLU(),
                    nn.Dropout(self.config.encoder_dropout)
                )
            )
        self.lstm = nn.LSTM(self.config.encoder_embed_dim,
                    self.config.encoder_embed_dim // 2, # divide by 2 because it is bidirectional
                    bidirectional=True, batch_first=True)
    
    def forward(self, x, input_lengths=None):
        '''
        forward pass
        inputs:
            x (torch.Tensor): input tensor of shape (batch_size, seq_len)
            input_lengths (torch.Tensor): input lengths of shape (batch_size)
        outputs:
            torch.Tensor: output tensor of shape (batch_size, seq_len, encoder_embed_dim)
        '''
        x = self.embedding(x).transpose(1, 2) # (batch_size, character_embedding_size, seq_len)
        batch_size, _, seq_len = x.shape
        if input_lengths is None:
            input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=x.device)
        
        for conv in self.convolutions:
            x = conv(x)
        x = x.transpose(1, 2) # (batch_size, seq_len, encoder_embed_dim)
        
        x = pack_padded_sequence(x, input_lengths.cpu(), batch_first=True)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True) # (batch_size, seq_len, encoder_embed_dim)
        return x
