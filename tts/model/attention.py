import torch
import torch.nn as nn
import torch.nn.functional as F

from tts.model.layers import ConvNorm, LinearNorm


class LocationLayer(nn.Module):
    '''
    Location layer for attention.
    Per the paper, this layer considers the current attention weights and cumulative attention
    and uses a convolutional layer to extract features
    init:
        attention_n_filters (int): number of filters in the convolutional layer
        attention_kernel_size (int): kernel size of the convolutional layer
        attention_dim (int): dimension of the attention
    '''

    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim):
        super(LocationLayer, self).__init__()
        self.conv = ConvNorm(in_channels=2,
                        out_channels=attention_n_filters,
                        kernel_size=attention_kernel_size, padding="same", bias=False)
        self.proj = LinearNorm(attention_n_filters, attention_dim, bias=False)

    
    def forward(self, attention_weights):
        """
        forward pass
        inputs:
            attention_weights (torch.Tensor): concatenated attention weights with cumulative attention (batch_size, 2, seq_len)
        outputs:
            torch.Tensor: location vector of shape (batch_size, seq_len, attention_dim)
        """
        location = self.conv(attention_weights).transpose(1, 2) # (batch_size, attention_n_filters, seq_len) -> (batch_size, seq_len, attention_n_filters)
        location = self.proj(location) # (batch_size, seq_len, attention_dim)
        return location

class LocalSensitiveAttention(nn.Module):
    '''
    Local sensitive attention module for Tacotron 2
    Per paper, this module is used to calculate the attention weights using:
    encoder output, current_decoder_step, cumulative attention
    init:
        attention_dim (int): dimension of the attention
        decoder_hidden_size (int): dimension of the decoder hidden state
        encoder_hidden_size (int): dimension of the encoder hidden state
        attention_n_filters (int): number of filters in the convolutional layer
        attention_kernel_size (int): kernel size of the convolutional layer
    '''

    def __init__(self, attention_dim, decoder_hidden_size, encoder_hidden_size,
            attention_n_filters, attention_kernel_size):
        super(LocalSensitiveAttention, self).__init__()
        self.in_proj = LinearNorm(decoder_hidden_size, attention_dim, bias=True, w_init_gain='tanh')
        self.enc_proj = LinearNorm(encoder_hidden_size, attention_dim, bias=True, w_init_gain='tanh')


        self.location_layer = LocationLayer(attention_n_filters, attention_kernel_size, attention_dim)
        
        self.proj = LinearNorm(attention_dim, 1, bias=False, w_init_gain='linear')
        self.reset()
    
    def reset(self):
        self.encoder_cache = None
    
    def _calculate_energy(self, decoder_hidden, encoder_hidden, attention_weights_cum, mask=None):
        '''
        calculate energy
        inputs:
            decoder_hidden (torch.Tensor): decoder hidden state of shape (batch_size, decoder_hidden_size)
            encoder_hidden (torch.Tensor): encoder hidden state of shape (batch_size, seq_len, encoder_hidden_size)
            attention_weights_cum (torch.Tensor): cumulative attention weights of shape (batch_size, seq_len)
            mask (torch.Tensor): mask of shape (batch_size, seq_len)
        outputs:
            torch.Tensor: energy of shape (batch_size, seq_len)
        '''
        decoder_hidden = self.in_proj(decoder_hidden).unsqueeze(1) # (batch_size, 1, attention_dim)
        if self.encoder_cache is None:
            self.encoder_cache = self.enc_proj(encoder_hidden) # (batch_size, seq_len, attention_dim)
        encoder_hidden = self.encoder_cache

        cumulative_attention = self.location_layer(attention_weights_cum) # (batch_size, seq_len, attention_dim)


        energy = self.proj(
            torch.tanh(decoder_hidden + encoder_hidden + cumulative_attention)
        ) # (batch_size, seq_len, 1)
        energy = energy.squeeze(-1) # (batch_size, seq_len)
        if mask is not None:
            energy = energy.masked_fill(mask, -float('inf'))
        return energy
    
    def forward(self, decoder_hidden, encoder_hidden, attention_weights_cum, mask=None):
        energy = self._calculate_energy(decoder_hidden, encoder_hidden,
                attention_weights_cum, mask)
        attention_weights = F.softmax(energy, dim=-1) # (batch_size, seq_len)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_hidden).squeeze(1) # (batch_size, encoder_hidden_size)
        return context, attention_weights



        
    
    


        