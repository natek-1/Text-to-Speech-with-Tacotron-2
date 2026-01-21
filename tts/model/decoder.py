import torch
import torch.nn as nn
import torch.nn.functional as F
from tts.model.layers import LinearNorm, ConvNorm
from tts.model.attention import LocalSensitiveAttention
from tts.model.config import Tacotron2Config





class Prenet(nn.Module):
    """
    Prenet module for Tacotron 2
    We will pass the previous timestep through the prenet. Helps with feature extraction and stochasticity.

    init:
        input_dim (int): input dimension
        prenet_dim (int): prenet dimension
        prenet_depth (int): prenet depth
        dropout (float): dropout rate
    """
    def __init__(self, input_dim, prenet_dim, prenet_depth, dropout=0.1):
        super(Prenet, self).__init__()
        self.dropout = dropout
        dims = [input_dim] + [prenet_dim for _ in range(prenet_depth)]
        self.layers = nn.ModuleList()
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.layers.append(
                nn.Sequential(
                    LinearNorm(in_dim, out_dim, bias=False, w_init_gain='relu'),
                    nn.ReLU(),
                )
            )
    
    def forward(self, x):
        '''
        forward pass
        inputs:
            x (torch.Tensor): input tensor of shape (batch_size, seq_len, input_dim) or (batch_size, input_dim)
        outputs:
            torch.Tensor: output tensor of shape (batch_size, seq_len, prenet_dim) or (batch_size, prenet_dim)
        '''
        for layer in self.layers:
            x = F.dropout(layer(x), p=self.dropout, training=True) # force training to true (even during inference)
        return x

class Postnet(nn.Module):
    '''
    Postnet module for Tacotron 2
    Per the paper, it takes the final generated mel-spectrogram and refines it
    
    init:
        num_mels (int): number of mel-spectrogram channels
        num_convs (int): number of convolutional layers
        num_filters (int): number of filters in the convolutional layers
        kernel_size (int): kernel size of the convolutional layers
        dropout (float): dropout rate
    '''
    def __init__(self, num_mels, num_convs=5, num_filters=512, kernel_size=5, dropout=0.5):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()
        self.dropout = dropout
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(num_mels, num_filters,
                kernel_size=kernel_size, padding="same", w_init_gain="tanh"),
                nn.BatchNorm1d(num_filters),
                nn.Tanh(),
                nn.Dropout(dropout)
            )
        )
        for i in range(num_convs - 2):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(num_filters, num_filters,
                    kernel_size=kernel_size, padding="same", w_init_gain="tanh"),
                    nn.BatchNorm1d(num_filters),
                    nn.Tanh(),
                    nn.Dropout(dropout)
                )
            )
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(num_filters, num_mels,
                kernel_size=kernel_size, padding="same"),
                nn.BatchNorm1d(num_mels),
                nn.Dropout(dropout)
            )
        )
    
    def forward(self, x):
        '''
        forward pass
        inputs:
            x (torch.Tensor): input tensor of shape (batch_size, seq_len, num_mels)
        outputs:
            torch.Tensor: output tensor of shape (batch_size, seq_len, num_mels)
        '''
        x = x.transpose(1, 2) # (batch_size, num_mels, seq_len)
        for conv in self.convolutions:
            x = F.dropout(conv(x), p=self.dropout, training=self.training)
        x = x.transpose(1, 2) # (batch_size, seq_len, num_mels)
        return x

    

class Decoder(nn.Module):
    '''
    Decoder module for Tacotron 2
    Per the paper, it takes encoder outputs and generates mel-spectrogram
    
    init:
    config (Tacotron2Config): configuration for the decoder
    '''

    def __init__(self, config: Tacotron2Config):
        super(Decoder, self).__init__()
        self.config = config

        self.prenet = Prenet(input_dim=config.num_mels,
                             prenet_dim=config.decoder_prenet_dim,
                             prenet_depth=config.decoder_prenet_num_layers,
                             dropout=config.decoder_prenet_dropout)
        
        self.rnn = nn.ModuleList(
            [
                nn.LSTMCell(config.decoder_prenet_dim + config.encoder_embed_dim, config.decoder_embed_dim),
                nn.LSTMCell(config.decoder_embed_dim + config.encoder_embed_dim, config.decoder_embed_dim)
            ]
        )

        self.attention = LocalSensitiveAttention(attention_dim=config.attention_dim,
                                                    decoder_hidden_size=config.decoder_embed_dim,
                                                    encoder_hidden_size=config.encoder_embed_dim,
                                                    attention_num_filters=config.attention_location_num_filters,
                                                    attention_kernel_size=config.attention_location_kernel_size)
        
        self.mel_proj = LinearNorm(config.decoder_embed_dim + config.encoder_embed_dim, config.num_mels,
                                    bias=True, w_init_gain='linear')
        self.stop_proj = LinearNorm(config.decoder_embed_dim + config.encoder_embed_dim, 1,
                                    bias=True, w_init_gain='sigmoid')
        
        self.postnet = Postnet(num_mels=config.num_mels,
                               num_convs=config.decoder_postnet_num_conv,
                               num_filters=config.decoder_postnet_num_filters,
                               kernel_size=config.decoder_postnet_kernel_size,
                               dropout=config.decoder_postnet_dropout)

    def _init_decoder(self, encoder_output, encoder_mask=None):
        '''
        Initialize the decoder with the encoder output
        inputs:
            encoder_output (torch.Tensor): encoder output of shape (batch_size, seq_len, encoder_embed_dim)
            encoder_mask (torch.Tensor): encoder mask of shape (batch_size, seq_len)
        '''
        B, S, E = encoder_output.shape
        device = encoder_output.device
        

        self.h = [torch.zeros((B, self.config.decoder_embed_dim), device=device) for _ in range(2)]
        self.c = [torch.zeros((B, self.config.decoder_embed_dim), device=device) for _ in range(2)]

        self.cumulative_attention_weights = torch.zeros((B, S), device=device)
        self.attention_weights = torch.zeros((B, S), device=device)
        self.attention_context = torch.zeros((B, self.config.encoder_embed_dim), device=device)

        self.encoder_output = encoder_output
        self.encoder_mask = encoder_mask
    
    def _bos_frame(self, B):
        '''
        returns a batch of BOS frames
        inputs:
            B (int): batch size
        outputs:
            torch.Tensor: BOS frames of shape (B, 1, num_mels) on cpu
        '''
        return torch.zeros((B, 1, self.config.num_mels))

    
    def decode(self, mel_step):
        '''
        At each step, the decoder takes the previous mel-step and the encoder output and generates the next mel-step
        inputs:
            mel_step (torch.Tensor): mel-step of shape (B, prenet_dim)
        outputs:
            torch.Tensor: mel-step of shape (B, num_mels)
            torch.Tensor: stop token of shape (B, 1)
            torch.Tensor: attention weights of shape (B, S)
        '''
        rnn_input = torch.cat([mel_step, self.attention_context], dim=-1) # (B, prenet_dim + encoder_embed_dim)

        self.h[0], self.c[0] = self.rnn[0](rnn_input, (self.h[0], self.c[0])) # (B, decoder_embed_dim), (B, decoder_embed_dim)
        attn_hidden = F.dropout(self.h[0], p=self.config.attention_dropout, training=self.training) # (B, decoder_embed_dim)

        attn_weights_cat = torch.cat([
            self.attention_weights.unsqueeze(1), self.cumulative_attention_weights.unsqueeze(1)
        ], dim=1) # (B, 2, S)

        attention_context, attention_weights = self.attention(attn_hidden, self.encoder_output,
            attn_weights_cat, self.encoder_mask) # (B, encoder_embed_dim), (B, S)

        self.attention_weights = attention_weights # (B, S)
        self.cumulative_attention_weights += attention_weights # (B, S)
        self.attention_context = attention_context # (B, encoder_embed_dim)

        decoder_input = torch.cat([
            attn_hidden,
            self.attention_context
        ], dim=-1) # (B, decoder_embed_dim + encoder_embed_dim)

        self.h[1], self.c[1] = self.rnn[1](decoder_input, (self.h[1], self.c[1])) # (B, decoder_embed_dim), (B, decoder_embed_dim)
        decoder_hidden = F.dropout(self.h[1], p=self.config.attention_dropout, training=self.training) # (B, decoder_embed_dim)

        next_pred = torch.cat([
            decoder_hidden,
            self.attention_context
        ], dim=-1) # (B, decoder_embed_dim + encoder_embed_dim)

        mel_pred = self.mel_proj(next_pred) # (B, num_mels)
        stop_pred = self.stop_proj(next_pred) # (B, 1)

        return mel_pred, stop_pred, attention_weights


    def forward(self, encoder_output, encoder_mask, mels, decoder_mask):
        """
        Forward pass of the decoder
        inputs:
            encoder_output (torch.Tensor): encoder output of shape (batch_size, seq_len, encoder_embed_dim)
            encoder_mask (torch.Tensor): encoder mask of shape (batch_size, seq_len)
            mels (torch.Tensor): mel-spectrogram (for teacher forcing) of shape (batch_size, mel_seq_len, num_mels)
            decoder_mask (torch.Tensor): decoder mask of shape (batch_size, mel_seq_len)
        outputs:
            torch.Tensor: predicted mel-spectrogram of shape (batch_size, mel_seq_len, num_mels)
            torch.tensor: mel-spectrogram residual of shape (batch_size, mel_seq_len, num_mels)
            torch.Tensor: stop token of shape (batch_size, mel_seq_len)
            torch.tensor: attention weights of shape (batch_size, mel_seq_len, seq_len)
        """
        start_feature_vector = self._bos_frame(mels.shape[0]).to(encoder_output.device)
        mels_w_start = torch.cat([start_feature_vector, mels], dim=1) # (B, mel_seq_len + 1, num_mels)
        
        self._init_decoder(encoder_output, encoder_mask)

        mels_output, stop_tokens, attention_weights = [], [], []
        total_steps = mels.shape[1] # notice we only make predictions for the mel_seq_len (not mel_seq_len + 1)
        mel_proj = self.prenet(mels_w_start) # teacher forcing (B, mel_seq_len + 1, prenet_dim)

        for step in range(total_steps):
            if step == 0:
                self.attention.reset()
            
            mel_step = mel_proj[:, step, :] # (B, prenet_dim)
            mel_pred, stop_pred, attention_weight = self.decode(mel_step) # (B, num_mels), (B, 1), (B, S)
            mels_output.append(mel_pred)
            stop_tokens.append(stop_pred)
            attention_weights.append(attention_weight)
        
        mels_output = torch.stack(mels_output, dim=1) # (B, mel_seq_len, num_mels)
        stop_tokens = torch.stack(stop_tokens, dim=1).squeeze() # (B, mel_seq_len)
        attention_weights = torch.stack(attention_weights, dim=1) # (B, mel_seq_len, seq_len)

        mel_residual = self.postnet(mels_output) # (B, mel_seq_len, num_mels)


        # masking
        decoder_mask = decoder_mask.unsqueeze(-1).bool() # (B, mel_seq_len, 1)
        mel_residual = mel_residual.masked_fill(decoder_mask, 0.0) # (B, mel_seq_len, num_mels)
        mels_output = mels_output.masked_fill(decoder_mask, 0.0) # (B, mel_seq_len, num_mels)
        attention_weights = attention_weights.masked_fill(decoder_mask, 0.0) # (B, mel_seq_len, seq_len)
        stop_tokens = stop_tokens.masked_fill(decoder_mask.squeeze(), 1e3) # (B, mel_seq_len)

        return mels_output, mel_residual, stop_tokens, attention_weights

    @torch.inference_mode()
    def inference(self, encoder_output, max_decode_steps=1000):
        '''
        Generates mel-spectrogram from encoder output
        inputs:
            encoder_output (torch.Tensor): encoder output of shape (1, seq_len, encoder_embed_dim)
            max_decode_steps (int): maximum number of decoding steps
        outputs:
            torch.Tensor: predicted mel-spectrogram of shape (1, mel_seq_len, num_mels)
            torch.tensor: mel-spectrogram residual of shape (1, mel_seq_len, num_mels)
            torch.Tensor: stop token of shape (1, mel_seq_len)
            torch.tensor: attention weights of shape (1, mel_seq_len, text_seq_len)
        '''
        self._init_decoder(encoder_output, None)

        mels_output, stop_outs, attention_weights = [], [], []

        _input = self._bos_frame(B=1).squeeze(0).to(encoder_output.device) # (1, num_mels)
        self.attention.reset()

        for step in range(max_decode_steps):
            _input = self.prenet(_input) # (1, prenet_dim)

            mel_pred, stop_pred, attention_weight = self.decode(_input) # (1, num_mels), (1, 1), (1, seq_len)
            mels_output.append(mel_pred)
            stop_outs.append(stop_pred)
            attention_weights.append(attention_weight)
            _input = mel_pred

            if torch.sigmoid(stop_pred) > 0.5:
                break
        
        mels_output = torch.stack(mels_output, dim=1) # (1, mel_seq_len, num_mels)
        mel_residual = self.postnet(mels_output) # (1, mel_seq_len, num_mels)
        stop_outs = torch.stack(stop_outs, dim=1).squeeze() # (1, mel_seq_len)
        attention_weights = torch.stack(attention_weights, dim=1) # (1, mel_seq_len, seq_len)
        
        return mels_output, mel_residual, stop_outs, attention_weights
        
        

            
        





            
            
            
        
        

    





