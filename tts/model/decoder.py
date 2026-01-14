import torch
import torch.nn as nn
import torch.nn.functional as F
from tts.model.layers import LinearNorm



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
    def __init__(self, input_dim, prenet_dim, prenet_depth, dropout=0.5):
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
            x (torch.Tensor): input tensor of shape (batch_size, seq_len, input_dim)
        outputs:
            torch.Tensor: output tensor of shape (batch_size, seq_len, prenet_dim)
        '''
        for layer in self.layers:
            x = F.dropout(layer(x), p=self.dropout, training=True) # force training to true (even during inference)
        return x
    

