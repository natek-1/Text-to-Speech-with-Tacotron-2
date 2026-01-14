import torch.nn as nn


class LinearNorm(nn.Module):
    '''
    A linear layer with a layer with proper initialization
    init:
        in_dim (int): input dimension
        out_dim (int): output dimension
        bias (bool): whether to use bias, default is True
        w_init_gain (str): weight initialization gain, default is 'linear'
    '''

    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        '''
        forward pass
        inputs:
            x (torch.Tensor): input tensor
        outputs:
            torch.Tensor: output tensor
        '''
        return self.linear(x)


class ConvNorm(nn.Module):
    '''
    A convolutional layer with a layer with proper initialization
    init:
        in_dim (int): input dimension
        out_dim (int): output dimension
        kernel_size (int): kernel size, default is 1
        stride (int): stride, default is 1
        padding (int or str): padding, default is None
        dilation (int): dilation, default is 1
        bias (bool): whether to use bias, default is True
        w_init_gain (str): weight initialization gain, default is 'linear'
    '''
    def __init__(self, in_channels, out_channels, kernel_size=1,
    stride=1, padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
        stride=stride, padding=padding, dilation=dilation, bias=bias)
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))
    
    def forward(self, x):
        '''
        forward pass
        inputs:
            x (torch.Tensor): input tensor
        outputs:
            torch.Tensor: output tensor
        '''
        return self.conv(x)
