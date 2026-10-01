import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.1):
        super().__init__()
        # Causal padding = (kernel_size - 1) * dilation
        pad = (kernel_size - 1) * dilation
        
        self.conv1 = weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size,
                                           padding=pad, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size,
                                           padding=pad, dilation=dilation))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        
        # Remove causal padding from the right
        self.chomp = lambda x, s: x[:, :, :-s].contiguous() if s > 0 else x

    def forward(self, x):
        pad = self.conv1.weight.shape[2]
        dilation = self.conv1.dilation[0]
        s = (pad - 1) * dilation
        
        out = self.relu(self.chomp(self.conv1(x), s))
        out = self.dropout(out)
        
        out = self.relu(self.chomp(self.conv2(out), s))
        out = self.dropout(out)
        
        res = self.downsample(x) if self.downsample else x
        return self.relu(out + res)

class TCNBaseline(nn.Module):
    def __init__(self, in_channels=1, channels=[32, 64, 64, 128, 128, 64], 
                 kernel_size=7, dropout=0.1):
        super().__init__()
        layers = []
        for i, (n_in, n_out) in enumerate(zip([in_channels] + channels[:-1], channels)):
            layers.append(TemporalBlock(n_in, n_out, kernel_size, 
                                        dilation=2**i, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.output_conv = nn.Conv1d(channels[-1], 1, 1)

    def forward(self, x):
        features = self.network(x)
        return self.output_conv(features)
