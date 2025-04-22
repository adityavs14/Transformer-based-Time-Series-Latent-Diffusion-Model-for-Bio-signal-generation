import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from PositionalEncoding import PositionalEncoding
from Block import Block1D
from Attention import Attention


class UNet1DAttn(nn.Module):
    def __init__(self):
        super().__init__()
        signal_channels = 1
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 1 
        time_emb_dim = 64

        self.time_mlp = nn.Sequential(
            PositionalEncoding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        self.conv0 = nn.Conv1d(signal_channels, down_channels[0], kernel_size=3, padding=1)

        self.downs = nn.ModuleList([
            Block1D(down_channels[i], down_channels[i+1], time_emb_dim)
            for i in range(len(down_channels)-1)
        ])

        self.ups = nn.ModuleList([
            Block1D(up_channels[i], up_channels[i+1], time_emb_dim, up=True)
            for i in range(len(up_channels)-1)
        ])

        self.attentions = nn.ModuleList([
            Attention(up_channels[i], up_channels[i], up_channels[i+1])
            for i in range(len(up_channels)-1)
        ])

        self.output = nn.Sequential(
            nn.Conv1d(up_channels[-1], out_dim, kernel_size=1),
        )

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)

        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        
        for i, up in enumerate(self.ups):
            residual_x = residual_inputs.pop()
            

            gated_residual = self.attentions[i](x, residual_x)

            x = torch.cat((x, gated_residual), dim=1)
            x = up(x, t)

        return self.output(x)