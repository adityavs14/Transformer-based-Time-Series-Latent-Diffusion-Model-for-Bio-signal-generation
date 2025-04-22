import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Attention(nn.Module):
    def __init__(self, g_channels, x_channels, intermediate_channels):
        super().__init__()
        
        self.Wg = nn.Conv1d(g_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.Wx = nn.Conv1d(x_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv1d(intermediate_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.act = nn.Sigmoid()
        self.relu = nn.LeakyReLU()

    def forward(self, g, x):

        g1 = self.Wg(g)  
        x1 = self.Wx(x)  
        psi = self.relu(g1 + x1)
        
        psi = self.act(self.psi(psi))  
        
        return x * psi  