import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .PositionalEncoding import PositionalEncoding


def scaleshift(x, scale, shift):
    return x * (1 + scale) + shift



class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)


class Patchify1D(nn.Module):
    def __init__(self, patch_size=16, in_channels=1, embed_dim=256, signal_length=1024):
        super().__init__()
        self.proj = nn.Linear(patch_size * in_channels, embed_dim)
        self.patch_size = patch_size
        
    def forward(self, x):
        # x shape: [b, c, length]
        b, c, length = x.shape
        p = self.patch_size
        

        x = x.reshape(b, c, length // p, p)
        x = x.permute(0, 2, 3, 1).contiguous()  # [b, num_patches, patch_size, channels]
        x = x.view(b, -1, p * c)  # [b, num_patches, patch_size * channels]
        
        x = self.proj(x)  # [b, num_patches, embed_dim]
        
        return x


class InputLayer1D(nn.Module):
    def __init__(self, patch_size=16, in_channels=1, embed_dim=256, signal_length=1024):
        super().__init__()
        self.patchify = Patchify1D(patch_size, in_channels, embed_dim, signal_length)
        
        num_patches = signal_length // patch_size
        self.pos_emb = nn.Parameter(self.sin_emb(num_patches, embed_dim))
        
        self.time_emb = nn.Sequential(
            PositionalEncoding(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def sin_emb(self, n_position, d_hid):
        position = torch.arange(n_position).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_hid, 2) * -(math.log(10000.0) / d_hid))
        pos_embedding = torch.zeros(1, n_position, d_hid)
        pos_embedding[0, :, 0::2] = torch.sin(position * div_term)
        pos_embedding[0, :, 1::2] = torch.cos(position * div_term)
        return pos_embedding
    
    def forward(self, x, t):
        x = self.patchify(x)
        x = x + self.pos_emb
        
        t_emb = self.time_emb(t.float().unsqueeze(-1))
        return x, t_emb


class DiTBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=16, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(embed_dim, 6 * embed_dim, bias=True)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.ffnet = FeedForward(embed_dim, hidden_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x, c):
        B, T, C = x.shape
        modulation = self.adaLN_modulation(c)  # Shape: (B, T, 6*C)
        modulation = modulation[:, -1, :]  # Take the last token, shape: (B, 6*C)
        modulation = modulation.view(1, 6, C)  # Reshape to (B, 6, C)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            modulation.chunk(6, dim=1)
        )
        

        int1 = scaleshift(self.norm1(x), scale=scale_msa, shift=shift_msa)
        x = x + gate_msa * self.attn(int1, int1, int1)[0]
        

        x = x + gate_mlp * self.ffnet(scaleshift(self.norm2(x), scale=scale_mlp, shift=shift_mlp))
        
        return x


class OutputLayer1D(nn.Module):
    def __init__(self, patch_size=16, out_channels=1, embed_dim=256, signal_length=1024):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.out = nn.Linear(embed_dim, patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(embed_dim, 2 * embed_dim, bias=True)
        )
        self.patch_size = patch_size
        self.out_channels = out_channels
        
    def forward(self, x, c):
        B, T, C = x.shape
        
        modulation = self.adaLN_modulation(c)
        modulation = modulation[:, -1, :]
        modulation = modulation.view(1, 2, C)

        shift, scale = modulation.chunk(2, dim=1)
        
        x = scaleshift(self.norm(x), scale=scale, shift=shift)
        x = self.out(x) 
        
        return x


class DiT1D(nn.Module):
    def __init__(self, patch_size=16,
                 in_channels=1,
                 out_channels=1,
                 signal_length=1024,
                 embed_dim=256,
                 hidden_dim=128, 
                 dropout=0.1,
                 num_blocks=8,
                 num_heads=16):
        super().__init__()
        self.input_layer = InputLayer1D(patch_size=patch_size,
                                        in_channels=in_channels,
                                        embed_dim=embed_dim,
                                        signal_length=signal_length)
        
        self.blocks = nn.ModuleList([DiTBlock(embed_dim=embed_dim,
                                             num_heads=num_heads,
                                             hidden_dim=hidden_dim,
                                             dropout=dropout) for _ in range(num_blocks)])
        
        self.output_layer = OutputLayer1D(patch_size=patch_size,
                                         out_channels=out_channels,
                                         embed_dim=embed_dim,
                                         signal_length=signal_length)
        
        self.patch_size = patch_size
        self.signal_length = signal_length
        self.out_channels = out_channels
        
        self.__init_weights()
        
    def __init_weights(self):
        # Initialize final layer weights to zero
        nn.init.zeros_(self.output_layer.out.weight)
        nn.init.zeros_(self.output_layer.out.bias)

        # Initialize adaLN weights to zero
        nn.init.zeros_(self.output_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.output_layer.adaLN_modulation[-1].bias)

        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
            
    def forward(self, x, t):
        x, c = self.input_layer(x, t)
        
        for block in self.blocks:
            x = block(x, c)
            
        x = self.output_layer(x, c)
        
        # Reshaping the output 
        b, num_patches, _ = x.shape
        p = self.patch_size
        
        x = x.view(b, num_patches, p, self.out_channels)
        x = x.permute(0, 3, 1, 2).contiguous()  
        
        x = x.view(b, self.out_channels, self.signal_length)
        
        return x

