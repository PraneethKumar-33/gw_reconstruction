import torch
import torch.nn as nn
import math

class PatchEmbedding1D(nn.Module):
    """
    Splits 1D time-series into non-overlapping patches and projects them to d_model.
    Input: [B, 1, 4096] -> Output: [B, num_patches, d_model]
    With patch_size=16, num_patches = 4096 / 16 = 256.
    """
    def __init__(self, in_channels=1, patch_size=16, d_model=128):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, 1, 4096]
        x = self.proj(x) # [B, d_model, 256]
        x = x.transpose(1, 2) # [B, 256, d_model]
        return x

class PatchDecoder1D(nn.Module):
    """
    Decodes patch representations back to the original time-series resolution.
    Input: [B, num_patches, d_model] -> Output: [B, 1, 4096]
    """
    def __init__(self, out_channels=1, patch_size=16, d_model=128):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(d_model, 64, kernel_size=patch_size, stride=patch_size)
        self.refine = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(32, out_channels, kernel_size=7, padding=3)
        )

    def forward(self, x):
        # x: [B, 256, d_model]
        x = x.transpose(1, 2) # [B, d_model, 256]
        x = self.upsample(x)  # [B, 64, 4096]
        x = self.refine(x)    # [B, 1, 4096]
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerReconstructor(nn.Module):
    """
    Waveform-Supervised 1D Temporal Transformer Reconstructor.
    Uses multi-head self-attention to capture long-range phase coherence 
    and frequency chirp evolution across the 4096-sample observation window.
    """
    def __init__(self, in_channels=1, patch_size=16, d_model=128, nhead=4, 
                 num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding1D(in_channels=in_channels, patch_size=patch_size, d_model=d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=512, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.decoder = PatchDecoder1D(out_channels=in_channels, patch_size=patch_size, d_model=d_model)

    def forward(self, x):
        # x: [B, 1, 4096]
        # 1. Patch Tokenization: 4096 samples -> 256 tokens
        tokens = self.patch_embed(x) # [B, 256, 128]
        
        # 2. Positional Encoding
        tokens = self.pos_encoder(tokens)
        
        # 3. Multi-Head Self-Attention
        encoded = self.transformer_encoder(tokens) # [B, 256, 128]
        encoded = self.norm(encoded)
        
        # 4. Patch Decoding: 256 tokens -> 4096 samples
        out = self.decoder(encoded) # [B, 1, 4096]
        return out

if __name__ == '__main__':
    model = TransformerReconstructor()
    dummy = torch.randn(2, 1, 4096)
    out = model(dummy)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TransformerReconstructor test passed! Output shape: {out.shape}, Params: {params:,}")
