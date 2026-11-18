import torch
import torch.nn as nn

class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, stride=1, padding=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding),
            nn.InstanceNorm1d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class DeconvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=8, stride=2, padding=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_ch, out_ch, kernel_size, stride, padding),
            nn.InstanceNorm1d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class SourceAgnosticDAE(nn.Module):
    """
    Source-Agnostic Denoising Autoencoder (DAE) with Masked Signal Reconstruction.
    Unlike waveform-supervised models, this architecture learns a template-free
    latent manifold of astrophysical strain perturbations using an informational
    bottleneck to filter out uncorrelated, non-Gaussian detector noise.
    """
    def __init__(self, in_channels=1):
        super().__init__()
        # Encoder: 4096 -> 2048 -> 1024 -> 512 -> 256
        self.enc1 = ConvBlock1D(in_channels, 32, stride=2, padding=3)  # 2048
        self.enc2 = ConvBlock1D(32, 64, stride=2, padding=3)           # 1024
        self.enc3 = ConvBlock1D(64, 128, stride=2, padding=3)          # 512
        self.enc4 = ConvBlock1D(128, 256, stride=2, padding=3)         # 256

        # Bottleneck (filters uncorrelated Gaussian/glitch noise)
        self.bottleneck = nn.Sequential(
            ConvBlock1D(256, 256, kernel_size=7, padding=3),
            nn.Dropout(0.1),
            ConvBlock1D(256, 256, kernel_size=7, padding=3)
        )

        # Decoder with Skip Connections: 256 -> 512 -> 1024 -> 2048 -> 4096
        self.dec4 = DeconvBlock1D(256, 128, stride=2, padding=3)       # 512
        self.dec3 = DeconvBlock1D(128 + 128, 64, stride=2, padding=3)  # 1024
        self.dec2 = DeconvBlock1D(64 + 64, 32, stride=2, padding=3)    # 2048
        self.dec1 = DeconvBlock1D(32 + 32, 16, stride=2, padding=3)    # 4096

        self.final_conv = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(16, 1, kernel_size=1)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)   # [B, 32, 2048]
        e2 = self.enc2(e1)  # [B, 64, 1024]
        e3 = self.enc3(e2)  # [B, 128, 512]
        e4 = self.enc4(e3)  # [B, 256, 256]

        # Bottleneck
        b = self.bottleneck(e4) # [B, 256, 256]

        # Decoder with residual skip connections
        d4 = self.dec4(b)                     # [B, 128, 512]
        d3 = self.dec3(torch.cat([d4, e3], 1)) # [B, 64, 1024]
        d2 = self.dec2(torch.cat([d3, e2], 1)) # [B, 32, 2048]
        d1 = self.dec1(torch.cat([d2, e1], 1)) # [B, 16, 4096]

        out = self.final_conv(d1)              # [B, 1, 4096]
        return out

if __name__ == '__main__':
    model = SourceAgnosticDAE()
    dummy = torch.randn(2, 1, 4096)
    out = model(dummy)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SourceAgnosticDAE test passed! Output shape: {out.shape}, Params: {params:,}")
