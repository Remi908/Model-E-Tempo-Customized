import torch
from torch import nn
from torch.nn import functional as F


class ResBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class AudioVAE1D(nn.Module):
    """
    Simple 1D convolutional VAE for stereo audio.

    Input:
        audio: [B, 2, T]

    Output:
        reconstruction: [B, 2, T]
        mu: [B, latent_channels, T / downsample_factor]
        logvar: [B, latent_channels, T / downsample_factor]
    """

    def __init__(self, in_channels=2, latent_channels=8, base_channels=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3),

            nn.Conv1d(base_channels, base_channels, kernel_size=4, stride=2, padding=1),
            ResBlock1D(base_channels),

            nn.Conv1d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            ResBlock1D(base_channels * 2),

            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            ResBlock1D(base_channels * 4),

            nn.Conv1d(base_channels * 4, base_channels * 4, kernel_size=4, stride=2, padding=1),
            ResBlock1D(base_channels * 4),
        )

        self.to_mu = nn.Conv1d(base_channels * 4, latent_channels, kernel_size=1)
        self.to_logvar = nn.Conv1d(base_channels * 4, latent_channels, kernel_size=1)

        self.from_latent = nn.Conv1d(latent_channels, base_channels * 4, kernel_size=1)

        self.decoder = nn.Sequential(
            ResBlock1D(base_channels * 4),
            nn.ConvTranspose1d(base_channels * 4, base_channels * 4, kernel_size=4, stride=2, padding=1),

            ResBlock1D(base_channels * 4),
            nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),

            ResBlock1D(base_channels * 2),
            nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),

            ResBlock1D(base_channels),
            nn.ConvTranspose1d(base_channels, base_channels, kernel_size=4, stride=2, padding=1),

            nn.GroupNorm(8, base_channels),
            nn.GELU(),
            nn.Conv1d(base_channels, in_channels, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    def encode(self, audio):
        h = self.encoder(audio)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        logvar = torch.clamp(logvar, min=-20.0, max=8.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z):
        h = self.from_latent(z)
        audio = self.decoder(h)
        return audio

    def forward(self, audio):
        mu, logvar = self.encode(audio)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar