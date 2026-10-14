"""Model E U-Net with global BPM FiLM and multi-resolution onset/beat features.

Retains the six-level backbone topology, widths, cubic resampling, attention
positions and residual-convolution shapes of the supplied Model E. New weights
and checkpoint schema are required. Uses the repository's existing blocks.py.
"""
import torch
from torch import nn
from torch.nn import functional as F
from audio_diffusion.blocks import FourierFeatures, SelfAttention1d, Downsample1d, Upsample1d

ARCHITECTURE = "model_e_bpm_film_multiscale_onset_beat_v1"
DEFAULT_CHANNELS = (128, 128, 256, 256, 512, 512)


class ConditionalResConv(nn.Module):
    def __init__(self, c_in, c_mid, c_out, embedding_dim, timing_dim, is_last=False):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_mid, 5, padding=2)
        self.norm1 = nn.GroupNorm(1, c_mid)
        self.conv2 = nn.Conv1d(c_mid, c_out, 5, padding=2)
        self.norm2 = nn.GroupNorm(1, c_out) if not is_last else nn.Identity()
        self.skip = nn.Identity() if c_in == c_out else nn.Conv1d(c_in, c_out, 1, bias=False)
        self.is_last = is_last
        if not is_last:
            self.film = nn.Linear(embedding_dim, 2 * c_out)
            self.timing_projection = nn.Conv1d(timing_dim, c_out, 1, bias=False)
            # Start the new conditioning paths as identity/zero residual additions.
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
            nn.init.zeros_(self.timing_projection.weight)

    def forward(self, x, embedding, timing):
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        if not self.is_last:
            gamma, beta = self.film(embedding).chunk(2, dim=1)
            h = h * (1 + gamma[:, :, None]) + beta[:, :, None]
            h = h + self.timing_projection(timing)
            h = F.gelu(h)
        return h + self.skip(x)


class ConditionedLevel(nn.Module):
    def __init__(self, index, channels, io_channels, embedding_dim, timing_dim, attention_start):
        super().__init__()
        self.index = index
        self.root = index == 0
        c = channels[index]
        c_prev = channels[index - 1] if index else io_channels + 16 + 3 + 1
        c_out = channels[index - 1] if index else io_channels
        depth = len(channels)
        self.down = nn.Identity() if self.root else Downsample1d("cubic")
        self.up = nn.Identity() if self.root else Upsample1d("cubic")
        attention = index >= attention_start and not self.root
        self.pre = nn.ModuleList([
            ConditionalResConv(c_prev if n == 0 else c, c, c, embedding_dim, timing_dim)
            for n in range(3)])
        self.pre_attention = nn.ModuleList([
            SelfAttention1d(c, max(1, c // 32)) if attention else nn.Identity() for _ in range(3)])
        self.child = ConditionedLevel(index + 1, channels, io_channels, embedding_dim,
                                      timing_dim, attention_start) if index + 1 < depth else None
        self.post = nn.ModuleList([
            ConditionalResConv(c * 2 if self.child else c, c, c, embedding_dim, timing_dim),
            ConditionalResConv(c, c, c, embedding_dim, timing_dim),
            ConditionalResConv(c, c, c_out, embedding_dim, timing_dim, is_last=self.root)])
        self.post_attention = nn.ModuleList([
            SelfAttention1d(d, max(1, d // 32)) if attention else nn.Identity()
            for d in (c, c, c_out)])

    def forward(self, x, embedding, timing_pyramid):
        skip = x
        x = self.down(x)
        timing = timing_pyramid[self.index]
        if x.shape[-1] != timing.shape[-1]:
            raise RuntimeError("Audio and timing pyramid lengths diverged.")
        for block, attention in zip(self.pre, self.pre_attention):
            x = attention(block(x, embedding, timing))
        if self.child is not None:
            x = self.child(x, embedding, timing_pyramid)
        for block, attention in zip(self.post, self.post_attention):
            x = attention(block(x, embedding, timing))
        x = self.up(x)
        return x if self.root else torch.cat([x, skip], dim=1)


class ConditionedModelE(nn.Module):
    def __init__(self, channels=DEFAULT_CHANNELS, io_channels=8, embedding_dim=128,
                 timing_dim=32, n_attn_layers=3):
        super().__init__()
        self.channels = tuple(channels)
        self.io_channels = io_channels
        self.timestep_embed = FourierFeatures(1, 16)
        self.bpm_embedding = nn.Sequential(nn.Linear(1, embedding_dim), nn.SiLU(),
                                            nn.Linear(embedding_dim, embedding_dim), nn.SiLU())
        self.null_embedding = nn.Parameter(torch.zeros(embedding_dim))
        self.timing_encoder = nn.Sequential(nn.Conv1d(2, timing_dim, 5, padding=2), nn.GELU(),
                                            nn.Conv1d(timing_dim, timing_dim, 3, padding=1), nn.GELU())
        self.net = ConditionedLevel(0, self.channels, io_channels, embedding_dim, timing_dim,
                                    len(self.channels) - n_attn_layers)
        # Preserve the old backbone's initialization scaling; zero adapters stay zero.
        with torch.no_grad():
            for param in self.net.parameters():
                param.mul_(.5)

    def forward(self, x, t, cond, condition_present=None):
        if x.ndim != 3 or x.shape[1] != self.io_channels or x.shape[-1] % (2 ** (len(self.channels)-1)):
            raise ValueError("Invalid latent dimensions or length for U-Net downsampling.")
        if tuple(cond.shape) != (x.shape[0], 3, x.shape[-1]):
            raise ValueError("Expected condition [batch,3,latent_length], got " + str(tuple(cond.shape)))
        cond = cond.to(device=x.device, dtype=x.dtype)
        if condition_present is None:
            present = x.new_ones((x.shape[0],))
        else:
            present = condition_present.to(device=x.device, dtype=x.dtype).reshape(-1)
            if present.numel() != x.shape[0]:
                raise ValueError("condition_present must have one value per example.")
        mask = present[:, None, None]
        masked = cond * mask
        bpm = self.bpm_embedding(masked[:, 0, :1])
        embedding = bpm * present[:, None] + self.null_embedding[None] * (1-present[:, None])
        timing_pyramid = []
        for level in range(len(self.channels)):
            factor = 2 ** level
            # Max pooling keeps short attacks/beats visible at coarser resolutions.
            timing = masked[:, 1:3] if factor == 1 else F.max_pool1d(masked[:, 1:3], factor, factor)
            timing_pyramid.append(self.timing_encoder(timing) * mask)
        time_embedding = self.timestep_embed(t[:, None]).to(dtype=x.dtype)
        time_embedding = time_embedding[:, :, None].expand(-1, -1, x.shape[-1])
        present_channel = mask.expand(-1, 1, x.shape[-1])
        return self.net(torch.cat([x, time_embedding, masked, present_channel], dim=1),
                        embedding, timing_pyramid)
