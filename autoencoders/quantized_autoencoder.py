import torch
import torch.nn as nn

from typing import Tuple, Dict, Any


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, bias=False)
        self.norm = nn.GroupNorm(out_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm = nn.GroupNorm(out_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv2DBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.GroupNorm(out_channels, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.block1(x)
        return x


class QuantizedEncoder(nn.Module):
    def __init__(self,
                 obs_shape: Tuple[int, ...],
                 num_latents: int,
                 amp: bool = False):
        super(QuantizedEncoder, self).__init__()
        self.obs_shape = obs_shape
        self.num_latents = num_latents

        in_channels = self.obs_shape[0]
        self.conv = nn.Sequential(
                    Conv2DBlock(in_channels=in_channels, out_channels=32, kernel_size=4, stride=2, padding=1),
                    Conv2DBlock(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
                    Conv2DBlock(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
                    Conv2DBlock(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
                    nn.Flatten(),
                )

        self.linear = nn.Sequential(
            nn.Linear(in_features=4096, out_features=256),
            nn.GELU(),
            nn.Linear(in_features=256, out_features=256),
            nn.GELU()
        )

        self.linear_head = nn.Linear(in_features=256, out_features=num_latents)

        self.amp = amp

    def encode(self, rgb: torch.Tensor):
        x = self.conv(rgb)
        features = self.linear(x)
        pre_z = self.linear_head(features)
        return pre_z

    def forward(self, rgb: torch.Tensor):
        with torch.autocast(device_type="cuda", enabled=self.amp):
            outs = self.encode(rgb)
        return outs


class OuterEncoder(nn.Module):
    def __init__(self, shared_trunk, shared_history_encoder, projection):
        super(OuterEncoder, self).__init__()
        self.shared_trunk = shared_trunk
        self.shared_history_encoder = shared_history_encoder
        self.projection = projection

    def forward(self, x, detach=False):
        x = self.shared_trunk(x)
        x = self.shared_history_encoder(x)
        if detach:
            x = x.detach()
        x = self.projection(x)
        return x


class Conv2DTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv2DTransposeBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.GroupNorm(out_channels, out_channels),
            nn.GELU()
        )

    def forward(self, x):
        x = self.block1(x)
        return x


class QuantizedDecoder(nn.Module):
    def __init__(self,
                 obs_shape: Tuple[int, ...],
                 transition_shape: Tuple[int, ...],
                 num_latents: int,
                 amp: bool = False):
        super().__init__()
        self.obs_shape = obs_shape
        out_channels = self.obs_shape[0]
        self.amp = amp
        self.projector = nn.Sequential(
            nn.Linear(num_latents, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
        )
        self.network = nn.Sequential(
            nn.Linear(256, 4096),
            nn.GELU(),
            nn.Unflatten(dim=-1, unflattened_size=transition_shape),
            Conv2DTransposeBlock(256, 128, kernel_size=4, stride=2, padding=1),
            Conv2DTransposeBlock(128, 64, kernel_size=4, stride=2, padding=1),
            Conv2DTransposeBlock(64, 32, kernel_size=4, stride=2, padding=1),
            Conv2DTransposeBlock(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(16, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, z):
        '''
        :param z: posterior
        :param deterministic: deterministics
        '''
        with torch.autocast(device_type="cuda", enabled=self.amp):
            z = self.projector(z)
            x_hat_logits = self.network(z)

        return x_hat_logits

    def project(self, z):
        return self.projector(z)