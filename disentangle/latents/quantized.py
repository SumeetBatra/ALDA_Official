import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from common.utils import off_diagonal


class QuantizedLatent(nn.Module):
    is_continuous: bool
    num_latents: int
    num_inputs: int

    num_values_per_latent: List[int]
    _values_per_latent: List[torch.Tensor]
    optimize_values: bool

    def __init__(self, num_latents, num_values_per_latent, optimize_values):
        super(QuantizedLatent, self).__init__()
        self.is_continuous = False
        self.num_latents = num_latents
        self.num_inputs = num_latents

        if isinstance(num_values_per_latent, int):
            self.num_values_per_latent = [num_values_per_latent] * num_latents
        else:
            self.num_values_per_latent = num_values_per_latent

        self._values_per_latent = [torch.linspace(-1.0, 1.0, self.num_values_per_latent[i]) for i in range(num_latents)]
        self._values_per_latent = nn.Parameter(torch.stack(self._values_per_latent), requires_grad=True)
        self.optimize_values = optimize_values

        self.latent_projector = nn.Sequential(
            nn.Linear(self.num_latents, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.LayerNorm(128)
        )

    @property
    def values_per_latent(self):
        if self.optimize_values:
            return self._values_per_latent
        else:
            return [v.detach() for v in self._values_per_latent]

    def quantize(self, x):
        x = torch.repeat_interleave(x[..., None], repeats=self.values_per_latent.shape[1], dim=-1)
        dists = torch.abs(x - self.values_per_latent[None])
        inds = torch.argmin(dists, dim=-1)
        values_expanded = self.values_per_latent[None].repeat_interleave(x.shape[0], dim=0)
        values = torch.gather(values_expanded, dim=-1, index=inds[..., None]).squeeze()
        return values, inds

    def forward(self, x):
        quantized, indices = self.quantize(x)
        quantized_sg = x + (quantized - x).detach()
        outs = {
            'z_continuous': x,
            'z_quantized': quantized,
            'z_hat': quantized_sg,
            'z_indices': indices
        }

        return outs

    def sample(self, batch_size):
        ret = []
        for values in self.values_per_latent:
            sample = values[torch.randint(len(values), (batch_size, 1))]
            ret.append(sample)
        return torch.hstack(ret)

    def decorrelate(self, z):
        proj_z = self.latent_projector(z)
        cov = (proj_z.T @ proj_z) / (proj_z.size(0) - 1)
        cov_loss = off_diagonal(cov).pow(2).sum().div(proj_z.size(1))
        return cov_loss
