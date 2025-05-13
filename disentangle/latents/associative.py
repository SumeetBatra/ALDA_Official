import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List


class AssociativeLatent(nn.Module):
    num_latents: int
    num_values_per_latent: List[int]
    _values_per_latent: List[torch.Tensor]

    def __init__(self, num_latents, num_values_per_latent, beta: float = 100.):
        super(AssociativeLatent, self).__init__()
        self.num_latents = num_latents
        self.beta = beta

        if isinstance(num_values_per_latent, int):
            self.num_values_per_latent = [num_values_per_latent] * num_latents
        else:
            self.num_values_per_latent = num_values_per_latent

        self._values_per_latent = [torch.linspace(-1.0, 1.0, self.num_values_per_latent[i]) for i in range(num_latents)]
        self._values_per_latent = nn.Parameter(torch.stack(self._values_per_latent), requires_grad=True)

    @property
    def values_per_latent(self):
        return self._values_per_latent

    def associate(self, x):
        x = x.unsqueeze(-1).repeat(1, 1, self.values_per_latent.size(-1))
        dists = torch.abs(x - self.values_per_latent[None])
        score = F.softmax(-dists * self.beta, dim=-1)
        return (score * self.values_per_latent).sum(-1)

    def forward(self, x):
        z_quantized = self.associate(x.detach())
        z_hat = x + (z_quantized - x).detach()
        outs = {
            'z_continuous': x,
            'z_quantized': z_quantized,
            'z_hat': z_hat,
        }
        return outs