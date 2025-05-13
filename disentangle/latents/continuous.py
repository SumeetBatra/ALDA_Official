import torch
import torch.nn as nn


class ContinuousLatent(nn.Module):
    def __init__(self, num_latents):
        super(ContinuousLatent, self).__init__()
        self.is_continuous = True
        self.num_latents = num_latents
        self.num_inputs = num_latents

    def forward(self, x):
        outs = {
            'z_hat': x
        }
        return outs