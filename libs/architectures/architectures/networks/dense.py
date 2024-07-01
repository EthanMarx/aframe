from typing import List

import torch
import torch.nn as nn


class FrequencyBandGrouping(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_groups: int):
        super().__init__()
        self.n_groups = n_groups
        self.conv = nn.Conv1d(
            in_channels * n_groups, out_channels, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, freq = x.shape
        x = x.view(batch, channels * self.n_groups, freq // self.n_groups)
        return self.conv(x)


class DenseResidual(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, norm_layer: nn.Module):
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim)
        self.shortcut = (
            nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        )
        self.norm = norm_layer(out_dim)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.layer(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.activation(x)
        return x + residual


class DenseResNet(nn.Module):
    def __init__(
        self,
        num_ifos: int,
        n_freq: int,
        hidden_dims: List[int],
        classes: int,
        norm_layer: nn.Module,
        zero_init_residual: bool = False,
    ):
        super().__init__()
        self.num_ifos = num_ifos
        self.n_freq = n_freq

        self.downsample = nn.Linear(n_freq, 256)
        self.initial_proj = nn.Linear(2 * num_ifos, hidden_dims[0])
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.blocks.append(
                DenseResidual(hidden_dims[i], hidden_dims[i + 1], norm_layer)
            )

        self.classifier = nn.Linear(hidden_dims[-1] * 256, classes)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, DenseResidual):
                    nn.init.constant_(m.norm.weight, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = x.permute(0, 2, 1)
        x = self.initial_proj(x)

        for block in self.blocks:
            x = block(x)

        x = x.reshape(x.shape[0], -1)
        return self.classifier(x)
