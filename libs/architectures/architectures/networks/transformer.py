import math

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=3,
            padding=3,
            bias=False,
        )
        self.norm = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        num_ifos: int,
        d_model: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_layers: int = 18,
        prenorm: bool = True,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.num_ifos = num_ifos
        self.dropout = dropout

        self.conv_proj = ConvBlock(
            in_channels=num_ifos,
            out_channels=self.d_model,
        )
        self.avgpool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)

        self.pos_encoder = PositionalEncoding(
            d_model, maxlen=512, dropout=dropout
        )
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
            norm_first=prenorm,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=self.num_layers
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.d_model, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_proj(x)
        x = self.avgpool(x)
        x = self.pos_encoder(x)

        # rearrange data dormat to (batch, seq, features)
        x = x.permute(0, 2, 1)
        x = self.transformer_encoder(x)

        # restore data dormat to (batch, features, seq)
        x = x.permute(0, 2, 1)
        # pooling and flatten
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
