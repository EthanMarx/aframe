import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        maxlen: int,
        dropout: float = 0.1,
    ):
        super(PositionalEncoding, self).__init__()
        self.PE = torch.tensor(
            [
                [pos / 1000 ** (i // 2 * 2 / d_model) for pos in range(maxlen)]
                for i in range(d_model)
            ]
        )
        self.PE[:, 0::2] = np.sin(self.PE[:, 0::2])
        self.PE[:, 1::2] = np.cos(self.PE[:, 1::2])

        self.register_buffer("pe", self.PE)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        out = self.dropout(input + self.pe)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        num_ifos: int,
        max_len: int,
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

        self.conv1d = nn.Conv1d(
            in_channels=num_ifos,
            out_channels=self.d_model,
            kernel_size=7,
            stride=3,
            padding=3,
            bias=False,
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
        x = self.conv1d(x)
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
