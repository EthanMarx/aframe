import torch
import torch.nn as nn


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, length, dropout=0.1):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos = nn.Parameter(torch.empty(1, length, d_model)) 
        nn.init.uniform_(self.pos, -0.02, 0.02)

    def forward(self, x):
        x = x + self.pos
        return self.dropout(x)
    

class TransformerEncoder(nn.Module):

    def __init__(
        self, 
        head_size, 
        num_heads, 
        ff_dim, 
        dropout=0
    ):
        """
        Args:
            head_size: dimension of model per head
            num_heads: number of heads to parallelize MHA
            ff_dim: dimension of feed forward layer
        """
        super().__init__()

        self.head_size = head_size
        self.num_heads = num_heads
        # k, q, v encoders in batch
        self.kqv = nn.Linear(self.d_model, self.d_model * 3)
        

        # Attention and Normalization
        self.multiheadattention_layer = nn.MultiheadAttention(self.d_model, num_heads, batch_first=True, dropout=dropout)
        self.dropout_layer = nn.Dropout(p = dropout)
        self.layernorm = nn.LayerNorm(self.d_model, eps=1e-6)

        # Feed Forward Part
        self.linear_layer1 = nn.Linear(self.d_model, ff_dim)
        self.gelu1 = nn.GELU()
        self.dropout_layer2 = nn.Dropout(p = dropout)
        self.linear_layer2 = nn.Linear(ff_dim, self.d_model)
        self.layernorm2 = nn.LayerNorm(self.d_model, eps=1e-6)

    @property
    def d_model(self):
        return self.head_size * self.num_heads

    def forward(self, x):
        # apply k, q, v embeddings
        q, k, v = self.kqv(x).split(self.d_model, dim=2)
    
        # mha dropout and residual connection
        o1, _ = self.multiheadattention_layer(q, k, v, need_weights=False)
        o2 = self.dropout_layer(o1)
        res = x + o2
        res = self.layernorm(res)
        

        # feed forward
        o4 = self.linear_layer1(res)
        o5 = self.gelu1(o4)
        o6 = self.dropout_layer2(o5)
        o7 = self.linear_layer2(o6)
        o8 = self.layernorm2(o7 + res)
        return o8 + res

class Transformer(nn.Module):
    def __init__(
        self, 
        input_size, 
        head_size, 
        num_heads, 
        ff_dim, 
        num_transformer_blocks, 
        mlp_units, 
        dropout=0, 
        mlp_dropout=0
    ):
        super().__init__()
        self.head_size = head_size
        self.num_heads = num_heads
        
        self.proj = nn.Linear(input_size, self.d_model)
        #self.proj = nn.Conv1d(input_size, self.d_model, stride=3, kernel_size=3)
        self.pos = LearnablePositionalEncoding(self.d_model, 3072)
        
        self.transformer_list = nn.ModuleList([
            TransformerEncoder(head_size, num_heads, ff_dim, dropout = dropout) 
            for _ in range(num_transformer_blocks)
        ])
        
        self.conv1 = nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1)
        self.pooling = nn.AdaptiveAvgPool1d(1)

        mlp_layers = []
        last_dim = head_size * num_heads
        for dim in mlp_units:
            mlp_layers.append(nn.Linear(last_dim, dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(mlp_dropout))
            last_dim = dim
        self.mlp = nn.Sequential(*mlp_layers)
        self.output_layer = nn.Linear(last_dim, 1)

    @property
    def d_model(self):
        return self.head_size * self.num_heads

    def forward(self, x):
        x = x.transpose(2, 1)
        x = self.proj(x)
        #x = x.transpose(2, 1)
        x = self.pos(x)
        for t in self.transformer_list:
            x = t(x)

        x = self.conv1(x.transpose(-2, -1))
        x = self.conv2(x)
        x = self.pooling(x).squeeze(-1)

        x = self.mlp(x)
        x = self.output_layer(x)
        return x