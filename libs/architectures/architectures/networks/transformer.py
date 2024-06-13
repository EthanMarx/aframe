import torch.nn as nn 
import torch


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size = x.size(0)
        # Linear projections
        Q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = self.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5))
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads and apply final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out(attn_output)
        
        return output
    

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x1, x2):
        batch_size = x1.size(0)
        
        # Linear projections
        Q = self.query(x1).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x2).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x2).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = self.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5))
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads and apply final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out(attn_output)
        
        return output



class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(SelfAttention, self).__init__()
        self.multihead_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.layernorm = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, x):
        attn_output = self.multihead_attn(x)
        attn_output = self.layernorm(attn_output + x)
        ff_output = self.feedforward(attn_output)
        output = self.layernorm(ff_output + attn_output)
        return output
    

class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(CrossAttention, self).__init__()
        self.multihead_attn = MultiHeadCrossAttention(d_model, num_heads)
        self.layernorm = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x1, x2):
        attn_output = self.multihead_attn(x1, x2)
        attn_output = self.layernorm(attn_output + x1)
        ff_output = self.feedforward(attn_output)
        output = self.layernorm(ff_output + attn_output)
        return output
    

class AttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super(AttentionBlock, self).__init__()
        self.self_attention = SelfAttention(d_model, num_heads)
        self.cross_attention = CrossAttention(d_model, num_heads)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x1, x2):
        # Self-attention
        sa1 = self.self_attention(x1)
        sa2 = self.self_attention(x2)
        
        # Add & Norm
        x1 = self.layernorm1(sa1 + x1)
        x2 = self.layernorm1(sa2 + x2)
        
        # Cross-attention
        ca12 = self.cross_attention(x1, x2)
        ca21 = self.cross_attention(x2, x1)
        
        # Add & Norm
        x1 = self.layernorm2(ca12 + x1)
        x2 = self.layernorm2(ca21 + x2)
        
        # Feedforward
        ff1 = self.feedforward(x1)
        ff2 = self.feedforward(x2)
        
        # Add & Norm
        x1 = self.layernorm1(ff1 + x1)
        x2 = self.layernorm1(ff2 + x2)
        
        return x1, x2
    
import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, time_steps):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        
        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(time_steps, d_model)
        position = torch.arange(0, time_steps, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1).unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
    

class Transformer(nn.Module):
    def __init__(self, time_steps, n_embd, hidden_dim, num_classes, num_heads, num_blocks=1):
        super(Transformer, self).__init__()

        # create embeddings for positional encoding
        self.embedding = nn.Conv1d(1, n_embd, kernel_size=1, stride=1, padding=0)
        self.positional_encoding = PositionalEncoding(n_embd, time_steps)

        # linear layer downsample before feeding into transformer
        self.downsample1 = nn.Linear(time_steps, n_embd)
        self.downsample2 = nn.Linear(time_steps, n_embd)
        self.layernorm1 = nn.LayerNorm(n_embd)
        self.layernorm2 = nn.LayerNorm(n_embd)

        self.attention_blocks = nn.ModuleList([AttentionBlock(n_embd, num_heads) for _ in range(num_blocks)])
        self.fc1 = nn.Linear(n_embd * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # Assume x is of shape (batch, channels, time)
        x1 = x[:, 0, :].unsqueeze(1) # Channel 1
        x2 = x[:, 1, :].unsqueeze(1)  # Channel 2
     
        
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)

        x1 = self.positional_encoding(x1)
        x2 = self.positional_encoding(x2)

        x1 = self.downsample1(x1)
        x2 = self.downsample2(x2)
        x1 = self.layernorm1(x1)
        x2 = self.layernorm2(x2)
        x1 = x1.transpose(1, 2)
        x2 = x2.transpose(1, 2)
        

        # Apply attention blocks
        for block in self.attention_blocks:
            x1, x2 = block(x1, x2)
        
        
        # Concatenate the outputs of the last attention block
        combined = torch.cat((x1, x2), dim=-1)
        
        # Classification
        combined = combined.mean(dim=1)  # Global average pooling
        x = nn.GELU()(self.fc1(combined))
        x = self.fc2(x)
        return x