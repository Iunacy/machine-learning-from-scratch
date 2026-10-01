import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        assert d_model % num_heads == 0

        self.query_layer = nn.Linear(d_model, d_model, bias=False)
        self.key_layer = nn.Linear(d_model, d_model, bias=False)
        self.value_layer = nn.Linear(d_model, d_model, bias=False)
        self.output_layer = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        h = self.num_heads
        d_k = self.d_head

        Q = self.query_layer(x)
        K = self.key_layer(x)
        V = self.value_layer(x)

        Q = Q.view(batch, seq_len, h, d_k).transpose(1, 2)
        K = K.view(batch, seq_len, h, d_k).transpose(1, 2)
        V = V.view(batch, seq_len, h, d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1))
        scaled = scores / math.sqrt(d_k)

        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool()
        scaled = scaled.masked_fill(mask, float("-inf"))

        attention_weights = F.softmax(scaled, dim=-1)

        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).reshape(batch, seq_len, d_model)
        output = self.output_layer(output)

        return output, attention_weights


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        attn_output, weights = self.attention(x)

        x = x + attn_output
        x = self.norm1(x)

        ffn_output = self.ffn(x)

        x = x + ffn_output
        x = self.norm2(x)

        return x


torch.manual_seed(42)

x = torch.randn(2, 4, 8)

block = TransformerBlock(d_model=8, num_heads=2, d_ff=32)
output = block(x)

print(x.shape)
print(output.shape)
