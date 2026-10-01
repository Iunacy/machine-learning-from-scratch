"""A small causal multi-head self-attention and Transformer block exercise."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.query_layer = nn.Linear(d_model, d_model, bias=False)
        self.key_layer = nn.Linear(d_model, d_model, bias=False)
        self.value_layer = nn.Linear(d_model, d_model, bias=False)
        self.output_layer = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        batch_size, sequence_length, d_model = x.shape
        num_heads = self.num_heads

        queries = self.query_layer(x)
        keys = self.key_layer(x)
        values = self.value_layer(x)

        queries = queries.view(
            batch_size, sequence_length, num_heads, self.d_head
        ).transpose(1, 2)
        keys = keys.view(
            batch_size, sequence_length, num_heads, self.d_head
        ).transpose(1, 2)
        values = values.view(
            batch_size, sequence_length, num_heads, self.d_head
        ).transpose(1, 2)

        scores = queries @ keys.transpose(-2, -1)
        scaled_scores = scores / math.sqrt(self.d_head)

        causal_mask = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=x.device,
        ).triu(diagonal=1)
        scaled_scores = scaled_scores.masked_fill(causal_mask, float("-inf"))

        attention_weights = F.softmax(scaled_scores, dim=-1)
        output = attention_weights @ values
        output = output.transpose(1, 2).reshape(
            batch_size, sequence_length, d_model
        )
        output = self.output_layer(output)

        return output, attention_weights


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        attention_output, _ = self.attention(x)
        x = self.norm1(x + attention_output)

        feed_forward_output = self.feed_forward(x)
        x = self.norm2(x + feed_forward_output)

        return x


if __name__ == "__main__":
    torch.manual_seed(42)

    inputs = torch.randn(2, 4, 8)
    block = TransformerBlock(d_model=8, num_heads=2, d_ff=32)
    outputs = block(inputs)

    print("Input shape:", inputs.shape)
    print("Output shape:", outputs.shape)
