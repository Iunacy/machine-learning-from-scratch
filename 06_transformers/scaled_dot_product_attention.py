"""A small single-head scaled dot-product attention exercise."""

import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(42)
X = torch.randn(3, 4)

d_model = 4
d_k = 2

query_layer = nn.Linear(d_model, d_k, bias=False)
key_layer = nn.Linear(d_model, d_k, bias=False)
value_layer = nn.Linear(d_model, d_k, bias=False)

Q = query_layer(X)
K = key_layer(X)
V = value_layer(X)

scores = Q @ K.T
scaled_scores = scores / (d_k**0.5)
attention_weights = F.softmax(scaled_scores, dim=-1)

output = attention_weights @ V

print("Query weights:\n", query_layer.weight)
print("Key weights:\n", key_layer.weight)
print("Value weights:\n", value_layer.weight)

print("Q:\n", Q)
print("K:\n", K)
print("V:\n", V)

print("Attention:\n", attention_weights)
print("Output:\n", output)

optimizer = torch.optim.Adam(
    list(query_layer.parameters())
    + list(key_layer.parameters())
    + list(value_layer.parameters()),
    lr=0.001,
)

target = torch.tensor([0, 1, 0], dtype=torch.long)

criterion = nn.CrossEntropyLoss()
loss = criterion(output, target)

print("\nLoss:", loss.item())

optimizer.zero_grad()
loss.backward()
optimizer.step()

print("\nWeight update complete.")
