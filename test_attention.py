import torch
from attention import MultiHeadAttention


# Small dimensions just for testing
B = 2
T = 8
C = 32

num_heads = 4
block_size = 16
dropout = 0.0


# Random input representing token embeddings
x = torch.randn(B, T, C)

print("Input shape:", x.shape)


attention = MultiHeadAttention(
    n_embd=C,
    num_heads=num_heads,
    block_size=block_size,
    dropout=dropout
)


out = attention(x)

print("Output shape:", out.shape)

assert out.shape == (B, T, C)

print("Attention test passed.")

