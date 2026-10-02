import torch

from block import TransformerBlock


B = 2
T = 8
C = 32

num_heads = 4
block_size = 16
dropout = 0.0


x = torch.randn(B, T, C)

print("Input shape:", x.shape)


block = TransformerBlock(
    n_embd=C,
    num_heads=num_heads,
    block_size=block_size,
    dropout=dropout,
)


out = block(x)

print("Output shape:", out.shape)


assert out.shape == (B, T, C)

print("Transformer block test passed.")

