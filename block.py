import torch
import torch.nn as nn

from attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    A simple feed-forward neural network.

    Input:
        (B, T, C)

    Output:
        (B, T, C)
    """

    def __init__(self, n_embd, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One decoder-style Transformer block.

    Input:
        (B, T, C)

    Output:
        (B, T, C)
    """

    def __init__(self, n_embd, num_heads, block_size, dropout):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)

        self.attention = MultiHeadAttention(
            n_embd=n_embd,
            num_heads=num_heads,
            block_size=block_size,
            dropout=dropout,
        )

        self.ln2 = nn.LayerNorm(n_embd)

        self.ffwd = FeedForward(
            n_embd=n_embd,
            dropout=dropout,
        )

    def forward(self, x):

        # Attention + residual connection
        x = x + self.attention(self.ln1(x))

        # Feed-forward network + residual connection
        x = x + self.ffwd(self.ln2(x))

        return x

