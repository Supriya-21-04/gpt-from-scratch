import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """
    One self-attention head.

    Each head independently learns:
        Q = query
        K = key
        V = value

    Input:
        (B, T, C)

    Output:
        (B, T, head_size)
    """

    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()

        # Learnable projections.
        #
        # C -> head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Causal mask.
        #
        # tril() creates:
        #
        # [[1, 0, 0],
        #  [1, 1, 0],
        #  [1, 1, 1]]
        #
        # This prevents tokens from looking into the future.
        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(block_size, block_size)
            )
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B, T, C = x.shape

        # -------------------------------------------------
        # Create Q, K, V
        # -------------------------------------------------

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # Shapes:
        #
        # q = (B, T, head_size)
        # k = (B, T, head_size)
        # v = (B, T, head_size)

        # -------------------------------------------------
        # Attention scores
        # -------------------------------------------------

        # k.transpose(-2, -1):
        #
        # (B, T, head_size)
        #       ↓
        # (B, head_size, T)
        #
        # q @ k^T:
        #
        # (B, T, head_size)
        # @
        # (B, head_size, T)
        #
        # =
        # (B, T, T)

        wei = q @ k.transpose(-2, -1)

        # Scale the scores.
        #
        # Without scaling, large head dimensions can
        # produce overly large dot products.
        wei = wei * (
            k.size(-1) ** -0.5
        )

        # -------------------------------------------------
        # Causal masking
        # -------------------------------------------------

        # Only allow the current position and previous
        # positions to be visible.
        #
        # Future positions receive -inf.
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        # -------------------------------------------------
        # Convert scores to probabilities
        # -------------------------------------------------

        wei = F.softmax(
            wei,
            dim=-1
        )

        # Dropout randomly removes some attention
        # connections during training to reduce overfitting.
        wei = self.dropout(wei)

        # -------------------------------------------------
        # Weighted combination of values
        # -------------------------------------------------

        # wei:
        # (B, T, T)
        #
        # v:
        # (B, T, head_size)
        #
        # result:
        # (B, T, head_size)

        out = wei @ v

        return out


class MultiHeadAttention(nn.Module):
    """
    Multiple independent self-attention heads.

    Each head learns a different projection of the
    input representation.
    """

    def __init__(
        self,
        n_embd,
        num_heads,
        block_size,
        dropout
    ):
        super().__init__()

        assert n_embd % num_heads == 0

        head_size = n_embd // num_heads

        self.heads = nn.ModuleList(
            [
                Head(
                    n_embd,
                    head_size,
                    block_size,
                    dropout
                )
                for _ in range(num_heads)
            ]
        )

        # After concatenating all heads:
        #
        # head_size * num_heads = n_embd
        #
        # This projection lets the model mix information
        # across the different heads.
        self.proj = nn.Linear(
            n_embd,
            n_embd
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # Each head produces:
        #
        # (B, T, head_size)
        #
        # Concatenating along the last dimension gives:
        #
        # (B, T, n_embd)

        out = torch.cat(
            [
                head(x)
                for head in self.heads
            ],
            dim=-1
        )

        # Mix the information from the heads.
        out = self.proj(out)

        out = self.dropout(out)

        return out