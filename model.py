import torch
import torch.nn as nn
import torch.nn.functional as F

from block import TransformerBlock


class GPTLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        n_embd=128,
        num_heads=4,
        num_layers=4,
        block_size=128,
        dropout=0.1,
    ):
        super().__init__()

        self.block_size = block_size

        # ------------------------------------------------
        # 1. Token embeddings
        # ------------------------------------------------
        self.token_embedding_table = nn.Embedding(
            vocab_size,
            n_embd
        )

        # ------------------------------------------------
        # 2. Positional embeddings
        # ------------------------------------------------
        self.position_embedding_table = nn.Embedding(
            block_size,
            n_embd
        )

        # ------------------------------------------------
        # 3. Transformer blocks
        # ------------------------------------------------
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    n_embd=n_embd,
                    num_heads=num_heads,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # ------------------------------------------------
        # 4. Final LayerNorm
        # ------------------------------------------------
        self.ln_f = nn.LayerNorm(n_embd)

        # ------------------------------------------------
        # 5. Language-model head
        # ------------------------------------------------
        self.lm_head = nn.Linear(
            n_embd,
            vocab_size
        )

    def forward(self, idx, targets=None):

        B, T = idx.shape

        # -----------------------------------------------
        # Token embeddings
        # -----------------------------------------------
        tok_emb = self.token_embedding_table(idx)

        # Shape:
        # (B,T) -> (B,T,C)

        # -----------------------------------------------
        # Position embeddings
        # -----------------------------------------------
        pos = torch.arange(
            T,
            device=idx.device
        )

        pos_emb = self.position_embedding_table(pos)

        # Shape:
        # (T,) -> (T,C)

        # -----------------------------------------------
        # Add token + position information
        # -----------------------------------------------
        x = tok_emb + pos_emb

        # Shape:
        # (B,T,C)

        # -----------------------------------------------
        # Transformer blocks
        # -----------------------------------------------
        x = self.blocks(x)

        # Shape:
        # (B,T,C)

        # -----------------------------------------------
        # Final normalization
        # -----------------------------------------------
        x = self.ln_f(x)

        # -----------------------------------------------
        # Convert hidden representation to logits
        # -----------------------------------------------
        logits = self.lm_head(x)

        # Shape:
        # (B,T,C) -> (B,T,vocab_size)

        loss = None

        if targets is not None:

            B, T, V = logits.shape

            logits_flat = logits.view(B * T, V)
            targets_flat = targets.view(B * T)

            loss = F.cross_entropy(
                logits_flat,
                targets_flat
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
    ):

        for _ in range(max_new_tokens):

            # Keep only the most recent context
            idx_cond = idx[:, -self.block_size:]

            # Forward pass
            logits, _ = self(idx_cond)

            # Only need predictions for final token
            logits = logits[:, -1, :]

            # Temperature
            logits = logits / temperature

            # Optional top-k filtering
            if top_k is not None:

                values, _ = torch.topk(
                    logits,
                    min(top_k, logits.size(-1))
                )

                logits[
                    logits < values[:, [-1]]
                ] = float("-inf")

            # Convert logits to probabilities
            probs = F.softmax(
                logits,
                dim=-1
            )

            # Sample next token
            idx_next = torch.multinomial(
                probs,
                num_samples=1
            )

            # Add new token
            idx = torch.cat(
                (idx, idx_next),
                dim=1
            )

        return idx

