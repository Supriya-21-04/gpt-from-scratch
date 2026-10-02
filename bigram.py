import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    """
    A very small character-level language model.

    For each input character, the model directly stores
    a learned vector of logits for the next character.
    """

    def __init__(self, vocab_size):
        super().__init__()

        # This is the entire model.
        #
        # Row i corresponds to input character i.
        # Column j represents the score for predicting character j.
        #
        # Shape:
        # (vocab_size, vocab_size)
        self.token_embedding_table = nn.Embedding(
            vocab_size,
            vocab_size
        )

    def forward(self, idx, targets=None):
        """
        Parameters
        ----------
        idx : Tensor
            Input token IDs.
            Shape: (B, T)

        targets : Tensor or None
            Correct next-token IDs.
            Shape: (B, T)

        Returns
        -------
        logits : Tensor
            Raw prediction scores.
            Shape: (B, T, V)

        loss : Tensor or None
            Cross-entropy loss if targets were provided.
        """

        # Look up the row associated with every input character.
        #
        # idx shape:
        # (B, T)
        #
        # logits shape:
        # (B, T, V)
        logits = self.token_embedding_table(idx)

        loss = None

        if targets is not None:

            # CrossEntropyLoss expects:
            #
            # logits:
            # (number_of_predictions, V)
            #
            # targets:
            # (number_of_predictions,)
            #
            # So we flatten B and T together.
            B, T, V = logits.shape

            logits_flat = logits.view(B * T, V)
            targets_flat = targets.view(B * T)

            loss = F.cross_entropy(
                logits_flat,
                targets_flat
            )

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Generate new characters autoregressively.

        idx:
            Starting sequence of token IDs.
            Shape: (B, T)

        Returns:
            Extended sequence.
        """

        for _ in range(max_new_tokens):

            # Get predictions for the current sequence.
            logits, _ = self(idx)

            # We only care about the prediction at the
            # final position.
            #
            # logits shape:
            # (B, T, V)
            #
            # logits[:, -1, :] shape:
            # (B, V)
            logits = logits[:, -1, :]

            # Convert logits into probabilities.
            probs = F.softmax(logits, dim=-1)

            # Randomly sample the next character according
            # to the predicted probability distribution.
            idx_next = torch.multinomial(
                probs,
                num_samples=1
            )

            # Append the new token to the sequence.
            #
            # idx shape:
            # (B, T)
            # idx_next shape:
            # (B, 1)
            #
            # Result:
            # (B, T + 1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx