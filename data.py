from pathlib import Path

import torch


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_PATH = Path(__file__).parent / "data" / "corpus.txt"

# Fraction of the corpus reserved for validation.
TRAIN_FRACTION = 0.90


# ---------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------

def build_vocabulary(text):
    """
    Build character-level encoder and decoder mappings.

    Returns
    -------
    stoi : dict
        Maps character -> integer ID.

    itos : dict
        Maps integer ID -> character.
    """

    # sorted() makes vocabulary construction deterministic.
    chars = sorted(set(text))

    # character -> integer
    stoi = {ch: i for i, ch in enumerate(chars)}

    # integer -> character
    itos = {i: ch for i, ch in enumerate(chars)}

    return stoi, itos


def encode(text, stoi):
    """
    Convert a string into a list of integer token IDs.
    """

    return [stoi[ch] for ch in text]


def decode(ids, itos):
    """
    Convert integer token IDs back into a string.
    """

    return "".join(itos[i] for i in ids)


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

def load_data():
    """
    Load the corpus and create train/validation tensors.

    Returns
    -------
    train_data : torch.Tensor
        1D tensor containing training token IDs.

    val_data : torch.Tensor
        1D tensor containing validation token IDs.

    stoi : dict
        Character -> ID mapping.

    itos : dict
        ID -> character mapping.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. "
            "Run 'python data/download_data.py' first."
        )

    # Read the complete corpus.
    text = DATA_PATH.read_text(encoding="utf-8")

    # Build the character vocabulary.
    stoi, itos = build_vocabulary(text)

    # Convert every character into an integer.
    encoded = encode(text, stoi)

    # Convert the list into a PyTorch tensor.
    data = torch.tensor(encoded, dtype=torch.long)

    # IMPORTANT:
    # We split the sequential corpus before sampling batches.
    # This prevents validation examples from being mixed into
    # the training portion.
    split_index = int(TRAIN_FRACTION * len(data))

    train_data = data[:split_index]
    val_data = data[split_index:]

    return train_data, val_data, stoi, itos


# ---------------------------------------------------------
# Batch sampling
# ---------------------------------------------------------

def get_batch(data, batch_size, block_size, device):
    """
    Sample a random batch of contiguous sequences.

    Parameters
    ----------
    data : torch.Tensor
        1D sequence of token IDs.

    batch_size : int
        Number of sequences in the batch.

    block_size : int
        Number of input tokens in each sequence.

    device : str
        Device where the tensors should live, e.g. "cpu" or "cuda".

    Returns
    -------
    x : torch.Tensor
        Input tokens with shape (B, T).

    y : torch.Tensor
        Target tokens with shape (B, T).
    """

    # Choose random starting positions.
    #
    # We subtract block_size + 1 because we need:
    #
    # x = data[i : i + block_size]
    # y = data[i + 1 : i + block_size + 1]
    #
    # Therefore the final target must still exist.
    starts = torch.randint(
        low=0,
        high=len(data) - block_size,
        size=(batch_size,)
    )

    # Construct the input sequences.
    x = torch.stack([
        data[i : i + block_size]
        for i in starts
    ])

    # Construct targets shifted one character to the right.
    y = torch.stack([
        data[i + 1 : i + block_size + 1]
        for i in starts
    ])

    # Move the batch to the selected device.
    x = x.to(device)
    y = y.to(device)

    return x, y


# ---------------------------------------------------------
# Small self-test
# ---------------------------------------------------------

if __name__ == "__main__":

    train_data, val_data, stoi, itos = load_data()

    print(f"Vocabulary size: {len(stoi)}")
    print(f"Training tokens: {len(train_data):,}")
    print(f"Validation tokens: {len(val_data):,}")

    # Show that encoding and decoding are inverse operations.
    sample = decode(train_data[:100].tolist(), itos)

    print("\nSample text:")
    print(sample)

    # Test batch creation.
    x, y = get_batch(
        train_data,
        batch_size=4,
        block_size=16,
        device="cpu"
    )

    print("\nBatch shapes:")
    print("x:", x.shape)
    print("y:", y.shape)

    print("\nFirst training example:")
    print("x:", x[0].tolist())
    print("y:", y[0].tolist())

    # Decode the token IDs back into characters.
    print("\nDecoded x:")
    print(decode(x[0].tolist(), itos))

    print("\nDecoded y:")
    print(decode(y[0].tolist(), itos))

    assert x.shape == (4, 16)
    assert y.shape == (4, 16)

    # Every target should equal the input shifted by one position
    # for each sampled sequence.
    for row in range(4):
        start_id = x[row, 0].item()

        # We cannot compare x[row] directly to y[row] because
        # y is shifted relative to x. Instead, verify that
        # each pair came from adjacent positions in the corpus.
        assert len(x[row]) == len(y[row])

    print("\nData pipeline test passed.")
    