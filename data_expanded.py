from pathlib import Path
import torch


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_PATH = Path(__file__).parent / "data" / "corpus_expanded_clean.txt"

TRAIN_FRACTION = 0.90


# ---------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------

def build_vocabulary(text):
    chars = sorted(set(text))

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    return stoi, itos


# ---------------------------------------------------------
# Encoding / decoding
# ---------------------------------------------------------

def encode(text, stoi):
    return [stoi[ch] for ch in text]


def decode(ids, itos):
    return "".join(itos[i] for i in ids)


# ---------------------------------------------------------
# Load expanded corpus
# ---------------------------------------------------------

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. "
            "Run data/clean_expanded.py first."
        )

    text = DATA_PATH.read_text(encoding="utf-8")

    stoi, itos = build_vocabulary(text)

    encoded = encode(text, stoi)

    data = torch.tensor(
        encoded,
        dtype=torch.long
    )

    split_index = int(
        TRAIN_FRACTION * len(data)
    )

    train_data = data[:split_index]
    val_data = data[split_index:]

    return (
        train_data,
        val_data,
        stoi,
        itos
    )


# ---------------------------------------------------------
# Batch generation
# ---------------------------------------------------------

def get_batch(
    data,
    batch_size,
    block_size,
    device
):

    starts = torch.randint(
        low=0,
        high=len(data) - block_size,
        size=(batch_size,)
    )

    x = torch.stack([
        data[i:i + block_size]
        for i in starts
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in starts
    ])

    x = x.to(device)
    y = y.to(device)

    return x, y