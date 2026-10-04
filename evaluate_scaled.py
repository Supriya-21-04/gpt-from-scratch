import math
from pathlib import Path

import torch

from data import load_data, get_batch
from model import GPTLanguageModel


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 64
EVAL_ITERS = 100

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

CHECKPOINT_PATH = Path(
    "/content/drive/MyDrive/"
    "gpt-from-scratch-checkpoints/"
    "gpt_scaled_best.pt"
)


# ============================================================
# LOAD DATA
# ============================================================

train_data, val_data, stoi, itos = load_data()

vocab_size = len(stoi)

print(f"Device: {DEVICE}")
print(f"Vocabulary size: {vocab_size}")
print(f"Checkpoint: {CHECKPOINT_PATH}")


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

config = checkpoint["config"]


print(
    f"Checkpoint step: "
    f"{checkpoint['step']}"
)


# ============================================================
# BUILD MODEL
# ============================================================

model = GPTLanguageModel(
    vocab_size=vocab_size,
    n_embd=config["n_embd"],
    num_heads=config["num_heads"],
    num_layers=config["num_layers"],
    block_size=config["block_size"],
    dropout=config["dropout"],
).to(DEVICE)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(data):

    losses = []

    for _ in range(EVAL_ITERS):

        x, y = get_batch(
            data,
            batch_size=BATCH_SIZE,
            block_size=config["block_size"],
            device=DEVICE,
        )

        _, loss = model(x, y)

        losses.append(
            loss.item()
        )

    return sum(losses) / len(losses)


train_loss = evaluate(
    train_data
)

val_loss = evaluate(
    val_data
)

train_perplexity = math.exp(
    train_loss
)

val_perplexity = math.exp(
    val_loss
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("SCALED GPT EVALUATION")
print("=" * 60)

print(
    f"Train loss:       "
    f"{train_loss:.4f}"
)

print(
    f"Train perplexity: "
    f"{train_perplexity:.2f}"
)

print()

print(
    f"Val loss:         "
    f"{val_loss:.4f}"
)

print(
    f"Val perplexity:   "
    f"{val_perplexity:.2f}"
)

print("=" * 60)