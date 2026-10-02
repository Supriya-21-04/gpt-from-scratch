import torch

from data import load_data, get_batch
from model import GPTLanguageModel


# -----------------------------
# Configuration
# -----------------------------

BATCH_SIZE = 4
BLOCK_SIZE = 64

N_EMBD = 128
NUM_HEADS = 4
NUM_LAYERS = 4
DROPOUT = 0.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Load dataset
# -----------------------------

train_data, val_data, stoi, itos = load_data()

vocab_size = len(stoi)

print("Vocabulary size:", vocab_size)
print("Device:", DEVICE)


# -----------------------------
# Get a real training batch
# -----------------------------

x, y = get_batch(
    train_data,
    batch_size=BATCH_SIZE,
    block_size=BLOCK_SIZE,
    device=DEVICE,
)

print("Input shape:", x.shape)
print("Target shape:", y.shape)


# -----------------------------
# Create GPT model
# -----------------------------

model = GPTLanguageModel(
    vocab_size=vocab_size,
    n_embd=N_EMBD,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    block_size=BLOCK_SIZE,
    dropout=DROPOUT,
).to(DEVICE)


# -----------------------------
# Count parameters
# -----------------------------

num_params = sum(
    p.numel()
    for p in model.parameters()
)

print("Number of parameters:", num_params)


# -----------------------------
# Forward pass
# -----------------------------

logits, loss = model(
    x,
    y
)


print("Logits shape:", logits.shape)
print("Loss:", loss.item())


# -----------------------------
# Verify everything
# -----------------------------

assert x.shape == (
    BATCH_SIZE,
    BLOCK_SIZE
)

assert y.shape == (
    BATCH_SIZE,
    BLOCK_SIZE
)

assert logits.shape == (
    BATCH_SIZE,
    BLOCK_SIZE,
    vocab_size
)

assert loss.ndim == 0


print("Full GPT forward pass test passed.")

