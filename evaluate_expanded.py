import math
import torch

from model import GPTLanguageModel
from data_expanded import load_data, get_batch


# =========================================================
# Configuration
# =========================================================

BATCH_SIZE = 64
BLOCK_SIZE = 256

N_EMBD = 384
NUM_HEADS = 6
NUM_LAYERS = 6
DROPOUT = 0.2

EVAL_ITERS = 100

CHECKPOINT_PATH = (
    "/content/drive/MyDrive/"
    "gpt-from-scratch-expanded/best.pt"
)


# =========================================================
# Device
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)


# =========================================================
# Load data
# =========================================================

train_data, val_data, stoi, itos = load_data()

vocab_size = len(stoi)

print("Vocabulary size:", vocab_size)
print("Training tokens:", len(train_data))
print("Validation tokens:", len(val_data))


# =========================================================
# Create model
# =========================================================

model = GPTLanguageModel(
    vocab_size=vocab_size,
    n_embd=N_EMBD,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    block_size=BLOCK_SIZE,
    dropout=DROPOUT,
)

model = model.to(DEVICE)


# =========================================================
# Load best checkpoint
# =========================================================

print()
print("=" * 60)
print("Loading best checkpoint")
print("=" * 60)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

best_step = checkpoint["step"]
best_val_loss_from_training = checkpoint[
    "best_val_loss"
]

print("Checkpoint:", CHECKPOINT_PATH)
print("Checkpoint step:", best_step)
print(
    "Best validation loss recorded during training:",
    f"{best_val_loss_from_training:.4f}"
)


# =========================================================
# Evaluation
# =========================================================

@torch.no_grad()
def estimate_loss(data):
    model.eval()

    losses = torch.zeros(EVAL_ITERS)

    for k in range(EVAL_ITERS):

        x, y = get_batch(
            data,
            BATCH_SIZE,
            BLOCK_SIZE,
            DEVICE
        )

        _, loss = model(x, y)

        losses[k] = loss.item()

    model.train()

    return losses.mean().item()


# =========================================================
# Evaluate
# =========================================================

print()
print("=" * 60)
print("Evaluating best checkpoint")
print("=" * 60)

train_loss = estimate_loss(train_data)
val_loss = estimate_loss(val_data)

train_ppl = math.exp(train_loss)
val_ppl = math.exp(val_loss)

print()
print(f"Train loss:       {train_loss:.4f}")
print(f"Validation loss:  {val_loss:.4f}")
print(f"Train perplexity: {train_ppl:.2f}")
print(f"Val perplexity:   {val_ppl:.2f}")