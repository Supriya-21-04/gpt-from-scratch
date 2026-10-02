import math
import torch

from data import load_data, get_batch
from model import GPTLanguageModel


# =========================
# Configuration
# =========================

BATCH_SIZE = 32
EVAL_ITERS = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_PATH = "checkpoints/gpt_checkpoint.pt"


# =========================
# Load data
# =========================

train_data, val_data, stoi, itos = load_data()

vocab_size = len(stoi)


# =========================
# Load checkpoint
# =========================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
)

config = checkpoint["config"]


model = GPTLanguageModel(
    vocab_size=config["vocab_size"],
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


# =========================
# Evaluate
# =========================

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

        losses.append(loss.item())

    return sum(losses) / len(losses)


train_loss = evaluate(train_data)
val_loss = evaluate(val_data)

train_perplexity = math.exp(train_loss)
val_perplexity = math.exp(val_loss)


print(f"Train loss:       {train_loss:.4f}")
print(f"Train perplexity: {train_perplexity:.2f}")

print()

print(f"Val loss:         {val_loss:.4f}")
print(f"Val perplexity:   {val_perplexity:.2f}")