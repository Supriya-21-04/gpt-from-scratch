import math
import torch
import torch.nn.functional as F

from data import load_data, get_batch
from model import GPTLanguageModel


# =========================
# Configuration
# =========================

BATCH_SIZE = 32
BLOCK_SIZE = 128

N_EMBD = 128
NUM_HEADS = 4
NUM_LAYERS = 4
DROPOUT = 0.1

MAX_ITERS = 20000

LEARNING_RATE = 3e-4
MIN_LR = 3e-5

WARMUP_ITERS = 200

EVAL_INTERVAL = 1000
EVAL_ITERS = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_PATH = "checkpoints/gpt_checkpoint.pt"


# =========================
# Reproducibility
# =========================

torch.manual_seed(1337)


# =========================
# Load data
# =========================

train_data, val_data, stoi, itos = load_data()

vocab_size = len(stoi)

print("Device:", DEVICE)
print("Vocabulary size:", vocab_size)


# =========================
# Model
# =========================

model = GPTLanguageModel(
    vocab_size=vocab_size,
    n_embd=N_EMBD,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    block_size=BLOCK_SIZE,
    dropout=DROPOUT,
).to(DEVICE)

num_params = sum(p.numel() for p in model.parameters())

print("Number of parameters:", f"{num_params:,}")


# =========================
# Optimizer
# =========================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)


# =========================
# Learning-rate schedule
# =========================

def get_learning_rate(step):

    # Warmup
    if step < WARMUP_ITERS:
        return LEARNING_RATE * (step + 1) / WARMUP_ITERS

    # After warmup
    if step >= MAX_ITERS:
        return MIN_LR

    # Progress through cosine decay
    decay_ratio = (
        (step - WARMUP_ITERS)
        / (MAX_ITERS - WARMUP_ITERS)
    )

    coeff = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )

    return MIN_LR + coeff * (
        LEARNING_RATE - MIN_LR
    )


# =========================
# Estimate loss
# =========================

@torch.no_grad()
def estimate_loss():

    model.eval()

    results = {}

    for split, data in [
        ("train", train_data),
        ("val", val_data),
    ]:

        losses = torch.zeros(EVAL_ITERS)

        for k in range(EVAL_ITERS):

            x, y = get_batch(
                data,
                batch_size=BATCH_SIZE,
                block_size=BLOCK_SIZE,
                device=DEVICE,
            )

            _, loss = model(x, y)

            losses[k] = loss.item()

        results[split] = losses.mean().item()

    model.train()

    return results


# =========================
# Training loop
# =========================

print("\nStarting training...\n")

for step in range(MAX_ITERS):

    # Update learning rate
    lr = get_learning_rate(step)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # Evaluate periodically
    if step % EVAL_INTERVAL == 0:

        losses = estimate_loss()

        print(
            f"Step {step:4d} | "
            f"LR {lr:.6f} | "
            f"Train loss {losses['train']:.4f} | "
            f"Val loss {losses['val']:.4f}"
        )

    # Get training batch
    x, y = get_batch(
        train_data,
        batch_size=BATCH_SIZE,
        block_size=BLOCK_SIZE,
        device=DEVICE,
    )

    # Forward pass
    logits, loss = model(x, y)

    # Clear old gradients
    optimizer.zero_grad(set_to_none=True)

    # Backpropagation
    loss.backward()

    # Update parameters
    optimizer.step()


# =========================
# Final evaluation
# =========================

losses = estimate_loss()

print("\nTraining complete.")

print(
    f"Final train loss: {losses['train']:.4f}"
)

print(
    f"Final validation loss: {losses['val']:.4f}"
)


# =========================
# Save checkpoint
# =========================

checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "stoi": stoi,
    "itos": itos,
    "config": {
        "vocab_size": vocab_size,
        "n_embd": N_EMBD,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "block_size": BLOCK_SIZE,
        "dropout": DROPOUT,
    },
}

torch.save(checkpoint, CHECKPOINT_PATH)

print(f"Checkpoint saved to: {CHECKPOINT_PATH}")