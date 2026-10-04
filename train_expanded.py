import math
import random
from pathlib import Path

import numpy as np
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

MAX_ITERS = 20000

LEARNING_RATE = 3e-4
MIN_LR = 3e-5
WARMUP_ITERS = 500

EVAL_INTERVAL = 500
EVAL_ITERS = 100

# Stop if validation loss fails to improve
# by at least MIN_DELTA for PATIENCE evaluations.
PATIENCE = 5
MIN_DELTA = 0.002

SEED = 1337


# =========================================================
# Device
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)


# =========================================================
# Reproducibility
# =========================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================================================
# Google Drive checkpoint directory
# =========================================================

DRIVE_CHECKPOINT_DIR = Path(
    "/content/drive/MyDrive/gpt-from-scratch-expanded"
)

DRIVE_CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LATEST_CHECKPOINT = (
    DRIVE_CHECKPOINT_DIR / "latest.pt"
)

BEST_CHECKPOINT = (
    DRIVE_CHECKPOINT_DIR / "best.pt"
)


# =========================================================
# Load data
# =========================================================

train_data, val_data, stoi, itos = load_data()

VOCAB_SIZE = len(stoi)

print("Vocabulary size:", VOCAB_SIZE)
print("Training tokens:", len(train_data))
print("Validation tokens:", len(val_data))


# =========================================================
# Model
# =========================================================

model = GPTLanguageModel(
    vocab_size=VOCAB_SIZE,
    n_embd=N_EMBD,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    block_size=BLOCK_SIZE,
    dropout=DROPOUT,
).to(DEVICE)


# =========================================================
# Optimizer
# =========================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================================================
# Mixed precision
# =========================================================

use_amp = DEVICE == "cuda"

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)


# =========================================================
# Parameter count
# =========================================================

num_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Number of parameters:", f"{num_params:,}")


# =========================================================
# Learning-rate schedule
# =========================================================

def get_learning_rate(step):

    # Warmup
    if step < WARMUP_ITERS:
        return LEARNING_RATE * (
            (step + 1) / WARMUP_ITERS
        )

    # After warmup: cosine decay
    progress = (
        step - WARMUP_ITERS
    ) / (
        MAX_ITERS - WARMUP_ITERS
    )

    progress = min(max(progress, 0.0), 1.0)

    cosine = 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )

    return MIN_LR + (
        LEARNING_RATE - MIN_LR
    ) * cosine


# =========================================================
# Evaluation
# =========================================================

@torch.no_grad()
def estimate_loss():

    model.eval()

    results = {}

    for split, data in [
        ("train", train_data),
        ("val", val_data)
    ]:

        losses = torch.zeros(
            EVAL_ITERS
        )

        for k in range(EVAL_ITERS):

            x, y = get_batch(
                data,
                BATCH_SIZE,
                BLOCK_SIZE,
                DEVICE
            )

            with torch.amp.autocast(
                device_type="cuda",
                enabled=use_amp
            ):

                _, loss = model(
                    x,
                    y
                )

            losses[k] = loss.item()

        results[split] = losses.mean().item()

    model.train()

    return results


# =========================================================
# Checkpoint helpers
# =========================================================

def save_checkpoint(
    path,
    step,
    best_val_loss,
    patience_counter
):

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),

        "step": step,

        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,

        "config": {
            "vocab_size": VOCAB_SIZE,
            "n_embd": N_EMBD,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "block_size": BLOCK_SIZE,
            "dropout": DROPOUT,
        },

        "stoi": stoi,
        "itos": itos,

        "rng_state": torch.get_rng_state(),

        "python_random_state": random.getstate(),

        "numpy_random_state": np.random.get_state(),
    }

    if torch.cuda.is_available():
        checkpoint["cuda_rng_state"] = (
            torch.cuda.get_rng_state_all()
        )

    torch.save(
        checkpoint,
        path
    )
from pathlib import Path

DRIVE_DIR = Path(
    "/content/drive/MyDrive/gpt-from-scratch-expanded"
)

DRIVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# Resume from checkpoint
# =========================================================

checkpoint_path = DRIVE_DIR / "latest.pt"

# =========================================================
# Resume from checkpoint
# =========================================================

checkpoint_path = DRIVE_DIR / "latest.pt"

if checkpoint_path.exists():

    print()
    print("=" * 60)
    print("Found existing checkpoint.")
    print(f"Loading: {checkpoint_path}")
    print("=" * 60)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    # -----------------------------------------------------
    # Restore model
    # -----------------------------------------------------

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # -----------------------------------------------------
    # Restore optimizer
    # -----------------------------------------------------

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    # -----------------------------------------------------
    # Restore scaler
    # -----------------------------------------------------

    if use_amp and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

    # -----------------------------------------------------
    # Restore training state
    # -----------------------------------------------------

    completed_step = checkpoint["step"]

    start_step = completed_step + 1

    best_val_loss = checkpoint[
        "best_val_loss"
    ]

    patience_counter = checkpoint[
        "patience_counter"
    ]

    # -----------------------------------------------------
    # Restore RNG states if compatible
    #
    # RNG restoration is useful for exact reproducibility,
    # but should never prevent a valid checkpoint from
    # resuming training.
    # -----------------------------------------------------

    try:

        rng_state = checkpoint.get(
            "rng_state"
        )

        if rng_state is not None:

            if not isinstance(
                rng_state,
                torch.ByteTensor
            ):
                rng_state = torch.as_tensor(
                    rng_state,
                    dtype=torch.uint8
                )

            torch.set_rng_state(
                rng_state
            )

    except Exception as e:

        print(
            "Warning: could not restore "
            "PyTorch RNG state."
        )

        print(
            "Training will continue with "
            "a new RNG state."
        )

        print(
            "Reason:",
            e
        )

    # -----------------------------------------------------
    # Restore Python random state
    # -----------------------------------------------------

    try:

        random_state = checkpoint.get(
            "python_random_state"
        )

        if random_state is not None:

            random.setstate(
                random_state
            )

    except Exception as e:

        print(
            "Warning: could not restore "
            "Python RNG state:",
            e
        )

    # -----------------------------------------------------
    # Restore NumPy random state
    # -----------------------------------------------------

    try:

        numpy_state = checkpoint.get(
            "numpy_random_state"
        )

        if numpy_state is not None:

            np.random.set_state(
                numpy_state
            )

    except Exception as e:

        print(
            "Warning: could not restore "
            "NumPy RNG state:",
            e
        )

    # -----------------------------------------------------
    # Restore CUDA RNG state
    # -----------------------------------------------------

    try:

        if (
            torch.cuda.is_available()
            and "cuda_rng_state" in checkpoint
        ):

            cuda_rng_state = checkpoint[
                "cuda_rng_state"
            ]

            torch.cuda.set_rng_state_all(
                cuda_rng_state
            )

    except Exception as e:

        print(
            "Warning: could not restore "
            "CUDA RNG state:",
            e
        )

    # -----------------------------------------------------
    # Resume information
    # -----------------------------------------------------

    print()

    print(
        f"Resumed from completed step "
        f"{completed_step}"
    )

    print(
        f"Next step: {start_step}"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )

else:

    print()
    print("No checkpoint found.")
    print("Starting training from scratch.")
# =========================================================
# Training
# =========================================================

print()
print("=" * 70)
print("STARTING EXPANDED-CORPUS TRAINING")
print("=" * 70)

for step in range(
    start_step,
    MAX_ITERS
):

    # -----------------------------------------------------
    # Learning rate
    # -----------------------------------------------------

    lr = get_learning_rate(step)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    if step % EVAL_INTERVAL == 0:

        losses = estimate_loss()

        train_loss = losses["train"]
        val_loss = losses["val"]

        print(
            f"Step {step:5d} | "
            f"LR {lr:.6f} | "
            f"Train {train_loss:.4f} | "
            f"Val {val_loss:.4f}"
        )

        # -------------------------------------------------
        # Best checkpoint
        # -------------------------------------------------

        improvement = (
            best_val_loss - val_loss
        )

        if improvement > MIN_DELTA:

            best_val_loss = val_loss

            patience_counter = 0

            save_checkpoint(
                BEST_CHECKPOINT,
                step,
                best_val_loss,
                patience_counter
            )

            print(
                f"  → New best validation loss: "
                f"{best_val_loss:.4f}"
            )

        else:

            patience_counter += 1

            print(
                f"  → No significant improvement "
                f"({patience_counter}/{PATIENCE})"
            )


        # -------------------------------------------------
        # Early stopping
        # -------------------------------------------------

        if patience_counter >= PATIENCE:

            print()
            print(
                "Early stopping triggered."
            )

            break


    # -----------------------------------------------------
    # Training batch
    # -----------------------------------------------------

    x, y = get_batch(
        train_data,
        BATCH_SIZE,
        BLOCK_SIZE,
        DEVICE
    )


    # -----------------------------------------------------
    # Forward + backward
    # -----------------------------------------------------

    with torch.amp.autocast(
        device_type="cuda",
        enabled=use_amp
    ):

        logits, loss = model(
            x,
            y
        )

    optimizer.zero_grad(
        set_to_none=True
    )

    scaler.scale(loss).backward()

    scaler.step(
        optimizer
    )

    scaler.update()


    # -----------------------------------------------------
    # Save latest checkpoint
    #
    # step is the completed optimizer step.
    # -----------------------------------------------------

    completed_step = step

    if (
        completed_step + 1
    ) % 250 == 0:

        save_checkpoint(
            LATEST_CHECKPOINT,
            completed_step,
            best_val_loss,
            patience_counter
        )

        print(
            f"  Checkpoint saved at "
            f"step {completed_step}"
        )


# =========================================================
# Final evaluation
# =========================================================

losses = estimate_loss()

final_train_loss = losses["train"]
final_val_loss = losses["val"]

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Final train loss: {final_train_loss:.4f}"
)

print(
    f"Final validation loss: "
    f"{final_val_loss:.4f}"
)

print(
    f"Best validation loss: "
    f"{best_val_loss:.4f}"
)

print(
    f"Train perplexity: "
    f"{math.exp(final_train_loss):.2f}"
)

print(
    f"Validation perplexity: "
    f"{math.exp(final_val_loss):.2f}"
)

print()
print(
    "Best checkpoint:",
    BEST_CHECKPOINT
)

print(
    "Latest checkpoint:",
    LATEST_CHECKPOINT
)