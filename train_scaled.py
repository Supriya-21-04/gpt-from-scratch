from pathlib import Path
import math
import torch
import torch.nn as nn

from data import load_data, get_batch
from model import GPTLanguageModel


# ============================================================
# SCALED MODEL CONFIGURATION
# ============================================================

BATCH_SIZE = 64
BLOCK_SIZE = 256

N_EMBD = 384
NUM_HEADS = 6
NUM_LAYERS = 6

DROPOUT = 0.2

LEARNING_RATE = 3e-4
MIN_LR = 3e-5
WARMUP_ITERS = 500

# This is only a safety ceiling.
# Training should stop based on validation behavior.
MAX_ITERS = 50

EVAL_INTERVAL = 10
EVAL_ITERS = 10

PATIENCE = 5
MIN_DELTA = 0.002

SEED = 1337

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

LATEST_CHECKPOINT = CHECKPOINT_DIR / "gpt_scaled_latest.pt"
BEST_CHECKPOINT = CHECKPOINT_DIR / "gpt_scaled_best.pt"


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# LEARNING RATE SCHEDULE
# ============================================================

def get_learning_rate(step):
    # Linear warmup
    if step < WARMUP_ITERS:
        return LEARNING_RATE * (step + 1) / WARMUP_ITERS

    # Cosine decay
    progress = (step - WARMUP_ITERS) / (
        MAX_ITERS - WARMUP_ITERS
    )

    progress = min(progress, 1.0)

    cosine = 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )

    return MIN_LR + (
        LEARNING_RATE - MIN_LR
    ) * cosine


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def estimate_loss(model, train_data, val_data):

    model.eval()

    results = {}

    for name, data in [
        ("train", train_data),
        ("val", val_data)
    ]:

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

        results[name] = losses.mean().item()

    model.train()

    return results


# ============================================================
# CHECKPOINTING
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    step,
    best_val_loss
):

    checkpoint = {
        "step": step,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scaler_state_dict":
            scaler.state_dict(),

        "best_val_loss":
            best_val_loss,

        "config": {
            "batch_size": BATCH_SIZE,
            "block_size": BLOCK_SIZE,
            "n_embd": N_EMBD,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "min_lr": MIN_LR,
            "warmup_iters": WARMUP_ITERS,
        },

        "rng_state":
            torch.get_rng_state(),

        "cuda_rng_state":
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
    }

    torch.save(checkpoint, path)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SCALED GPT TRAINING")
    print("=" * 70)

    print(f"Device: {DEVICE}")

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    train_data, val_data, stoi, itos = load_data()

    vocab_size = len(stoi)

    print(f"Vocabulary size: {vocab_size}")
    print(f"Training tokens: {len(train_data):,}")
    print(f"Validation tokens: {len(val_data):,}")

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=N_EMBD,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        block_size=BLOCK_SIZE,
        dropout=DROPOUT,
    ).to(DEVICE)

    num_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"Number of parameters: {num_parameters:,}")

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # MIXED PRECISION
    # --------------------------------------------------------

    use_amp = DEVICE == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    print(f"Mixed precision: {use_amp}")

    # --------------------------------------------------------
    # TRAINING STATE
    # --------------------------------------------------------

    best_val_loss = float("inf")
    evaluations_without_improvement = 0

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\nStarting training...\n")

    for step in range(MAX_ITERS):

        # --------------------------------------------
        # Learning rate
        # --------------------------------------------

        lr = get_learning_rate(step)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # --------------------------------------------
        # Evaluation
        # --------------------------------------------

        if step % EVAL_INTERVAL == 0:

            losses = estimate_loss(
                model,
                train_data,
                val_data
            )

            train_loss = losses["train"]
            val_loss = losses["val"]

            print(
                f"Step {step:5d} | "
                f"LR {lr:.6f} | "
                f"Train loss {train_loss:.4f} | "
                f"Val loss {val_loss:.4f}"
            )

            # ----------------------------------------
            # Best model
            # ----------------------------------------

            if val_loss < best_val_loss - MIN_DELTA:

                best_val_loss = val_loss
                evaluations_without_improvement = 0

                save_checkpoint(
                    BEST_CHECKPOINT,
                    model,
                    optimizer,
                    scaler,
                    step,
                    best_val_loss
                )

                print(
                    f"  New best validation loss: "
                    f"{best_val_loss:.4f}"
                )

            else:

                evaluations_without_improvement += 1

                print(
                    f"  No significant improvement "
                    f"({evaluations_without_improvement}/"
                    f"{PATIENCE})"
                )

            # ----------------------------------------
            # Latest checkpoint
            # ----------------------------------------

            save_checkpoint(
                LATEST_CHECKPOINT,
                model,
                optimizer,
                scaler,
                step,
                best_val_loss
            )

            # ----------------------------------------
            # Early stopping
            # ----------------------------------------

            if evaluations_without_improvement >= PATIENCE:

                print(
                    "\nValidation loss has plateaued."
                )

                print(
                    "Stopping training early."
                )

                break

        # --------------------------------------------
        # Batch
        # --------------------------------------------

        x, y = get_batch(
            train_data,
            BATCH_SIZE,
            BLOCK_SIZE,
            DEVICE
        )

        # --------------------------------------------
        # Forward + loss
        # --------------------------------------------

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp
        ):

            logits, loss = model(x, y)

        # --------------------------------------------
        # Backpropagation
        # --------------------------------------------

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

    # ========================================================
    # FINAL
    # ========================================================

    final_losses = estimate_loss(
        model,
        train_data,
        val_data
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Final train loss: "
        f"{final_losses['train']:.4f}"
    )

    print(
        f"Final validation loss: "
        f"{final_losses['val']:.4f}"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Best checkpoint: "
        f"{BEST_CHECKPOINT}"
    )

    print(
        f"Latest checkpoint: "
        f"{LATEST_CHECKPOINT}"
    )


if __name__ == "__main__":
    main()