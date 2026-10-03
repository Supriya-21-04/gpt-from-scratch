from pathlib import Path
import math

import torch

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

# Safety ceiling only.
MAX_ITERS = 20000

# Validation / checkpoint settings
EVAL_INTERVAL = 500
EVAL_ITERS = 100
CHECKPOINT_INTERVAL = 250

# Early stopping
PATIENCE = 5
MIN_DELTA = 0.002

SEED = 1337

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# GOOGLE DRIVE CHECKPOINTS
# ============================================================

DRIVE_ROOT = Path("/content/drive/MyDrive")

if not DRIVE_ROOT.exists():
    raise RuntimeError(
        "Google Drive is not mounted. Run:\n"
        "from google.colab import drive\n"
        'drive.mount("/content/drive")\n'
        "and then run this script again."
    )

CHECKPOINT_DIR = DRIVE_ROOT / "gpt-from-scratch-checkpoints"
CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LATEST_CHECKPOINT = (
    CHECKPOINT_DIR / "gpt_scaled_latest.pt"
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR / "gpt_scaled_best.pt"
)

# Automatically resume if a checkpoint exists.
RESUME = True


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
    """
    Linear warmup followed by cosine decay.
    """

    # Linear warmup
    if step < WARMUP_ITERS:
        return LEARNING_RATE * (
            step + 1
        ) / WARMUP_ITERS

    # Cosine decay
    progress = (
        step - WARMUP_ITERS
    ) / (
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
def estimate_loss(
    model,
    train_data,
    val_data
):
    """
    Estimate average train and validation loss
    over several random batches.
    """

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
# RNG STATE HELPERS
# ============================================================

def prepare_cuda_rng_state_for_save():
    """
    Convert CUDA RNG states to CPU uint8 tensors.

    This makes the checkpoint format robust across
    different PyTorch / CUDA environments.
    """

    if not torch.cuda.is_available():
        return None

    states = torch.cuda.get_rng_state_all()

    return [
        state.detach().cpu().to(torch.uint8)
        for state in states
    ]


def restore_cuda_rng_state(cuda_rng_state):
    """
    Restore CUDA RNG state safely.

    Old checkpoints may contain the state in a format
    that the current PyTorch version does not accept.
    In that case, skip RNG restoration rather than
    crashing the entire training run.
    """

    if (
        not torch.cuda.is_available()
        or cuda_rng_state is None
    ):
        return

    try:

        # A single tensor is converted into a list.
        if isinstance(cuda_rng_state, torch.Tensor):
            states = [cuda_rng_state]
        else:
            states = list(cuda_rng_state)

        # Ensure every state is a CPU ByteTensor.
        states = [
            state.detach().cpu().to(torch.uint8)
            for state in states
        ]

        torch.cuda.set_rng_state_all(states)

        print("CUDA RNG state restored.")

    except Exception as exc:

        print(
            "Warning: could not restore CUDA RNG state."
        )
        print(
            f"Reason: {exc}"
        )
        print(
            "Continuing training without restoring "
            "CUDA RNG state."
        )


# ============================================================
# CHECKPOINTING
# ============================================================

CHECKPOINT_VERSION = 2


def save_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    step,
    best_val_loss,
    evaluations_without_improvement
):
    """
    Save the complete training state.

    In version 2 checkpoints, `step` means:
        the last completed optimizer step.
    """

    checkpoint = {

        # Checkpoint format version
        "checkpoint_version":
            CHECKPOINT_VERSION,

        # Last completed optimizer step
        "step": step,

        # Model
        "model_state_dict":
            model.state_dict(),

        # Optimizer
        "optimizer_state_dict":
            optimizer.state_dict(),

        # AMP scaler
        "scaler_state_dict":
            scaler.state_dict(),

        # Validation tracking
        "best_val_loss":
            best_val_loss,

        "evaluations_without_improvement":
            evaluations_without_improvement,

        # Configuration
        "config": {

            "batch_size":
                BATCH_SIZE,

            "block_size":
                BLOCK_SIZE,

            "n_embd":
                N_EMBD,

            "num_heads":
                NUM_HEADS,

            "num_layers":
                NUM_LAYERS,

            "dropout":
                DROPOUT,

            "learning_rate":
                LEARNING_RATE,

            "min_lr":
                MIN_LR,

            "warmup_iters":
                WARMUP_ITERS,

            "max_iters":
                MAX_ITERS,
        },

        # CPU RNG
        "rng_state":
            torch.get_rng_state().cpu().to(torch.uint8),

        # CUDA RNG
        "cuda_rng_state":
            prepare_cuda_rng_state_for_save(),
    }

    torch.save(
        checkpoint,
        path
    )


def load_checkpoint(
    path,
    model,
    optimizer,
    scaler
):
    """
    Load a checkpoint and return:

        start_step
        best_val_loss
        evaluations_without_improvement

    Supports BOTH:

    1. Old checkpoint format from the previous run
       that stopped around step 6500.

    2. New version-2 checkpoints created by this script.
    """

    print("\nLoading checkpoint:")
    print(path)

    checkpoint = torch.load(
        path,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    # --------------------------------------------------------
    # AMP scaler
    # --------------------------------------------------------

    if "scaler_state_dict" in checkpoint:

        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

    # --------------------------------------------------------
    # Restore CPU random state
    # --------------------------------------------------------

    if "rng_state" in checkpoint:

        try:

            cpu_rng_state = (
                checkpoint["rng_state"]
                .detach()
                .cpu()
                .to(torch.uint8)
            )

            torch.set_rng_state(
                cpu_rng_state
            )

            print("CPU RNG state restored.")

        except Exception as exc:

            print(
                "Warning: could not restore CPU RNG state."
            )

            print(
                f"Reason: {exc}"
            )

    # --------------------------------------------------------
    # Restore CUDA random state
    # --------------------------------------------------------

    if (
        DEVICE == "cuda"
        and checkpoint.get(
            "cuda_rng_state"
        ) is not None
    ):

        restore_cuda_rng_state(
            checkpoint["cuda_rng_state"]
        )

    # --------------------------------------------------------
    # Determine checkpoint format
    # --------------------------------------------------------

    checkpoint_version = checkpoint.get(
        "checkpoint_version",
        1
    )

    checkpoint_step = checkpoint["step"]

    # --------------------------------------------------------
    # Resume position
    # --------------------------------------------------------
    #
    # OLD checkpoint:
    #
    # The old script saved the checkpoint BEFORE running
    # the optimizer update for that step.
    #
    # Therefore the old step 6500 checkpoint needs to
    # replay step 6500.
    #
    # NEW checkpoint:
    #
    # The new script saves AFTER the optimizer update.
    #
    # Therefore the next step is checkpoint_step + 1.
    #
    # --------------------------------------------------------

    if checkpoint_version >= 2:

        start_step = checkpoint_step + 1

        print(
            "Checkpoint format: version 2"
        )

        print(
            "Checkpoint represents a completed "
            "optimizer step."
        )

    else:

        start_step = checkpoint_step

        print(
            "Checkpoint format: legacy"
        )

        print(
            "Legacy checkpoint detected."
        )

        print(
            "Replaying the saved step so that "
            "no training update is skipped."
        )

    # --------------------------------------------------------
    # Validation state
    # --------------------------------------------------------

    best_val_loss = checkpoint.get(
        "best_val_loss",
        float("inf")
    )

    evaluations_without_improvement = (
        checkpoint.get(
            "evaluations_without_improvement",
            0
        )
    )

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    print(
        f"Checkpoint step: "
        f"{checkpoint_step}"
    )

    print(
        f"Resuming from step: "
        f"{start_step}"
    )

    print(
        f"Best validation loss so far: "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Evaluations without improvement: "
        f"{evaluations_without_improvement}"
    )

    return (
        start_step,
        best_val_loss,
        evaluations_without_improvement
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SCALED GPT TRAINING")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    train_data, val_data, stoi, itos = (
        load_data()
    )

    vocab_size = len(stoi)

    print(
        f"Vocabulary size: "
        f"{vocab_size}"
    )

    print(
        f"Training tokens: "
        f"{len(train_data):,}"
    )

    print(
        f"Validation tokens: "
        f"{len(val_data):,}"
    )

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

    print(
        f"Number of parameters: "
        f"{num_parameters:,}"
    )

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

    use_amp = (
        DEVICE == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    print(
        f"Mixed precision: "
        f"{use_amp}"
    )

    # --------------------------------------------------------
    # TRAINING STATE
    # --------------------------------------------------------

    start_step = 0

    best_val_loss = float("inf")

    evaluations_without_improvement = 0

    # --------------------------------------------------------
    # RESUME
    # --------------------------------------------------------

    if (
        RESUME
        and LATEST_CHECKPOINT.exists()
    ):

        (
            start_step,
            best_val_loss,
            evaluations_without_improvement
        ) = load_checkpoint(
            LATEST_CHECKPOINT,
            model,
            optimizer,
            scaler
        )

    else:

        print(
            "\nNo checkpoint found."
        )

        print(
            "Starting training from step 0."
        )

    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    if start_step >= MAX_ITERS:

        print(
            "\nTraining is already at or beyond "
            "MAX_ITERS."
        )

        print(
            f"start_step = {start_step}"
        )

        print(
            f"MAX_ITERS = {MAX_ITERS}"
        )

        print(
            "Nothing more to train."
        )

        return

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print(
        "\nStarting training...\n"
    )

    last_completed_step = (
        start_step - 1
    )

    for step in range(
        start_step,
        MAX_ITERS
    ):

        # --------------------------------------------
        # Learning rate
        # --------------------------------------------

        lr = get_learning_rate(
            step
        )

        for param_group in (
            optimizer.param_groups
        ):

            param_group["lr"] = lr

        # --------------------------------------------
        # Evaluation
        # --------------------------------------------

        if (
            step % EVAL_INTERVAL == 0
        ):

            losses = estimate_loss(
                model,
                train_data,
                val_data
            )

            train_loss = (
                losses["train"]
            )

            val_loss = (
                losses["val"]
            )

            print(
                f"Step {step:5d} | "
                f"LR {lr:.6f} | "
                f"Train loss "
                f"{train_loss:.4f} | "
                f"Val loss "
                f"{val_loss:.4f}"
            )

            # ----------------------------------------
            # Best model
            # ----------------------------------------

            if (
                val_loss
                < best_val_loss - MIN_DELTA
            ):

                best_val_loss = val_loss

                evaluations_without_improvement = 0

                save_checkpoint(
                    BEST_CHECKPOINT,
                    model,
                    optimizer,
                    scaler,
                    step,
                    best_val_loss,
                    evaluations_without_improvement
                )

                print(
                    "  New best validation loss: "
                    f"{best_val_loss:.4f}"
                )

            else:

                evaluations_without_improvement += 1

                print(
                    "  No significant improvement "
                    f"({evaluations_without_improvement}/"
                    f"{PATIENCE})"
                )

            # ----------------------------------------
            # Early stopping
            # ----------------------------------------

            if (
                evaluations_without_improvement
                >= PATIENCE
            ):

                print(
                    "\nValidation loss "
                    "has plateaued."
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

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp
        ):

            logits, loss = model(
                x,
                y
            )

        # --------------------------------------------
        # Backpropagation
        # --------------------------------------------

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        # This optimizer update is now complete.
        last_completed_step = step

        # --------------------------------------------
        # LATEST CHECKPOINT
        # --------------------------------------------
        #
        # IMPORTANT:
        #
        # Saved AFTER optimizer update.
        #
        # Therefore "step" means the LAST COMPLETED
        # training step.
        #
        # Future resume will use step + 1.
        #
        # --------------------------------------------

        if (
            (step + 1)
            % CHECKPOINT_INTERVAL
            == 0
        ):

            save_checkpoint(
                LATEST_CHECKPOINT,
                model,
                optimizer,
                scaler,
                step,
                best_val_loss,
                evaluations_without_improvement
            )

            print(
                "  Checkpoint saved at "
                f"completed step {step}"
            )

    # ========================================================
    # FINAL EVALUATION
    # ========================================================

    final_losses = estimate_loss(
        model,
        train_data,
        val_data
    )

    final_val_loss = (
        final_losses["val"]
    )

    # --------------------------------------------------------
    # Save latest final state
    # --------------------------------------------------------

    if last_completed_step >= 0:

        save_checkpoint(
            LATEST_CHECKPOINT,
            model,
            optimizer,
            scaler,
            last_completed_step,
            best_val_loss,
            evaluations_without_improvement
        )

    # --------------------------------------------------------
    # Save final model as best if appropriate
    # --------------------------------------------------------

    if (
        final_val_loss
        < best_val_loss
    ):

        best_val_loss = (
            final_val_loss
        )

        save_checkpoint(
            BEST_CHECKPOINT,
            model,
            optimizer,
            scaler,
            last_completed_step,
            best_val_loss,
            evaluations_without_improvement
        )

        print(
            "Final model is the new best: "
            f"{best_val_loss:.4f}"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

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