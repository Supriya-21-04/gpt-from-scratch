import torch
import torch.nn.functional as F

from data import load_data, get_batch
from bigram import BigramLanguageModel


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BATCH_SIZE = 64
BLOCK_SIZE = 64

MAX_ITERS = 5000

EVAL_INTERVAL = 500
EVAL_ITERS = 100

LEARNING_RATE = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    """
    Estimate average training and validation loss.

    torch.no_grad() is important because we are evaluating,
    not training. We don't need to construct a gradient graph.
    """

    model.eval()

    results = {}

    for split_name, data in [
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

        results[split_name] = losses.mean().item()

    model.train()

    return results


# ---------------------------------------------------------
# Main training
# ---------------------------------------------------------

def main():

    print(f"Using device: {DEVICE}")

    # Load our character-level dataset.
    train_data, val_data, stoi, itos = load_data()

    vocab_size = len(stoi)

    print(f"Vocabulary size: {vocab_size}")

    # Create the model.
    model = BigramLanguageModel(vocab_size).to(DEVICE)

    print(
        f"Number of parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # AdamW is our optimizer.
    #
    # The optimizer uses the gradients calculated by
    # loss.backward() to update model parameters.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------

    for step in range(MAX_ITERS):

        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(
                model,
                train_data,
                val_data
            )

            print(
                f"step {step:5d} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f}"
            )

        # Get one random training batch.
        xb, yb = get_batch(
            train_data,
            BATCH_SIZE,
            BLOCK_SIZE,
            DEVICE
        )

        # Forward pass:
        # input → model → predictions → loss
        logits, loss = model(xb, yb)

        # Remove gradients from the previous iteration.
        optimizer.zero_grad(set_to_none=True)

        # Backpropagation:
        # calculate gradients of the loss
        # with respect to every trainable parameter.
        loss.backward()

        # Update model parameters using those gradients.
        optimizer.step()

    # -----------------------------------------------------
    # Save checkpoint
    # -----------------------------------------------------

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "vocab_size": vocab_size,
        "block_size": BLOCK_SIZE
    }

    torch.save(
        checkpoint,
        "bigram_checkpoint.pt"
    )

    print("\nSaved checkpoint to bigram_checkpoint.pt")


if __name__ == "__main__":
    main()