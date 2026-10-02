import math
import torch

from data import load_data, get_batch
from model import GPTLanguageModel


# =========================
# Configuration
# =========================

BATCH_SIZE = 32
BLOCK_SIZE = 128

N_EMBD = 128
NUM_LAYERS = 4
DROPOUT = 0.1

MAX_ITERS = 2000
EVAL_ITERS = 100

LEARNING_RATE = 3e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEEDS = [1337, 2024, 42]
HEAD_CONFIGS = [2, 4]


# =========================
# Load data
# =========================

train_data, val_data, stoi, itos = load_data()

vocab_size = len(stoi)

print("Device:", DEVICE)
print("Vocabulary size:", vocab_size)


# =========================
# Evaluation
# =========================

@torch.no_grad()
def estimate_loss(model, data, seed):

    model.eval()

    # Use a separate generator so evaluation sampling
    # does not affect the training RNG.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    losses = []

    for _ in range(EVAL_ITERS):

        starts = torch.randint(
            low=0,
            high=len(data) - BLOCK_SIZE,
            size=(BATCH_SIZE,),
            generator=generator,
        )

        x = torch.stack([
            data[i:i + BLOCK_SIZE]
            for i in starts
        ]).to(DEVICE)

        y = torch.stack([
            data[i + 1:i + BLOCK_SIZE + 1]
            for i in starts
        ]).to(DEVICE)

        _, loss = model(x, y)

        losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)


# =========================
# Run experiments
# =========================

results = {}

for num_heads in HEAD_CONFIGS:

    results[num_heads] = []

    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {num_heads} ATTENTION HEADS")
    print("=" * 70)

    for seed in SEEDS:

        print("\n" + "-" * 50)
        print(f"Seed: {seed}")
        print("-" * 50)

        # Reproducible model initialization + training randomness
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model = GPTLanguageModel(
            vocab_size=vocab_size,
            n_embd=N_EMBD,
            num_heads=num_heads,
            num_layers=NUM_LAYERS,
            block_size=BLOCK_SIZE,
            dropout=DROPOUT,
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
        )

        # -------------------------
        # Training
        # -------------------------

        for step in range(MAX_ITERS):

            x, y = get_batch(
                train_data,
                batch_size=BATCH_SIZE,
                block_size=BLOCK_SIZE,
                device=DEVICE,
            )

            logits, loss = model(x, y)

            optimizer.zero_grad(set_to_none=True)

            loss.backward()

            optimizer.step()

            if step % 500 == 0:

                val_loss = estimate_loss(
                    model,
                    val_data,
                    seed + 1000,
                )

                print(
                    f"Step {step:4d} | "
                    f"Train loss {loss.item():.4f} | "
                    f"Val loss {val_loss:.4f}"
                )

        # -------------------------
        # Final evaluation
        # -------------------------

        train_loss = estimate_loss(
            model,
            train_data,
            seed + 2000,
        )

        val_loss = estimate_loss(
            model,
            val_data,
            seed + 3000,
        )

        results[num_heads].append(val_loss)

        print(
            f"Final | "
            f"Train loss {train_loss:.4f} | "
            f"Val loss {val_loss:.4f}"
        )


# =========================
# Summary
# =========================

print("\n" + "=" * 70)
print("FINAL ABLATION RESULTS")
print("=" * 70)

for num_heads in HEAD_CONFIGS:

    values = results[num_heads]

    mean_loss = sum(values) / len(values)

    variance = sum(
        (x - mean_loss) ** 2
        for x in values
    ) / len(values)

    std_loss = math.sqrt(variance)

    print(
        f"{num_heads} heads | "
        f"Val losses: "
        f"{[round(x, 4) for x in values]} | "
        f"Mean: {mean_loss:.4f} | "
        f"Std: {std_loss:.4f}"
    )