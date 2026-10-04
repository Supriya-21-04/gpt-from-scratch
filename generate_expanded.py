import torch

from model import GPTLanguageModel
from data_expanded import load_data


# =========================================================
# Configuration
# =========================================================

N_EMBD = 384
NUM_HEADS = 6
NUM_LAYERS = 6
BLOCK_SIZE = 256
DROPOUT = 0.2

CHECKPOINT_PATH = (
    "/content/drive/MyDrive/"
    "gpt-from-scratch-expanded/best.pt"
)

# Generation settings
PROMPT = "The "
MAX_NEW_TOKENS = 1000

TEMPERATURE = 0.8
TOP_K = 20


# =========================================================
# Device
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)


# =========================================================
# Load vocabulary
# =========================================================

_, _, stoi, itos = load_data()

vocab_size = len(stoi)

print("Vocabulary size:", vocab_size)


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

model.eval()

print("Checkpoint:", CHECKPOINT_PATH)
print("Checkpoint step:", checkpoint["step"])
print(
    "Best validation loss:",
    f"{checkpoint['best_val_loss']:.4f}"
)


# =========================================================
# Encode prompt
# =========================================================

unknown_chars = [
    ch for ch in PROMPT
    if ch not in stoi
]

if unknown_chars:
    raise ValueError(
        f"Prompt contains characters not in vocabulary: "
        f"{unknown_chars}"
    )

encoded_prompt = [
    stoi[ch]
    for ch in PROMPT
]

idx = torch.tensor(
    [encoded_prompt],
    dtype=torch.long,
    device=DEVICE
)


# =========================================================
# Generate
# =========================================================

print()
print("=" * 60)
print("GENERATING TEXT")
print("=" * 60)

print("Prompt:", repr(PROMPT))
print("Temperature:", TEMPERATURE)
print("Top-k:", TOP_K)
print("New tokens:", MAX_NEW_TOKENS)

with torch.no_grad():

    generated = model.generate(
        idx,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K
    )


# =========================================================
# Decode
# =========================================================

generated_ids = generated[0].tolist()

text = "".join(
    itos[i]
    for i in generated_ids
)


# =========================================================
# Display
# =========================================================

print()
print("=" * 60)
print("GENERATED TEXT")
print("=" * 60)

print(text)