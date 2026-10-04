import torch

from data import load_data
from model import GPTLanguageModel


CHECKPOINT_PATH = (
    "/content/drive/MyDrive/"
    "gpt-from-scratch-checkpoints/"
    "gpt_scaled_best.pt"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------
# Load vocabulary from the original data pipeline
# --------------------------------------------------

train_data, val_data, stoi, itos = load_data()
vocab_size = len(stoi)


# --------------------------------------------------
# Load scaled model checkpoint
# --------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False,
)

config = checkpoint["config"]

print(f"Device: {DEVICE}")
print(f"Vocabulary size: {vocab_size}")
print(f"Checkpoint step: {checkpoint['step']}")


# --------------------------------------------------
# Rebuild the model using checkpoint configuration
# --------------------------------------------------

model = GPTLanguageModel(
    vocab_size=vocab_size,
    n_embd=config["n_embd"],
    num_heads=config["num_heads"],
    num_layers=config["num_layers"],
    block_size=config["block_size"],
    dropout=config["dropout"],
).to(DEVICE)


model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# --------------------------------------------------
# Generate text
# --------------------------------------------------

prompt = "The "

input_ids = torch.tensor(
    [[stoi[ch] for ch in prompt]],
    dtype=torch.long,
    device=DEVICE,
)


with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=1000,
        temperature=0.8,
        top_k=20,
    )


generated_text = "".join(
    itos[i]
    for i in output_ids[0].tolist()
)


print("\nGenerated text:\n")
print(generated_text)