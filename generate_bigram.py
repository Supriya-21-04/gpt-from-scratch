import torch

from bigram import BigramLanguageModel
from data import load_data


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Load the checkpoint.
checkpoint = torch.load(
    "bigram_checkpoint.pt",
    map_location=DEVICE
)

stoi = checkpoint["stoi"]
itos = checkpoint["itos"]
vocab_size = checkpoint["vocab_size"]

# Recreate the exact model architecture.
model = BigramLanguageModel(vocab_size).to(DEVICE)

# Load the learned parameters.
model.load_state_dict(
    checkpoint["model_state_dict"]
)

# Switch to evaluation mode.
model.eval()


# Start generation with a newline.
start_character = "\n"

start_id = torch.tensor(
    [[stoi[start_character]]],
    dtype=torch.long,
    device=DEVICE
)

with torch.no_grad():

    generated = model.generate(
        start_id,
        max_new_tokens=500
    )


# Convert token IDs back into characters.
text = "".join(
    itos[i]
    for i in generated[0].tolist()
)

print(text)