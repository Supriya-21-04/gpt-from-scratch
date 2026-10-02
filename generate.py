import torch
import torch.nn.functional as F

from model import GPTLanguageModel


CHECKPOINT_PATH = "checkpoints/gpt_checkpoint.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Load checkpoint
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
)

stoi = checkpoint["stoi"]
itos = checkpoint["itos"]
config = checkpoint["config"]


# Recreate model with the same architecture
model = GPTLanguageModel(
    vocab_size=config["vocab_size"],
    n_embd=config["n_embd"],
    num_heads=config["num_heads"],
    num_layers=config["num_layers"],
    block_size=config["block_size"],
    dropout=config["dropout"],
).to(DEVICE)


# Load learned weights
model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# Starting prompt
prompt = "The "

# Convert characters → token IDs
input_ids = torch.tensor(
    [[stoi[ch] for ch in prompt]],
    dtype=torch.long,
    device=DEVICE,
)


# Generate
with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=1000,
        temperature=0.8,
        top_k=20,
    )


# Convert token IDs → characters
generated_text = "".join(
    itos[i]
    for i in output_ids[0].tolist()
)


print("\nGenerated text:\n")
print(generated_text)