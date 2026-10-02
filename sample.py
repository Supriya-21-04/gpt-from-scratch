import torch

from model import GPTLanguageModel


CHECKPOINT_PATH = "checkpoints/gpt_checkpoint.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Load checkpoint
# -------------------------

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
)

stoi = checkpoint["stoi"]
itos = checkpoint["itos"]
config = checkpoint["config"]

model = GPTLanguageModel(
    vocab_size=config["vocab_size"],
    n_embd=config["n_embd"],
    num_heads=config["num_heads"],
    num_layers=config["num_layers"],
    block_size=config["block_size"],
    dropout=config["dropout"],
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# -------------------------
# Generate at different temperatures
# -------------------------

prompt = "The "

temperatures = [0.5, 0.8, 1.0]

for temperature in temperatures:

    input_ids = torch.tensor(
        [[stoi[ch] for ch in prompt]],
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.no_grad():

        output_ids = model.generate(
            input_ids,
            max_new_tokens=300,
            temperature=temperature,
            top_k=20,
        )

    text = "".join(
        itos[i]
        for i in output_ids[0].tolist()
    )

    print("\n" + "=" * 70)
    print(f"TEMPERATURE = {temperature}")
    print("=" * 70)
    print(text)