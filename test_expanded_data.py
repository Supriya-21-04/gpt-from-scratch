import torch

from data_expanded import load_data, get_batch


train_data, val_data, stoi, itos = load_data()

print("Vocabulary size:", len(stoi))
print("Training tokens:", len(train_data))
print("Validation tokens:", len(val_data))

device = "cuda" if torch.cuda.is_available() else "cpu"

x, y = get_batch(
    train_data,
    batch_size=4,
    block_size=128,
    device=device
)

print()
print("Device:", device)
print("X shape:", x.shape)
print("Y shape:", y.shape)

print()
print("Example X:")
print(repr("".join(itos[int(i)] for i in x[0])))

print()
print("Example Y:")
print(repr("".join(itos[int(i)] for i in y[0])))

# ---------------------------------------------------------
# Verify next-token alignment
# ---------------------------------------------------------

alignment_ok = torch.equal(
    x[:, 1:],
    y[:, :-1]
)

print()

if alignment_ok:
    print("Next-token alignment: PASSED")
else:
    print("Next-token alignment: FAILED")