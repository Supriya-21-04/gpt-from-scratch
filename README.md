# GPT From Scratch

A character-level GPT-style language model implemented from scratch in PyTorch.

This project was built to understand the core components of a decoder-only Transformer rather than using a pretrained model, API, or framework wrapper.

## Architecture

The model is a decoder-only Transformer with:

- Character-level tokenization
- Learned token embeddings
- Learned positional embeddings
- Causal multi-head self-attention
- Feed-forward MLP blocks
- Residual connections
- Layer normalization
- GELU activation
- AdamW optimization
- Learning-rate warmup
- Cosine learning-rate decay
- Temperature and top-k text generation

### Model configuration

| Parameter | Value |
|---|---:|
| Vocabulary size | 137 |
| Embedding dimension | 128 |
| Transformer layers | 4 |
| Attention heads | 4 |
| Context length | 128 |
| Dropout | 0.1 |
| Parameters | 843,401 |
| Batch size | 32 |

## Dataset

The model is trained on a public-domain Project Gutenberg text corpus.

The corpus contains approximately 2.7 million characters and is split chronologically into:

- 90% training data
- 10% validation data

The model uses character-level tokenization, so each unique character receives an integer ID.

## Training

The final training run used:

- 20,000 optimization steps
- AdamW
- Initial learning rate: `3e-4`
- Minimum learning rate: `3e-5`
- 200-step warmup
- Cosine learning-rate decay
- Batch size: 32
- Context length: 128

Final evaluation:

| Metric | Value |
|---|---:|
| Training loss | 1.2233 |
| Validation loss | 1.3049 |
| Validation perplexity | ~3.69 |

## Generation

The model generates text autoregressively.

At every step:

1. The previous context is passed through the Transformer.
2. The final position produces logits over the vocabulary.
3. Temperature modifies the logits before sampling.
4. Optional top-k filtering restricts sampling to the most likely tokens.
5. One new character is sampled and appended to the context.

The generated text shows recognizable English words, punctuation, dialogue structure, and names, but it does not maintain reliable grammar or long-range semantic coherence.

This is expected for a small character-level model trained on a relatively narrow corpus.

## Temperature Experiment

Generation was tested at different temperatures.

- `T = 0.5`: more conservative and repetitive
- `T = 0.8`: more varied while retaining reasonable structure
- `T = 1.0`: more diverse but also more noisy

Temperature changes the sharpness of the sampling distribution; it does not change the trained model parameters.

## Attention-Head Ablation

A controlled ablation compared 2 attention heads against 4 attention heads.

Both configurations used the same dataset, model dimensions, training setup, and three random seeds.

| Configuration | Mean validation loss |
|---|---:|
| 2 heads | 1.804 |
| 4 heads | 1.829 |

Across the three seeds, the 2-head configuration had a lower mean validation loss under this specific 2,000-step ablation setup.

This result should be interpreted only within the tested configuration and is not evidence that fewer attention heads are generally better.

## Project Structure

```text
gpt-from-scratch/
├── attention.py
├── block.py
├── model.py
├── data.py
├── train.py
├── evaluate.py
├── generate.py
├── sample.py
├── ablation.py
├── bigram.py
├── train_bigram.py
├── generate_bigram.py
├── test_attention.py
├── test_block.py
├── test_model.py
├── requirements.txt
├── data/
│   ├── corpus.txt
│   └── download_data.py
├── notebooks/
├── results/
├── checkpoints/
├── README.md
└── NOTES.md