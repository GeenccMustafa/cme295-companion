"""Walk through Lecture 01 end to end and print every intermediate result.

The script runs each stage of the lecture in order: train a BPE tokenizer,
measure vanishing gradients in an RNN, apply causal self-attention, build
sinusoidal positional encodings, and finally push a sentence through a full
encoder-decoder Transformer.

Run it from the repository root::

    uv run python lectures/01-transformers/demo.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from l01.attention import causal_mask, scaled_dot_product_attention
from l01.positional_encoding import sinusoidal_positional_encoding
from l01.rnn import VanillaRNN
from l01.tokenization import BPETokenizer
from l01.transformer import Transformer, TransformerConfig


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 8} {title} {'=' * (60 - len(title))}")


def demo_tokenization() -> None:
    """Train BPE on a toy corpus and tokenize a few words."""
    section("1. BPE tokenization")
    corpus = "low " * 5 + "lower " * 2 + "newest " * 6 + "widest " * 3
    tokenizer = BPETokenizer.train(corpus, num_merges=10)
    print("first merges:", tokenizer.merges[:5])
    print("vocab size:  ", len(tokenizer.vocab))
    for word in ["newest", "lowest", "wider"]:
        print(f"  {word!r:10} -> {tokenizer.tokenize(word)}")


def demo_rnn() -> None:
    """Show how quickly an RNN forgets early inputs."""
    section("2. Why RNNs struggle with long sequences")
    rnn = VanillaRNN(input_dim=4, hidden_dim=16)
    norms = rnn.gradient_norms(np.random.default_rng(0).normal(size=(20, 4)))
    print("‖∂h_T/∂h_t‖ for t = 1, 5, 10, 15, 20:", norms[[0, 4, 9, 14, 19]])


def demo_attention() -> None:
    """Print a causal self-attention weight matrix."""
    section("3. Scaled dot-product attention with a causal mask")
    x = np.random.default_rng(0).normal(size=(4, 8))
    _, weights = scaled_dot_product_attention(x, x, x, causal_mask(4))
    print("attention weights (row = query, column = key):")
    print(weights)


def demo_positional_encoding() -> None:
    """Print the first dimensions of the sinusoidal encoding table."""
    section("4. Sinusoidal positional encoding")
    pe = sinusoidal_positional_encoding(max_len=6, d_model=8)
    print("PE[:, :4] =")
    print(pe[:, :4])


def demo_transformer() -> None:
    """Encode a sentence and greedily decode with an untrained Transformer."""
    section("5. A full encoder-decoder forward pass")
    cfg = TransformerConfig(vocab_size=100, d_model=32, num_heads=4, d_ff=64, num_layers=2)
    model = Transformer(cfg)
    src = np.array([1, 17, 42, 8, 2])  # e.g. [BOS] my teddy-bear is-cute [EOS]
    memory = model.encode(src)
    print("source ids:      ", src, "->  encoder output", memory.shape)
    generated = model.greedy_decode(src, bos_id=1, eos_id=2, max_new=6)
    print("greedy decode:   ", generated, "(untrained weights, so the ids are arbitrary)")


def main() -> None:
    """Run every section of the demo."""
    np.set_printoptions(precision=3, suppress=True)
    demo_tokenization()
    demo_rnn()
    demo_attention()
    demo_positional_encoding()
    demo_transformer()


if __name__ == "__main__":
    main()
