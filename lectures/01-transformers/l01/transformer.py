"""The original encoder-decoder Transformer, forward pass only, in NumPy.

Post-norm, as in the 2017 paper:

    Encoder layer:  x = LN(x + SelfAttn(x))
                    x = LN(x + FFN(x))

    Decoder layer:  y = LN(y + MaskedSelfAttn(y))
                    y = LN(y + CrossAttn(q=y, kv=encoder_output))
                    y = LN(y + FFN(y))

The weights are random, so outputs are meaningless; the point is to follow the
shapes and the data flow. Training lives in the PyTorch version.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .attention import MultiHeadAttention, causal_mask
from .positional_encoding import sinusoidal_positional_encoding


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = 1000
    d_model: int = 64
    num_heads: int = 4
    d_ff: int = 256
    num_layers: int = 2
    max_len: int = 128
    # "Transformer base" from the paper: d_model=512, h=8, d_ff=2048, N=6.


class LayerNorm:
    """Normalize each position's vector to mean 0, variance 1, then scale and shift."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps

    def __call__(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta


class FeedForward:
    """Position-wise FFN: widen to d_ff, ReLU, project back. Applied to each token independently."""

    def __init__(self, d_model: int, d_ff: int, rng: np.random.Generator):
        self.W1 = rng.normal(0, 1 / np.sqrt(d_model), (d_model, d_ff))
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.normal(0, 1 / np.sqrt(d_ff), (d_ff, d_model))
        self.b2 = np.zeros(d_model)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x @ self.W1 + self.b1) @ self.W2 + self.b2


class EncoderLayer:
    def __init__(self, cfg: TransformerConfig, rng: np.random.Generator):
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.num_heads, rng)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, rng)
        self.ln1 = LayerNorm(cfg.d_model)
        self.ln2 = LayerNorm(cfg.d_model)

    def __call__(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        attn, _ = self.self_attn(x, mask=mask)
        x = self.ln1(x + attn)
        return self.ln2(x + self.ffn(x))


class DecoderLayer:
    def __init__(self, cfg: TransformerConfig, rng: np.random.Generator):
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.num_heads, rng)
        self.cross_attn = MultiHeadAttention(cfg.d_model, cfg.num_heads, rng)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, rng)
        self.ln1 = LayerNorm(cfg.d_model)
        self.ln2 = LayerNorm(cfg.d_model)
        self.ln3 = LayerNorm(cfg.d_model)

    def __call__(
        self,
        y: np.ndarray,
        memory: np.ndarray,
        memory_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        n = y.shape[0]
        attn, _ = self.self_attn(y, mask=causal_mask(n))  # no peeking at future tokens
        y = self.ln1(y + attn)
        cross, _ = self.cross_attn(y, memory, mask=memory_mask)  # Q from decoder, K/V from encoder
        y = self.ln2(y + cross)
        return self.ln3(y + self.ffn(y))


class Transformer:
    def __init__(self, cfg: TransformerConfig, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.cfg = cfg
        # One embedding matrix shared by encoder input, decoder input and output projection,
        # as in the paper (weight tying).
        self.embedding = rng.normal(0, 1 / np.sqrt(cfg.d_model), (cfg.vocab_size, cfg.d_model))
        self.pe = sinusoidal_positional_encoding(cfg.max_len, cfg.d_model)
        self.encoder = [EncoderLayer(cfg, rng) for _ in range(cfg.num_layers)]
        self.decoder = [DecoderLayer(cfg, rng) for _ in range(cfg.num_layers)]

    def embed(self, ids: np.ndarray) -> np.ndarray:
        """(n,) token ids -> (n, d_model). Embeddings are scaled by √d_model before adding PE."""
        n = len(ids)
        return self.embedding[ids] * np.sqrt(self.cfg.d_model) + self.pe[:n]

    def encode(self, src_ids: np.ndarray, src_mask: np.ndarray | None = None) -> np.ndarray:
        """(n_src,) -> (n_src, d_model): one context-aware vector per source token."""
        x = self.embed(src_ids)
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x

    def decode(
        self,
        tgt_ids: np.ndarray,
        memory: np.ndarray,
        src_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """(n_tgt,) -> logits (n_tgt, vocab_size). Row t predicts token t + 1."""
        y = self.embed(tgt_ids)
        for layer in self.decoder:
            y = layer(y, memory, src_mask)
        return y @ self.embedding.T

    def __call__(self, src_ids: np.ndarray, tgt_ids: np.ndarray) -> np.ndarray:
        return self.decode(tgt_ids, self.encode(src_ids))

    def greedy_decode(
        self, src_ids: np.ndarray, bos_id: int, eos_id: int, max_new: int
    ) -> list[int]:
        """Generate one token at a time, always taking the most likely next token."""
        memory = self.encode(src_ids)
        out = [bos_id]
        for _ in range(max_new):
            logits = self.decode(np.array(out), memory)
            next_id = int(np.argmax(logits[-1]))
            out.append(next_id)
            if next_id == eos_id:
                break
        return out


def cross_entropy(logits: np.ndarray, targets: np.ndarray, label_smoothing: float = 0.0) -> float:
    """Mean cross-entropy over positions. logits (n, V), targets (n,).

    With label smoothing ε the target distribution is (1 - ε) on the true token
    plus ε spread uniformly over all V tokens (the paper uses ε = 0.1).
    """
    n, vocab = logits.shape
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))  # stable log-softmax
    q = np.full((n, vocab), label_smoothing / vocab)
    q[np.arange(n), targets] += 1.0 - label_smoothing
    return float(-(q * log_probs).sum(axis=-1).mean())
