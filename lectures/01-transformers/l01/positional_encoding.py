"""Sinusoidal positional encoding from "Attention Is All You Need" (Vaswani et al., 2017).

Self-attention is permutation-equivariant: shuffle the input tokens and the
outputs are shuffled the same way. Adding a position-dependent vector to each
token embedding is what lets the model know word order.

    PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
    PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))

Each pair of dimensions is a sine/cosine at its own frequency, from one turn
every 2π positions (i = 0) down to one turn every 2π·10000 positions.
"""

from __future__ import annotations

import numpy as np


def sinusoidal_positional_encoding(max_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """Returns PE of shape (max_len, d_model). d_model must be even."""
    if d_model % 2:
        raise ValueError("d_model must be even")
    pos = np.arange(max_len)[:, None]  # (max_len, 1)
    i = np.arange(d_model // 2)[None, :]  # (1, d_model/2)
    angles = pos / base ** (2 * i / d_model)  # (max_len, d_model/2)
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return pe
