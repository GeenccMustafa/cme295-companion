"""Scaled dot-product attention and multi-head attention, in NumPy.

    Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V

Shapes (no batch dimension, to keep the algebra visible):
    Q: (n_q, d_k)   one query per position that is asking
    K: (n_k, d_k)   one key per position that can be looked at
    V: (n_k, d_v)   one value per position that can be looked at
    out: (n_q, d_v)

Why √d_k: if the entries of q and k are independent with mean 0 and variance 1,
q · k has variance d_k. Dividing by √d_k brings it back to variance 1, so the
softmax does not saturate and its gradients do not vanish.
"""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax. Rows that are entirely -inf become all zeros."""
    x_max = np.max(x, axis=axis, keepdims=True)
    x_max = np.where(np.isfinite(x_max), x_max, 0.0)
    e = np.exp(x - x_max)
    total = e.sum(axis=axis, keepdims=True)
    return np.divide(e, total, out=np.zeros_like(e), where=total > 0)


def causal_mask(n: int) -> np.ndarray:
    """(n, n) boolean mask, True where position i may attend to position j (j <= i)."""
    return np.tril(np.ones((n, n), dtype=bool))


def padding_mask(length: int, n: int) -> np.ndarray:
    """(n,) boolean key mask for a sequence padded to n with `length` real tokens.

    Broadcasts against (n_q, n) scores, so padded keys are blocked for every query.
    """
    return np.arange(n) < length


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (output (n_q, d_v), attention weights (n_q, n_k)).

    `mask` is boolean and broadcastable to (n_q, n_k); False entries are blocked.
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)  # (n_q, n_k)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    weights = softmax(scores, axis=-1)  # each row sums to 1
    return weights @ V, weights


class MultiHeadAttention:
    """h heads of size d_k = d_model / h, run in parallel and concatenated.

    Parameters: W_Q, W_K, W_V, W_O, each (d_model, d_model), so 4 · d_model² weights
    (plus 4 · d_model biases, omitted here as in many modern models).
    """

    def __init__(self, d_model: int, num_heads: int, rng: np.random.Generator):
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_model // num_heads
        std = 1.0 / np.sqrt(d_model)
        self.W_Q = rng.normal(0, std, (d_model, d_model))
        self.W_K = rng.normal(0, std, (d_model, d_model))
        self.W_V = rng.normal(0, std, (d_model, d_model))
        self.W_O = rng.normal(0, std, (d_model, d_model))

    def num_parameters(self) -> int:
        return 4 * self.d_model * self.d_model

    def _split(self, x: np.ndarray) -> np.ndarray:
        """(n, d_model) -> (h, n, d_k)"""
        n = x.shape[0]
        return x.reshape(n, self.h, self.d_k).transpose(1, 0, 2)

    def __call__(
        self,
        x_q: np.ndarray,
        x_kv: np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Self-attention when x_kv is None, cross-attention otherwise.

        x_q: (n_q, d_model), x_kv: (n_k, d_model).
        Returns (output (n_q, d_model), weights (h, n_q, n_k)).
        """
        if x_kv is None:
            x_kv = x_q
        Q = self._split(x_q @ self.W_Q)  # (h, n_q, d_k)
        K = self._split(x_kv @ self.W_K)  # (h, n_k, d_k)
        V = self._split(x_kv @ self.W_V)  # (h, n_k, d_k)

        heads, weights = [], []
        for i in range(self.h):
            out, w = scaled_dot_product_attention(Q[i], K[i], V[i], mask)
            heads.append(out)
            weights.append(w)

        concat = np.concatenate(heads, axis=-1)  # (n_q, d_model)
        return concat @ self.W_O, np.stack(weights)
