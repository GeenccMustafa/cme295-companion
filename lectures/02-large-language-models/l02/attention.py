r"""Causal self-attention as used in modern decoder-only LLMs.

Lecture 01 built multi-head attention (MHA) for the original Transformer.
LLMs keep the same core computation but change three things around it:

1. **Grouped-query attention (GQA).** During generation the keys and values
   of every past token are kept in memory (the *KV cache*), and reading that
   cache, not arithmetic, limits decoding speed. Multi-query attention
   (MQA) [1]_ shares a single key/value head across all query heads; GQA [2]_
   interpolates by sharing each key/value head within a *group* of query
   heads. With ``h`` query heads and ``g`` key/value heads, ``g = h`` is MHA,
   ``g = 1`` is MQA, and anything in between is GQA.
2. **Rotary position embeddings** are applied to queries and keys inside
   every layer (see :mod:`l02.rope`) instead of adding positions to the
   input once.
3. **Sliding-window attention (SWA)** [3]_ optionally restricts each token
   to the most recent ``W`` positions, making the cost per token constant.
   Stacking layers still lets information travel further: after ``L`` layers
   a token can be influenced by inputs about ``L · W`` positions back.

The attention itself is the scaled dot-product attention of Lecture 01,
reused from :func:`l01.attention.scaled_dot_product_attention`.

Notes
-----
Query head :math:`i` uses key/value head :math:`\lfloor i / (h/g) \rfloor`.
With model width :math:`d` and head size :math:`d_h = d/h`, the projections
have shapes

.. math::

    W_Q \in \mathbb{R}^{d \times h d_h}, \quad
    W_K, W_V \in \mathbb{R}^{d \times g d_h}, \quad
    W_O \in \mathbb{R}^{h d_h \times d},

so the attention block has :math:`2d^2 + 2 d\, g\, d_h` weights, and the KV
cache stores

.. math::

    2 \cdot L \cdot g \cdot d_h \cdot \text{bytes per value}

bytes per token for a model with :math:`L` layers (the factor 2 counts keys
and values).

*Window convention.* Here a window of size :math:`W` means that position
:math:`i` attends to positions :math:`i - W + 1, \dots, i`, that is :math:`W`
positions including itself, matching the figure in the Mistral paper.
The paper's text says "between :math:`i - W` and :math:`i`"; the
difference of one position does not change any conclusion.

References
----------
.. [1] N. Shazeer, "Fast Transformer Decoding: One Write-Head is All You
   Need", 2019. https://arxiv.org/abs/1911.02150
.. [2] J. Ainslie et al., "GQA: Training Generalized Multi-Query Transformer
   Models from Multi-Head Checkpoints", EMNLP 2023.
   https://arxiv.org/abs/2305.13245
.. [3] A. Q. Jiang et al., "Mistral 7B", 2023, Section 2.
   https://arxiv.org/abs/2310.06825

Examples
--------
Eight query heads sharing two key/value heads:

>>> attn = GroupedQueryAttention(
...     d_model=64, num_heads=8, num_kv_heads=2, rng=np.random.default_rng(0)
... )
>>> out, weights = attn(np.random.default_rng(1).normal(size=(5, 64)))
>>> out.shape, weights.shape
((5, 64), (8, 5, 5))
"""

from __future__ import annotations

import numpy as np

from l01.attention import causal_mask, scaled_dot_product_attention

from .rope import apply_rope

__all__ = [
    "GroupedQueryAttention",
    "kv_cache_bytes",
    "receptive_field",
    "sliding_window_mask",
]


def sliding_window_mask(n: int, window: int) -> np.ndarray:
    """Build a causal mask that only keeps the most recent `window` positions.

    Parameters
    ----------
    n : int
        Sequence length.
    window : int
        Number of positions each query may attend to, itself included.

    Returns
    -------
    numpy.ndarray of bool, shape (n, n)
        Entry ``[i, j]`` is True when ``i - window < j <= i``.

    Examples
    --------
    >>> sliding_window_mask(4, window=2).astype(int)
    array([[1, 0, 0, 0],
           [1, 1, 0, 0],
           [0, 1, 1, 0],
           [0, 0, 1, 1]])
    """
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    return (j <= i) & (j > i - window)


def receptive_field(mask: np.ndarray, num_layers: int) -> np.ndarray:
    """Compute which inputs can influence which outputs after stacked layers.

    Each attention layer lets position ``i`` read the positions allowed by
    `mask`; stacking layers composes these one-step connections.

    Parameters
    ----------
    mask : numpy.ndarray of bool, shape (n, n)
        One layer's attention mask.
    num_layers : int
        Number of stacked layers using that mask.

    Returns
    -------
    numpy.ndarray of bool, shape (n, n)
        Entry ``[i, j]`` is True when input position ``j`` can influence
        output position ``i``.

    Examples
    --------
    With a window of 3, each layer reaches 2 positions further back:

    >>> reach = receptive_field(sliding_window_mask(10, window=3), num_layers=3)
    >>> int(reach[9].nonzero()[0].min())
    3
    """
    reach = np.eye(mask.shape[0], dtype=bool)
    step = mask.astype(int)
    for _ in range(num_layers):
        reach = (step @ reach.astype(int)) > 0
    return reach


def kv_cache_bytes(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    num_tokens: int,
    bytes_per_value: int = 2,
) -> int:
    """Return the memory needed to cache keys and values.

    Parameters
    ----------
    num_layers : int
        Number of attention layers ``L``.
    num_kv_heads : int
        Key/value heads per layer ``g``.
    head_dim : int
        Size of each head ``d_h``.
    num_tokens : int
        Number of cached tokens (prompt plus generated tokens).
    bytes_per_value : int, default 2
        2 for bfloat16 or float16, 1 for 8-bit formats.

    Returns
    -------
    int
        ``2 · L · g · d_h · num_tokens · bytes_per_value``.

    Examples
    --------
    Mistral 7B (32 layers, 8 KV heads of size 128) in bfloat16 needs
    128 KiB per token; with all 32 heads as KV heads (plain MHA) it would
    need four times as much:

    >>> kv_cache_bytes(32, 8, 128, num_tokens=1) // 1024
    128
    >>> kv_cache_bytes(32, 32, 128, num_tokens=1) // 1024
    512
    """
    return 2 * num_layers * num_kv_heads * head_dim * num_tokens * bytes_per_value


class GroupedQueryAttention:
    """Causal grouped-query self-attention with rotary position embeddings.

    Parameters
    ----------
    d_model : int
        Model width ``d``.
    num_heads : int
        Number of query heads ``h``. Must divide `d_model`.
    num_kv_heads : int
        Number of key/value heads ``g``. Must divide `num_heads`. Use
        `num_heads` for MHA and 1 for MQA.
    rng : numpy.random.Generator
        Source of randomness for weight initialization.
    rope_base : float or None, default 10000.0
        Base of the RoPE frequencies. None disables RoPE, which turns the
        block into plain causal attention without position information.
    window : int or None, default None
        Sliding-window size. None means full causal attention.

    Attributes
    ----------
    W_Q : numpy.ndarray of shape (d_model, num_heads · head_dim)
        Query projection.
    W_K, W_V : numpy.ndarray of shape (d_model, num_kv_heads · head_dim)
        Key and value projections.
    W_O : numpy.ndarray of shape (num_heads · head_dim, d_model)
        Output projection.
    head_dim : int
        ``d_model // num_heads``.

    Raises
    ------
    ValueError
        If `d_model` is not divisible by `num_heads`, or `num_heads` is not
        divisible by `num_kv_heads`.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        rng: np.random.Generator,
        rope_base: float | None = 10000.0,
        window: int | None = None,
    ):
        if d_model % num_heads:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        if num_heads % num_kv_heads:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.rope_base = rope_base
        self.window = window
        kv_dim = num_kv_heads * self.head_dim
        std = 1.0 / np.sqrt(d_model)
        self.W_Q = rng.normal(0, std, (d_model, d_model))
        self.W_K = rng.normal(0, std, (d_model, kv_dim))
        self.W_V = rng.normal(0, std, (d_model, kv_dim))
        self.W_O = rng.normal(0, std, (d_model, d_model))

    def num_parameters(self) -> int:
        """Return the number of weights, ``2 · d² + 2 · d · g · d_h``."""
        return self.W_Q.size + self.W_K.size + self.W_V.size + self.W_O.size

    def _split_heads(self, x: np.ndarray, num_heads: int) -> np.ndarray:
        """Reshape ``(n, num_heads · head_dim)`` into ``(num_heads, n, head_dim)``."""
        n = x.shape[0]
        return x.reshape(n, num_heads, self.head_dim).transpose(1, 0, 2)

    def __call__(
        self, x: np.ndarray, positions: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply causal self-attention to a sequence.

        Parameters
        ----------
        x : numpy.ndarray of shape (n, d_model)
            Input vectors.
        positions : numpy.ndarray of int, shape (n,), optional
            Positions used by RoPE. Defaults to ``0, 1, ..., n - 1``.

        Returns
        -------
        output : numpy.ndarray of shape (n, d_model)
            Attention output after the output projection.
        weights : numpy.ndarray of shape (num_heads, n, n)
            Attention weights of every query head.
        """
        n = x.shape[0]
        if positions is None:
            positions = np.arange(n)
        Q = self._split_heads(x @ self.W_Q, self.num_heads)  # (h, n, d_h)
        K = self._split_heads(x @ self.W_K, self.num_kv_heads)  # (g, n, d_h)
        V = self._split_heads(x @ self.W_V, self.num_kv_heads)  # (g, n, d_h)
        if self.rope_base is not None:
            Q = apply_rope(Q, positions, self.rope_base)
            K = apply_rope(K, positions, self.rope_base)

        mask = causal_mask(n) if self.window is None else sliding_window_mask(n, self.window)
        group_size = self.num_heads // self.num_kv_heads
        heads, weights = [], []
        for i in range(self.num_heads):
            kv = i // group_size  # key/value head shared by this query head's group
            out, w = scaled_dot_product_attention(Q[i], K[kv], V[kv], mask)
            heads.append(out)
            weights.append(w)

        concat = np.concatenate(heads, axis=-1)  # (n, d_model)
        return concat @ self.W_O, np.stack(weights)
