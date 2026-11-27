r"""Scaled dot-product attention and multi-head attention.

Attention lets every position in a sequence build its representation as a
weighted average over all positions, with weights computed from the content
itself. Each position emits a **query** (what it is looking for), a **key**
(what it offers to be matched against) and a **value** (what it contributes
once matched). Unlike a recurrent network, every position is processed at
once and any two positions are a single step apart.

All functions here work on one sequence at a time, without a batch
dimension, so that the linear algebra stays visible.

Notes
-----
Scaled dot-product attention [1]_ is

.. math::

    \operatorname{Attention}(Q, K, V)
        = \operatorname{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V,

with :math:`Q \in \mathbb{R}^{n_q \times d_k}`,
:math:`K \in \mathbb{R}^{n_k \times d_k}` and
:math:`V \in \mathbb{R}^{n_k \times d_v}`. The softmax is taken over each
row, so every query distributes a total weight of 1 over the keys.

*Why divide by* :math:`\sqrt{d_k}`. If the components of a query and a key
are independent with zero mean and unit variance, their dot product has
variance :math:`d_k`. For large :math:`d_k` the raw scores therefore spread
widely, the softmax saturates towards a one-hot vector and its gradients
vanish. Dividing by :math:`\sqrt{d_k}` restores unit variance.

*Masking.* Positions that must not be attended to (future tokens in a
decoder, padding in a batch) receive a score of :math:`-\infty` before the
softmax, which gives them exactly zero weight.

References
----------
.. [1] A. Vaswani et al., "Attention Is All You Need", NeurIPS 2017,
   Sections 3.2.1 and 3.2.2. https://arxiv.org/abs/1706.03762
.. [2] D. Bahdanau, K. Cho and Y. Bengio, "Neural Machine Translation by
   Jointly Learning to Align and Translate", ICLR 2015.
   https://arxiv.org/abs/1409.0473
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "MultiHeadAttention",
    "causal_mask",
    "padding_mask",
    "scaled_dot_product_attention",
    "softmax",
]


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute a numerically stable softmax.

    The maximum along `axis` is subtracted before exponentiating, which
    leaves the result unchanged mathematically but prevents overflow.

    Parameters
    ----------
    x : numpy.ndarray
        Input scores (logits). May contain ``-inf`` for masked entries.
    axis : int, default -1
        Axis along which the probabilities sum to one.

    Returns
    -------
    numpy.ndarray
        Probabilities with the same shape as `x`. A slice that is entirely
        ``-inf`` (every position masked) becomes all zeros instead of NaN.

    Examples
    --------
    >>> softmax(np.array([1000.0, 1000.0])).tolist()
    [0.5, 0.5]
    >>> softmax(np.array([0.0, -np.inf])).tolist()
    [1.0, 0.0]
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    x_max = np.where(np.isfinite(x_max), x_max, 0.0)
    e = np.exp(x - x_max)
    total = e.sum(axis=axis, keepdims=True)
    return np.divide(e, total, out=np.zeros_like(e), where=total > 0)


def causal_mask(n: int) -> np.ndarray:
    """Build a causal (look-ahead) mask for a sequence of length `n`.

    Parameters
    ----------
    n : int
        Sequence length.

    Returns
    -------
    numpy.ndarray of bool, shape (n, n)
        Entry ``[i, j]`` is True when query position ``i`` may attend to
        key position ``j``, that is when ``j <= i``.

    Examples
    --------
    >>> causal_mask(3).astype(int)
    array([[1, 0, 0],
           [1, 1, 0],
           [1, 1, 1]])
    """
    return np.tril(np.ones((n, n), dtype=bool))


def padding_mask(length: int, n: int) -> np.ndarray:
    """Build a key mask that hides padding at the end of a sequence.

    Parameters
    ----------
    length : int
        Number of real (non-padding) tokens.
    n : int
        Padded sequence length.

    Returns
    -------
    numpy.ndarray of bool, shape (n,)
        True for real tokens, False for padding. It broadcasts against an
        ``(n_q, n)`` score matrix, blocking the padded keys for every query.

    Examples
    --------
    >>> padding_mask(3, 5).tolist()
    [True, True, True, False, False]
    """
    return np.arange(n) < length


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute scaled dot-product attention.

    Parameters
    ----------
    Q : numpy.ndarray of shape (n_q, d_k)
        Queries, one per position that is attending.
    K : numpy.ndarray of shape (n_k, d_k)
        Keys, one per position that can be attended to.
    V : numpy.ndarray of shape (n_k, d_v)
        Values, one per position that can be attended to.
    mask : numpy.ndarray of bool, optional
        Broadcastable to ``(n_q, n_k)``. False entries are excluded from
        attention. See :func:`causal_mask` and :func:`padding_mask`.

    Returns
    -------
    output : numpy.ndarray of shape (n_q, d_v)
        For each query, the attention-weighted average of the values.
    weights : numpy.ndarray of shape (n_q, n_k)
        Attention weights. Each row sums to 1, or to 0 if fully masked.

    Examples
    --------
    With a causal mask the first token can only attend to itself:

    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=(4, 8))
    >>> out, w = scaled_dot_product_attention(x, x, x, mask=causal_mask(4))
    >>> out.shape, w.shape
    ((4, 8), (4, 4))
    >>> w[0].tolist()
    [1.0, 0.0, 0.0, 0.0]
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)  # (n_q, n_k)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


class MultiHeadAttention:
    r"""Multi-head attention with learned projections.

    A single attention head can only form one weighted average per position.
    Multi-head attention runs ``h`` heads in parallel, each in its own
    learned subspace of size ``d_k = d_model / h``, so that different heads
    can track different relations (syntax, coreference, position). Their
    outputs are concatenated and mixed by an output projection:

    .. math::

        \operatorname{MultiHead}(X_q, X_{kv})
            = \operatorname{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_O,
        \qquad
        \text{head}_i = \operatorname{Attention}(X_q W_Q^{(i)},
                                                  X_{kv} W_K^{(i)},
                                                  X_{kv} W_V^{(i)}).

    The per-head matrices :math:`W_Q^{(i)}` are stored side by side as the
    column blocks of one ``(d_model, d_model)`` matrix, which is how most
    implementations lay them out.

    Parameters
    ----------
    d_model : int
        Model dimension.
    num_heads : int
        Number of heads ``h``. Must divide `d_model`.
    rng : numpy.random.Generator
        Source of randomness for weight initialization.

    Attributes
    ----------
    W_Q, W_K, W_V, W_O : numpy.ndarray of shape (d_model, d_model)
        Query, key, value and output projections. Initialized from
        :math:`\mathcal{N}(0, 1/d_{model})`. Biases are omitted, as in many
        modern models.

    Raises
    ------
    ValueError
        If `d_model` is not divisible by `num_heads`.

    Examples
    --------
    Cross-attention from 7 decoder positions to 9 encoder positions, with the
    "Transformer base" sizes:

    >>> mha = MultiHeadAttention(d_model=512, num_heads=8, rng=np.random.default_rng(0))
    >>> rng = np.random.default_rng(1)
    >>> out, weights = mha(rng.normal(size=(7, 512)), rng.normal(size=(9, 512)))
    >>> out.shape, weights.shape
    ((7, 512), (8, 7, 9))
    >>> mha.num_parameters()
    1048576
    """

    def __init__(self, d_model: int, num_heads: int, rng: np.random.Generator):
        if d_model % num_heads:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_model // num_heads
        std = 1.0 / np.sqrt(d_model)
        self.W_Q = rng.normal(0, std, (d_model, d_model))
        self.W_K = rng.normal(0, std, (d_model, d_model))
        self.W_V = rng.normal(0, std, (d_model, d_model))
        self.W_O = rng.normal(0, std, (d_model, d_model))

    def num_parameters(self) -> int:
        """Return the number of learned weights, ``4 · d_model²``.

        The count does not depend on the number of heads: more heads means
        smaller heads, not more parameters.
        """
        return 4 * self.d_model * self.d_model

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """Reshape ``(n, d_model)`` into ``(h, n, d_k)``, one slice per head."""
        n = x.shape[0]
        return x.reshape(n, self.h, self.d_k).transpose(1, 0, 2)

    def __call__(
        self,
        x_q: np.ndarray,
        x_kv: np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply multi-head attention.

        Parameters
        ----------
        x_q : numpy.ndarray of shape (n_q, d_model)
            Sequence that produces the queries.
        x_kv : numpy.ndarray of shape (n_k, d_model), optional
            Sequence that produces the keys and values. Defaults to `x_q`
            (self-attention). Pass the encoder output here for
            cross-attention.
        mask : numpy.ndarray of bool, optional
            Broadcastable to ``(n_q, n_k)``; shared by all heads.

        Returns
        -------
        output : numpy.ndarray of shape (n_q, d_model)
            Attention output after the output projection.
        weights : numpy.ndarray of shape (h, n_q, n_k)
            Attention weights of every head, useful for visualization.
        """
        if x_kv is None:
            x_kv = x_q
        Q = self._split_heads(x_q @ self.W_Q)  # (h, n_q, d_k)
        K = self._split_heads(x_kv @ self.W_K)  # (h, n_k, d_k)
        V = self._split_heads(x_kv @ self.W_V)  # (h, n_k, d_k)

        heads, weights = [], []
        for i in range(self.h):
            out, w = scaled_dot_product_attention(Q[i], K[i], V[i], mask)
            heads.append(out)
            weights.append(w)

        concat = np.concatenate(heads, axis=-1)  # (n_q, d_model)
        return concat @ self.W_O, np.stack(weights)
