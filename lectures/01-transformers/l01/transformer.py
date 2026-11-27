r"""The original encoder-decoder Transformer, forward pass only.

This module assembles the pieces from :mod:`l01.attention` and
:mod:`l01.positional_encoding` into the architecture of Vaswani et al. [1]_:
a stack of encoder layers that turns the source sentence into context-aware
vectors, and a stack of decoder layers that generates the target sentence one
token at a time while attending to those vectors.

The weights are randomly initialized and never trained here, so the model's
predictions are meaningless. The goal is to make every tensor shape and every
connection in the architecture explicit; training is covered by the PyTorch
version of this lecture.

Notes
-----
The layers use **post-norm** residual blocks, as in the original paper.
Writing :math:`\mathrm{LN}` for layer normalization:

.. math::

    \text{Encoder layer:}\quad
    x &\leftarrow \mathrm{LN}(x + \mathrm{SelfAttn}(x)) \\
    x &\leftarrow \mathrm{LN}(x + \mathrm{FFN}(x))

.. math::

    \text{Decoder layer:}\quad
    y &\leftarrow \mathrm{LN}(y + \mathrm{MaskedSelfAttn}(y)) \\
    y &\leftarrow \mathrm{LN}(y + \mathrm{CrossAttn}(y, m)) \\
    y &\leftarrow \mathrm{LN}(y + \mathrm{FFN}(y))

where :math:`m` is the encoder output ("memory"). Most modern LLMs use
pre-norm instead, :math:`x \leftarrow x + \mathrm{Sublayer}(\mathrm{LN}(x))`,
which trains more stably in deep stacks.

The input, the decoder input and the output projection share a single
embedding matrix (weight tying, Section 3.4 of the paper). Embeddings are
multiplied by :math:`\sqrt{d_{model}}` before the positional encoding is
added.

References
----------
.. [1] A. Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
   https://arxiv.org/abs/1706.03762
.. [2] J. L. Ba, J. R. Kiros and G. E. Hinton, "Layer Normalization", 2016.
   https://arxiv.org/abs/1607.06450
.. [3] C. Szegedy et al., "Rethinking the Inception Architecture for Computer
   Vision", CVPR 2016, Section 7 (label smoothing).
   https://arxiv.org/abs/1512.00567

Examples
--------
>>> cfg = TransformerConfig(vocab_size=100, d_model=32, num_heads=4, d_ff=64, num_layers=2)
>>> model = Transformer(cfg)
>>> src = np.array([1, 17, 42, 8, 2])
>>> tgt = np.array([1, 5, 9])
>>> model.encode(src).shape
(5, 32)
>>> model(src, tgt).shape
(3, 100)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .attention import MultiHeadAttention, causal_mask
from .positional_encoding import sinusoidal_positional_encoding

__all__ = [
    "DecoderLayer",
    "EncoderLayer",
    "FeedForward",
    "LayerNorm",
    "Transformer",
    "TransformerConfig",
    "cross_entropy",
]


@dataclass(frozen=True)
class TransformerConfig:
    """Hyperparameters of an encoder-decoder Transformer.

    The defaults describe a small model that runs instantly in NumPy. The
    "Transformer base" model of the paper uses ``vocab_size≈37000``,
    ``d_model=512``, ``num_heads=8``, ``d_ff=2048`` and ``num_layers=6``.

    Attributes
    ----------
    vocab_size : int
        Number of tokens, shared by source and target.
    d_model : int
        Width of every token representation. Must be even and divisible by
        `num_heads`.
    num_heads : int
        Attention heads per attention block.
    d_ff : int
        Hidden width of the feed-forward network, typically ``4 · d_model``.
    num_layers : int
        Number of layers in the encoder and, separately, in the decoder.
    max_len : int
        Longest sequence the positional encoding table covers.
    """

    vocab_size: int = 1000
    d_model: int = 64
    num_heads: int = 4
    d_ff: int = 256
    num_layers: int = 2
    max_len: int = 128


class LayerNorm:
    r"""Layer normalization over the feature dimension.

    Each position's vector is normalized to zero mean and unit variance,
    then rescaled by a learned gain :math:`\gamma` and shifted by a learned
    bias :math:`\beta`:

    .. math::

        \mathrm{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta.

    Unlike batch normalization, the statistics are computed per position,
    so the result does not depend on the batch or on the sequence length.

    Parameters
    ----------
    d_model : int
        Feature dimension.
    eps : float, default 1e-5
        Added to the variance for numerical stability.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Normalize `x` of shape ``(..., d_model)`` along its last axis."""
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta


class FeedForward:
    r"""Position-wise feed-forward network.

    .. math::

        \mathrm{FFN}(x) = \max(0,\; x W_1 + b_1)\, W_2 + b_2

    The same two-layer MLP is applied to every position independently. It
    widens each vector to `d_ff`, applies a ReLU and projects back. Attention
    mixes information *between* positions; the FFN transforms it *within*
    each position, and holds about two thirds of a layer's parameters.

    Parameters
    ----------
    d_model : int
        Input and output width.
    d_ff : int
        Hidden width.
    rng : numpy.random.Generator
        Source of randomness for weight initialization.
    """

    def __init__(self, d_model: int, d_ff: int, rng: np.random.Generator):
        self.W1 = rng.normal(0, 1 / np.sqrt(d_model), (d_model, d_ff))
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.normal(0, 1 / np.sqrt(d_ff), (d_ff, d_model))
        self.b2 = np.zeros(d_model)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the FFN to `x` of shape ``(n, d_model)``; returns the same shape."""
        return np.maximum(0, x @ self.W1 + self.b1) @ self.W2 + self.b2


class EncoderLayer:
    """One encoder layer: self-attention, then a feed-forward network.

    Every source position attends to every other source position, in both
    directions. Each sub-layer is wrapped in a residual connection followed
    by layer normalization.

    Parameters
    ----------
    cfg : TransformerConfig
        Model hyperparameters.
    rng : numpy.random.Generator
        Source of randomness for weight initialization.
    """

    def __init__(self, cfg: TransformerConfig, rng: np.random.Generator):
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.num_heads, rng)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, rng)
        self.ln1 = LayerNorm(cfg.d_model)
        self.ln2 = LayerNorm(cfg.d_model)

    def __call__(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Transform source vectors.

        Parameters
        ----------
        x : numpy.ndarray of shape (n_src, d_model)
            Source representations.
        mask : numpy.ndarray of bool, optional
            Key mask, typically a :func:`~l01.attention.padding_mask`.

        Returns
        -------
        numpy.ndarray of shape (n_src, d_model)
            Updated source representations.
        """
        attn, _ = self.self_attn(x, mask=mask)
        x = self.ln1(x + attn)
        return self.ln2(x + self.ffn(x))


class DecoderLayer:
    """One decoder layer: masked self-attention, cross-attention, feed-forward.

    The masked self-attention lets each target position see only earlier
    target positions, so that the model cannot copy the token it is asked to
    predict. Cross-attention then takes its queries from the decoder and its
    keys and values from the encoder output, which is how information flows
    from the source sentence into the translation.

    Parameters
    ----------
    cfg : TransformerConfig
        Model hyperparameters.
    rng : numpy.random.Generator
        Source of randomness for weight initialization.
    """

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
        """Transform target vectors, conditioned on the encoder output.

        Parameters
        ----------
        y : numpy.ndarray of shape (n_tgt, d_model)
            Target representations.
        memory : numpy.ndarray of shape (n_src, d_model)
            Encoder output.
        memory_mask : numpy.ndarray of bool, optional
            Mask over the source positions, typically hiding padding.

        Returns
        -------
        numpy.ndarray of shape (n_tgt, d_model)
            Updated target representations.
        """
        n = y.shape[0]
        attn, _ = self.self_attn(y, mask=causal_mask(n))
        y = self.ln1(y + attn)
        cross, _ = self.cross_attn(y, memory, mask=memory_mask)
        y = self.ln2(y + cross)
        return self.ln3(y + self.ffn(y))


class Transformer:
    """Encoder-decoder Transformer for sequence-to-sequence tasks.

    Parameters
    ----------
    cfg : TransformerConfig
        Model hyperparameters.
    seed : int, default 0
        Seed for weight initialization.

    Attributes
    ----------
    embedding : numpy.ndarray of shape (vocab_size, d_model)
        Token embedding matrix, shared by the encoder input, the decoder
        input and the output projection.
    pe : numpy.ndarray of shape (max_len, d_model)
        Precomputed sinusoidal positional encodings.
    encoder, decoder : list
        The stacks of :class:`EncoderLayer` and :class:`DecoderLayer`.
    """

    def __init__(self, cfg: TransformerConfig, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.cfg = cfg
        self.embedding = rng.normal(0, 1 / np.sqrt(cfg.d_model), (cfg.vocab_size, cfg.d_model))
        self.pe = sinusoidal_positional_encoding(cfg.max_len, cfg.d_model)
        self.encoder = [EncoderLayer(cfg, rng) for _ in range(cfg.num_layers)]
        self.decoder = [DecoderLayer(cfg, rng) for _ in range(cfg.num_layers)]

    def embed(self, ids: np.ndarray) -> np.ndarray:
        """Look up token embeddings and add positional encodings.

        Parameters
        ----------
        ids : numpy.ndarray of int, shape (n,)
            Token ids. ``n`` must not exceed ``cfg.max_len``.

        Returns
        -------
        numpy.ndarray of shape (n, d_model)
            ``embedding[ids] · √d_model + PE[:n]``.
        """
        n = len(ids)
        return self.embedding[ids] * np.sqrt(self.cfg.d_model) + self.pe[:n]

    def encode(self, src_ids: np.ndarray, src_mask: np.ndarray | None = None) -> np.ndarray:
        """Run the encoder stack.

        Parameters
        ----------
        src_ids : numpy.ndarray of int, shape (n_src,)
            Source token ids.
        src_mask : numpy.ndarray of bool, optional
            Mask over source positions, typically hiding padding.

        Returns
        -------
        numpy.ndarray of shape (n_src, d_model)
            One context-aware vector per source token (the "memory").
        """
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
        """Run the decoder stack and project to vocabulary logits.

        Parameters
        ----------
        tgt_ids : numpy.ndarray of int, shape (n_tgt,)
            Target token ids generated so far, starting with ``[BOS]``.
        memory : numpy.ndarray of shape (n_src, d_model)
            Encoder output from :meth:`encode`.
        src_mask : numpy.ndarray of bool, optional
            Mask over source positions, typically hiding padding.

        Returns
        -------
        numpy.ndarray of shape (n_tgt, vocab_size)
            Unnormalized scores. Row ``t`` scores the token at position
            ``t + 1``, so during training the targets are ``tgt_ids``
            shifted left by one (teacher forcing).
        """
        y = self.embed(tgt_ids)
        for layer in self.decoder:
            y = layer(y, memory, src_mask)
        return y @ self.embedding.T

    def __call__(self, src_ids: np.ndarray, tgt_ids: np.ndarray) -> np.ndarray:
        """Encode `src_ids` and decode `tgt_ids`; see :meth:`decode` for the output."""
        return self.decode(tgt_ids, self.encode(src_ids))

    def greedy_decode(
        self, src_ids: np.ndarray, bos_id: int, eos_id: int, max_new: int
    ) -> list[int]:
        """Generate a target sequence by always choosing the most likely token.

        The source is encoded once. The decoder is then run repeatedly on the
        tokens generated so far, and the arg-max of the last row of logits is
        appended, until ``[EOS]`` is produced or `max_new` tokens have been
        generated.

        Parameters
        ----------
        src_ids : numpy.ndarray of int, shape (n_src,)
            Source token ids.
        bos_id : int
            Id of the beginning-of-sequence token that starts generation.
        eos_id : int
            Id of the end-of-sequence token that stops generation.
        max_new : int
            Maximum number of tokens to generate.

        Returns
        -------
        list of int
            ``[bos_id, ...]``, ending in `eos_id` unless the limit was hit.

        Notes
        -----
        Re-running the decoder on the full prefix costs quadratic time in
        the output length. Real systems cache the keys and values of past
        positions instead (KV caching, Lecture 05).
        """
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
    r"""Mean cross-entropy loss, with optional label smoothing.

    Parameters
    ----------
    logits : numpy.ndarray of shape (n, V)
        Unnormalized scores for each of ``n`` positions.
    targets : numpy.ndarray of int, shape (n,)
        Correct token id at each position.
    label_smoothing : float, default 0.0
        Smoothing strength :math:`\epsilon`. The paper uses 0.1.

    Returns
    -------
    float
        Loss averaged over positions, in nats.

    Notes
    -----
    Label smoothing replaces the one-hot target with

    .. math::

        q'(k) = (1 - \epsilon)\, \mathbb{1}[k = y] + \frac{\epsilon}{V},

    which penalizes over-confident predictions. It raises perplexity but
    tends to improve accuracy and BLEU.

    Examples
    --------
    Uniform logits give a loss of :math:`\log V`:

    >>> V = 50
    >>> loss = cross_entropy(np.zeros((4, V)), np.array([0, 1, 2, 3]))
    >>> bool(np.isclose(loss, np.log(V)))
    True
    """
    n, vocab = logits.shape
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    q = np.full((n, vocab), label_smoothing / vocab)
    q[np.arange(n), targets] += 1.0 - label_smoothing
    return float(-(q * log_probs).sum(axis=-1).mean())
