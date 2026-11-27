r"""Sinusoidal positional encoding.

Self-attention is *permutation-equivariant*: shuffling the input tokens
shuffles the outputs in exactly the same way, so on its own the model has no
notion of word order. The original Transformer [1]_ fixes this by adding a
fixed, position-dependent vector to every token embedding before the first
layer.

Notes
-----
For position :math:`p` and dimension pair :math:`i \in \{0, \dots, d/2 - 1\}`,

.. math::

    PE_{p,\,2i}   = \sin\!\left(\frac{p}{10000^{2i/d}}\right), \qquad
    PE_{p,\,2i+1} = \cos\!\left(\frac{p}{10000^{2i/d}}\right).

Each dimension pair is a sine and cosine at its own frequency. Wavelengths
form a geometric progression from :math:`2\pi` (fast, fine-grained position)
to :math:`10000 \cdot 2\pi` (slow, coarse position). Because
:math:`PE_{p+k}` is a fixed linear function of :math:`PE_p` for every offset
:math:`k`, relative positions are easy for attention to pick up, and the
encoding is defined for positions longer than any seen in training.

References
----------
.. [1] A. Vaswani et al., "Attention Is All You Need", NeurIPS 2017,
   Section 3.5. https://arxiv.org/abs/1706.03762
"""

from __future__ import annotations

import numpy as np

__all__ = ["sinusoidal_positional_encoding"]


def sinusoidal_positional_encoding(max_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """Build the sinusoidal positional encoding table.

    Parameters
    ----------
    max_len : int
        Number of positions to encode.
    d_model : int
        Model dimension. Must be even, since dimensions come in sine/cosine
        pairs.
    base : float, default 10000.0
        Controls the longest wavelength, ``2π · base``.

    Returns
    -------
    numpy.ndarray of shape (max_len, d_model)
        Row ``p`` is the encoding of position ``p``. Even columns hold
        sines, odd columns hold cosines.

    Raises
    ------
    ValueError
        If `d_model` is odd.

    Examples
    --------
    Position 0 is ``[sin 0, cos 0, sin 0, cos 0, ...] = [0, 1, 0, 1, ...]``:

    >>> pe = sinusoidal_positional_encoding(max_len=50, d_model=4)
    >>> pe.shape
    (50, 4)
    >>> pe[0].tolist()
    [0.0, 1.0, 0.0, 1.0]
    """
    if d_model % 2:
        raise ValueError(f"d_model must be even, got {d_model}")
    pos = np.arange(max_len)[:, None]  # (max_len, 1)
    i = np.arange(d_model // 2)[None, :]  # (1, d_model / 2)
    angles = pos / base ** (2 * i / d_model)  # (max_len, d_model / 2)
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return pe
