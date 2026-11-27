r"""Rotary position embeddings (RoPE).

The 2017 Transformer adds an absolute position vector to each token
embedding. What attention really needs, however, is the *relative* position
of a query and a key: "the word two places back" means the same thing at the
start of a document and in the middle of it. Several methods inject relative
position directly into the attention scores (T5 learns a bias per distance,
ALiBi adds a fixed linear penalty). RoPE [1]_ does it by *rotating* queries
and keys by an angle proportional to their position, and has become the
default in LLMs.

Notes
-----
For a head dimension :math:`d` (even), the coordinates of a vector are split
into :math:`d/2` adjacent pairs :math:`(x_{2i}, x_{2i+1})`. At position
:math:`m`, pair :math:`i` is rotated by the angle :math:`m\,\theta_i`, with
frequencies

.. math::

    \theta_i = b^{-2i/d}, \qquad i = 0, \dots, d/2 - 1,

and base :math:`b = 10000` in the paper (``rope_theta`` in model configs).
In matrix form, :math:`R_m` is block diagonal with 2×2 blocks

.. math::

    \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\
                    \sin m\theta_i & \cos m\theta_i \end{pmatrix}.

The rotation is applied to queries and keys (never to values) after their
projections, inside every attention layer. Because rotations are orthogonal
and :math:`R_m^\top R_n = R_{n-m}`,

.. math::

    (R_m q)^\top (R_n k) = q^\top R_{n-m}\, k,

so the attention score depends on the positions only through their
difference :math:`n - m`. Low-index pairs rotate quickly and resolve nearby
positions; high-index pairs rotate slowly and carry long-range information.
Raising the base slows every rotation down, which is one way models extend
their context length.

*Pairing convention.* This module pairs adjacent coordinates, as in the
paper (equation 15). Many released implementations instead pair coordinate
:math:`i` with :math:`i + d/2` (the ``rotate_half`` trick). The two are the
same operation applied to a permuted vector, so they are equivalent as long
as weights are trained and run with the same convention, but weights cannot
be moved between the conventions without permuting the projection columns.

References
----------
.. [1] J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen and Y. Liu, "RoFormer:
   Enhanced Transformer with Rotary Position Embedding", 2021,
   Section 3.2, equations 14–16. https://arxiv.org/abs/2104.09864

Examples
--------
Shifting a query and a key by the same number of positions leaves their
score unchanged:

>>> rng = np.random.default_rng(0)
>>> q, k = rng.normal(size=(2, 1, 8))
>>> def score(m, n):
...     return (apply_rope(q, np.array([m])) @ apply_rope(k, np.array([n])).T).item()
>>> bool(np.isclose(score(3, 1), score(103, 101)))
True
"""

from __future__ import annotations

import numpy as np

__all__ = ["apply_rope", "rope_frequencies", "rotation_matrix"]


def rope_frequencies(head_dim: int, base: float = 10000.0) -> np.ndarray:
    """Return the rotation frequency of every coordinate pair.

    Parameters
    ----------
    head_dim : int
        Dimension ``d`` of each query or key head. Must be even.
    base : float, default 10000.0
        Base ``b`` of the geometric schedule (``rope_theta``).

    Returns
    -------
    numpy.ndarray of shape (head_dim // 2,)
        ``theta_i = base ** (-2 i / head_dim)``, decreasing from 1.

    Raises
    ------
    ValueError
        If `head_dim` is odd.
    """
    if head_dim % 2:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    return base ** (-np.arange(0, head_dim, 2) / head_dim)


def apply_rope(x: np.ndarray, positions: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """Rotate query or key vectors according to their positions.

    Parameters
    ----------
    x : numpy.ndarray of shape (..., n, head_dim)
        Queries or keys, for example ``(n, d)`` for one head or
        ``(h, n, d)`` for several heads.
    positions : numpy.ndarray of int, shape (n,)
        Position of each of the ``n`` vectors.
    base : float, default 10000.0
        Base of the frequency schedule.

    Returns
    -------
    numpy.ndarray
        Rotated vectors, same shape as `x`.
    """
    theta = rope_frequencies(x.shape[-1], base)  # (d/2,)
    angles = positions[:, None] * theta[None, :]  # (n, d/2)
    cos, sin = np.cos(angles), np.sin(angles)
    x_even, x_odd = x[..., 0::2], x[..., 1::2]  # (..., n, d/2) each
    out = np.empty_like(x, dtype=np.result_type(x, np.float64))
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


def rotation_matrix(position: int, head_dim: int, base: float = 10000.0) -> np.ndarray:
    """Build the explicit rotation matrix ``R_m`` for one position.

    :func:`apply_rope` never materializes this matrix; it is provided to
    make the block-diagonal structure visible and to test against.

    Parameters
    ----------
    position : int
        Position ``m``.
    head_dim : int
        Dimension ``d``. Must be even.
    base : float, default 10000.0
        Base of the frequency schedule.

    Returns
    -------
    numpy.ndarray of shape (head_dim, head_dim)
        Orthogonal, block-diagonal matrix of 2×2 rotations.

    Examples
    --------
    With ``head_dim=2`` the only frequency is 1, so position 1 rotates by
    one radian:

    >>> rotation_matrix(1, 2).round(3).tolist()
    [[0.54, -0.841], [0.841, 0.54]]
    """
    R = np.zeros((head_dim, head_dim))
    for i, theta in enumerate(rope_frequencies(head_dim, base)):
        c, s = np.cos(position * theta), np.sin(position * theta)
        R[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = [[c, -s], [s, c]]
    return R
