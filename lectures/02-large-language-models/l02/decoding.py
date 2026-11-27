r"""Decoding strategies: turning next-token probabilities into text.

A language model only scores the next token. Generating text means choosing
a token, appending it to the context and repeating, and *how* the token is
chosen changes the output as much as the model does:

- **Greedy search** always takes the most likely token. It is deterministic
  and can miss a better sequence whose first token looked worse.
- **Beam search** keeps the ``k`` most likely partial sequences at every
  step. It finds higher-probability sequences than greedy search, at ``k``
  times the cost, but is still deterministic, and maximizing likelihood
  tends to produce bland, repetitive text [1]_.
- **Sampling** draws the next token at random from the model's
  distribution, usually after reshaping it: the *temperature* sharpens or
  flattens it, and *top-k* or *top-p* (nucleus) filtering [1]_ removes the
  unreliable tail before sampling.

Notes
-----
With logits :math:`x_1, \dots, x_V` and temperature :math:`T > 0`,

.. math::

    P_T(w_i) = \frac{\exp(x_i / T)}{\sum_j \exp(x_j / T)}.

:math:`T \to 0` approaches greedy search, :math:`T = 1` leaves the
distribution unchanged and :math:`T \to \infty` approaches the uniform
distribution.

*Top-k* keeps the :math:`k` most likely tokens. *Top-p* keeps the smallest
set of most likely tokens whose total probability is at least :math:`p`, so
the number of candidates adapts to the model's confidence: one token when
the distribution is peaked, many when it is flat. Both renormalize the kept
probabilities before sampling. The order used here, as in most libraries,
is: divide logits by :math:`T`, softmax, top-k, top-p, sample.

Beam search here ranks sequences by their total log-probability, without
the length penalty that translation systems often add (the original
Transformer used beam size 4 with a length penalty of 0.6 [2]_).

References
----------
.. [1] A. Holtzman, J. Buys, L. Du, M. Forbes and Y. Choi, "The Curious Case
   of Neural Text Degeneration", ICLR 2020. https://arxiv.org/abs/1904.09751
.. [2] A. Vaswani et al., "Attention Is All You Need", NeurIPS 2017,
   Section 6.1. https://arxiv.org/abs/1706.03762

Examples
--------
>>> logits = np.array([3.0, 2.0, 1.0, 0.0])
>>> temperature_softmax(logits, 0.5).round(3).tolist()
[0.865, 0.117, 0.016, 0.002]
>>> temperature_softmax(logits, 2.0).round(3).tolist()
[0.455, 0.276, 0.167, 0.102]
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from l01.attention import softmax

__all__ = [
    "beam_search",
    "greedy_search",
    "sample_next_token",
    "temperature_softmax",
    "top_k_filter",
    "top_p_filter",
]

LogProbFn = Callable[[list[int]], np.ndarray]
"""Maps a token prefix to log-probabilities of every next token, shape (V,)."""


def temperature_softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Convert logits to probabilities at a given temperature.

    Parameters
    ----------
    logits : numpy.ndarray of shape (V,)
        Unnormalized next-token scores.
    temperature : float
        Must be positive. Values below 1 sharpen the distribution, values
        above 1 flatten it.

    Returns
    -------
    numpy.ndarray of shape (V,)
        Probabilities summing to 1.

    Raises
    ------
    ValueError
        If `temperature` is not positive. Use greedy search for the
        ``T = 0`` limit.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return softmax(logits / temperature)


def top_k_filter(probs: np.ndarray, k: int) -> np.ndarray:
    """Keep the `k` most likely tokens and renormalize.

    Parameters
    ----------
    probs : numpy.ndarray of shape (V,)
        Probabilities.
    k : int
        Number of tokens to keep.

    Returns
    -------
    numpy.ndarray of shape (V,)
        Probabilities that are zero outside the top `k` and sum to 1.

    Examples
    --------
    >>> top_k_filter(np.array([0.5, 0.25, 0.15, 0.1]), 2).round(3).tolist()
    [0.667, 0.333, 0.0, 0.0]
    """
    keep = np.argsort(-probs, kind="stable")[:k]
    out = np.zeros_like(probs)
    out[keep] = probs[keep]
    return out / out.sum()


def top_p_filter(probs: np.ndarray, p: float) -> np.ndarray:
    """Keep the smallest set of most likely tokens with total probability at least `p`.

    Parameters
    ----------
    probs : numpy.ndarray of shape (V,)
        Probabilities.
    p : float
        Probability mass to keep, in (0, 1].

    Returns
    -------
    numpy.ndarray of shape (V,)
        Probabilities that are zero outside the nucleus and sum to 1.

    Examples
    --------
    >>> top_p_filter(np.array([0.5, 0.25, 0.15, 0.07, 0.03]), 0.85).round(3).tolist()
    [0.556, 0.278, 0.167, 0.0, 0.0]
    """
    order = np.argsort(-probs, kind="stable")
    cumulative = np.cumsum(probs[order])
    last = min(int(np.searchsorted(cumulative, p)), len(probs) - 1)
    keep = order[: last + 1]
    out = np.zeros_like(probs)
    out[keep] = probs[keep]
    return out / out.sum()


def sample_next_token(
    logits: np.ndarray,
    rng: np.random.Generator,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> int:
    """Choose the next token by sampling from the reshaped distribution.

    Parameters
    ----------
    logits : numpy.ndarray of shape (V,)
        Next-token scores.
    rng : numpy.random.Generator
        Source of randomness.
    temperature : float, default 1.0
        Sampling temperature. 0 means greedy: return the arg-max.
    top_k : int, optional
        If given, sample only among the `top_k` most likely tokens.
    top_p : float, optional
        If given, sample only from the nucleus of mass `top_p`.

    Returns
    -------
    int
        The chosen token id.
    """
    if temperature == 0:
        return int(np.argmax(logits))
    probs = temperature_softmax(logits, temperature)
    if top_k is not None:
        probs = top_k_filter(probs, top_k)
    if top_p is not None:
        probs = top_p_filter(probs, top_p)
    return int(rng.choice(len(probs), p=probs))


def beam_search(
    log_prob_fn: LogProbFn,
    bos_id: int,
    eos_id: int,
    beam_size: int,
    max_len: int,
) -> tuple[list[int], float]:
    """Find a high-probability sequence by keeping the best `beam_size` prefixes.

    At every step each unfinished prefix is extended by its most likely
    next tokens, and only the `beam_size` best candidates overall survive.
    A candidate that ends with `eos_id` is finished and stops growing.

    Parameters
    ----------
    log_prob_fn : callable
        Maps a prefix (list of token ids) to next-token log-probabilities of
        shape ``(V,)``.
    bos_id : int
        Token that starts every sequence.
    eos_id : int
        Token that ends a sequence.
    beam_size : int
        Number of prefixes kept at each step. 1 gives greedy search.
    max_len : int
        Maximum number of tokens generated after `bos_id`.

    Returns
    -------
    tokens : list of int
        The best sequence found, starting with `bos_id`.
    log_prob : float
        Its total log-probability.
    """
    beams: list[tuple[list[int], float]] = [([bos_id], 0.0)]
    finished: list[tuple[list[int], float]] = []
    for _ in range(max_len):
        candidates = []
        for tokens, score in beams:
            log_probs = log_prob_fn(tokens)
            for token in np.argsort(-log_probs, kind="stable")[:beam_size]:
                if np.isfinite(log_probs[token]):
                    candidates.append(([*tokens, int(token)], score + float(log_probs[token])))
        candidates.sort(key=lambda c: -c[1])
        beams = []
        for tokens, score in candidates[:beam_size]:
            (finished if tokens[-1] == eos_id else beams).append((tokens, score))
        if not beams:
            break
    return max(finished + beams, key=lambda c: c[1])


def greedy_search(
    log_prob_fn: LogProbFn, bos_id: int, eos_id: int, max_len: int
) -> tuple[list[int], float]:
    """Generate by always taking the most likely next token.

    Equivalent to :func:`beam_search` with ``beam_size=1``; ties go to the
    lower token id.

    Parameters
    ----------
    log_prob_fn : callable
        Maps a prefix to next-token log-probabilities of shape ``(V,)``.
    bos_id, eos_id : int
        Start and end tokens.
    max_len : int
        Maximum number of generated tokens.

    Returns
    -------
    tokens : list of int
        The generated sequence, starting with `bos_id`.
    log_prob : float
        Its total log-probability.
    """
    return beam_search(log_prob_fn, bos_id, eos_id, beam_size=1, max_len=max_len)
