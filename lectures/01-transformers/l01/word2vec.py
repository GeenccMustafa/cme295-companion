r"""Skip-gram word2vec with negative sampling.

Word2vec [1]_ learns dense word vectors from raw text through a *proxy task*:
the task itself is never used again, but solving it forces words that occur
in similar contexts to receive similar vectors. In the skip-gram variant the
proxy task is to predict the words surrounding a given center word.

A full softmax over the vocabulary would cost O(V) per training pair.
Negative sampling [2]_ replaces it with ``k + 1`` binary classifications:
the observed context word should be scored as a real neighbour, and ``k``
randomly drawn words should be scored as fake ones.

Notes
-----
For a center word with input vector :math:`v_c`, an observed context word
with output vector :math:`u_o` and negative samples :math:`u_1, \dots, u_k`,
the loss of one training pair is

.. math::

    \mathcal{L} = -\log \sigma(u_o^\top v_c)
                   - \sum_{j=1}^{k} \log \sigma(-u_j^\top v_c),

where :math:`\sigma` is the logistic sigmoid. Its gradients are

.. math::

    \frac{\partial \mathcal{L}}{\partial v_c}
        = (\sigma(u_o^\top v_c) - 1)\, u_o
          + \sum_j \sigma(u_j^\top v_c)\, u_j,

    \frac{\partial \mathcal{L}}{\partial u_o}
        = (\sigma(u_o^\top v_c) - 1)\, v_c, \qquad
    \frac{\partial \mathcal{L}}{\partial u_j}
        = \sigma(u_j^\top v_c)\, v_c.

Only the input vectors are kept as the final embeddings.

This implementation draws negatives uniformly. The original paper samples
them from the unigram distribution raised to the power 3/4, and also
subsamples very frequent words; both are omitted for clarity.

References
----------
.. [1] T. Mikolov, K. Chen, G. Corrado and J. Dean, "Efficient Estimation of
   Word Representations in Vector Space", 2013. https://arxiv.org/abs/1301.3781
.. [2] T. Mikolov, I. Sutskever, K. Chen, G. Corrado and J. Dean,
   "Distributed Representations of Words and Phrases and their
   Compositionality", NeurIPS 2013. https://arxiv.org/abs/1310.4546
"""

from __future__ import annotations

import numpy as np

__all__ = ["SkipGram", "sigmoid", "skipgram_pairs"]


def skipgram_pairs(token_ids: list[int], window: int) -> list[tuple[int, int]]:
    """Generate (center, context) training pairs from a token sequence.

    Parameters
    ----------
    token_ids : list of int
        A tokenized sentence.
    window : int
        Maximum distance between a center word and a context word.

    Returns
    -------
    list of tuple of (int, int)
        Every pair ``(token_ids[i], token_ids[j])`` with
        ``0 < |i - j| <= window``, ordered by ``i`` then ``j``.

    Examples
    --------
    >>> skipgram_pairs([10, 11, 12], window=1)
    [(10, 11), (11, 10), (11, 12), (12, 11)]
    """
    pairs = []
    for i, center in enumerate(token_ids):
        for j in range(max(0, i - window), min(len(token_ids), i + window + 1)):
            if j != i:
                pairs.append((center, token_ids[j]))
    return pairs


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute the logistic sigmoid ``1 / (1 + exp(-x))`` element-wise.

    Parameters
    ----------
    x : numpy.ndarray
        Input array of any shape.

    Returns
    -------
    numpy.ndarray
        Values in the open interval (0, 1), same shape as `x`.
    """
    return 1.0 / (1.0 + np.exp(-x))


class SkipGram:
    """Skip-gram model trained with negative sampling and plain SGD.

    Parameters
    ----------
    vocab_size : int
        Number of distinct tokens ``V``.
    dim : int
        Embedding dimension ``d``.
    seed : int, default 0
        Seed for weight initialization and negative sampling.

    Attributes
    ----------
    W_in : numpy.ndarray of shape (V, d)
        Input (center-word) vectors. These are the learned embeddings.
    W_out : numpy.ndarray of shape (V, d)
        Output (context-word) vectors, used only during training.

    Examples
    --------
    Two groups of words that never share a context end up far apart:

    >>> sentences = [[0, 1, 2, 0, 2, 1], [3, 4, 5, 3, 5, 4]] * 20
    >>> pairs = [p for s in sentences for p in skipgram_pairs(s, window=2)]
    >>> model = SkipGram(vocab_size=6, dim=8)
    >>> history = model.train(pairs, epochs=15, num_negatives=2)
    >>> bool(history[-1] < history[0])
    True
    >>> bool(model.similarity(0, 1) > model.similarity(0, 4))
    True
    """

    def __init__(self, vocab_size: int, dim: int, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.W_in = self.rng.normal(0, 0.1, (vocab_size, dim))
        self.W_out = self.rng.normal(0, 0.1, (vocab_size, dim))
        self.vocab_size = vocab_size

    def loss_and_step(self, center: int, context: int, num_negatives: int, lr: float) -> float:
        """Take one SGD step on a single (center, context) pair.

        Parameters
        ----------
        center : int
            Id of the center word.
        context : int
            Id of an observed context word.
        num_negatives : int
            Number of negative samples ``k``.
        lr : float
            Learning rate.

        Returns
        -------
        float
            The loss of this pair, evaluated before the update.
        """
        negatives = self.rng.integers(0, self.vocab_size, num_negatives)
        v = self.W_in[center]  # (d,)
        u_pos = self.W_out[context]  # (d,)
        u_neg = self.W_out[negatives]  # (k, d)

        s_pos = sigmoid(u_pos @ v)  # scalar
        s_neg = sigmoid(-(u_neg @ v))  # (k,)
        loss = -np.log(s_pos + 1e-10) - np.log(s_neg + 1e-10).sum()

        g_pos = s_pos - 1.0  # dL / d(u_o · v_c)
        g_neg = 1.0 - s_neg  # dL / d(u_j · v_c) = σ(u_j · v_c), shape (k,)
        grad_v = g_pos * u_pos + g_neg @ u_neg
        grad_u_pos = g_pos * v
        grad_u_neg = g_neg[:, None] * v[None, :]

        self.W_in[center] -= lr * grad_v
        self.W_out[context] -= lr * grad_u_pos
        # add.at accumulates correctly when the same negative is drawn twice.
        np.add.at(self.W_out, negatives, -lr * grad_u_neg)
        return float(loss)

    def train(
        self,
        pairs: list[tuple[int, int]],
        epochs: int = 50,
        num_negatives: int = 5,
        lr: float = 0.05,
    ) -> list[float]:
        """Train on a list of pairs, shuffling them every epoch.

        Parameters
        ----------
        pairs : list of tuple of (int, int)
            Training pairs, for example from :func:`skipgram_pairs`.
        epochs : int, default 50
            Number of passes over `pairs`.
        num_negatives : int, default 5
            Negative samples per pair.
        lr : float, default 0.05
            Learning rate.

        Returns
        -------
        list of float
            Mean loss of each epoch, useful for plotting a learning curve.
        """
        history = []
        for _ in range(epochs):
            order = self.rng.permutation(len(pairs))
            losses = [
                self.loss_and_step(*pairs[i], num_negatives=num_negatives, lr=lr) for i in order
            ]
            history.append(float(np.mean(losses)))
        return history

    def similarity(self, a: int, b: int) -> float:
        """Return the cosine similarity between the embeddings of two words.

        Parameters
        ----------
        a, b : int
            Token ids.

        Returns
        -------
        float
            A value in [-1, 1]; 1 means the vectors point the same way.
        """
        va, vb = self.W_in[a], self.W_in[b]
        return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))
