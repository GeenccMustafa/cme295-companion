"""Skip-gram word2vec with negative sampling (Mikolov et al., 2013).

The proxy task: given a center word, predict the words around it. Doing this
well forces words that appear in similar contexts to get similar vectors.

Instead of a softmax over the whole vocabulary, negative sampling turns each
(center, context) pair into a binary classification: the true context word
should score high, `k` randomly drawn "negative" words should score low.

    loss = -log σ(u_o · v_c) - Σ_k log σ(-u_k · v_c)

v_c: input (center) embedding, u_o: output (context) embedding.
"""

from __future__ import annotations

import numpy as np


def skipgram_pairs(token_ids: list[int], window: int) -> list[tuple[int, int]]:
    """All (center, context) pairs within `window` positions of each other."""
    pairs = []
    for i, center in enumerate(token_ids):
        for j in range(max(0, i - window), min(len(token_ids), i + window + 1)):
            if j != i:
                pairs.append((center, token_ids[j]))
    return pairs


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class SkipGram:
    def __init__(self, vocab_size: int, dim: int, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.W_in = self.rng.normal(0, 0.1, (vocab_size, dim))  # (V, d): the embeddings we keep
        self.W_out = self.rng.normal(0, 0.1, (vocab_size, dim))  # (V, d): context vectors
        self.vocab_size = vocab_size

    def loss_and_step(self, center: int, context: int, num_negatives: int, lr: float) -> float:
        """One SGD step on a single (center, context) pair. Returns the loss before the step."""
        negatives = self.rng.integers(0, self.vocab_size, num_negatives)
        v = self.W_in[center]  # (d,)
        u_pos = self.W_out[context]  # (d,)
        u_neg = self.W_out[negatives]  # (k, d)

        s_pos = sigmoid(u_pos @ v)  # scalar
        s_neg = sigmoid(-(u_neg @ v))  # (k,)
        loss = -np.log(s_pos + 1e-10) - np.log(s_neg + 1e-10).sum()

        # Gradients of the loss above.
        g_pos = s_pos - 1.0  # d loss / d (u_pos · v)
        g_neg = 1.0 - s_neg  # d loss / d (u_neg · v), shape (k,)
        grad_v = g_pos * u_pos + g_neg @ u_neg
        grad_u_pos = g_pos * v
        grad_u_neg = g_neg[:, None] * v[None, :]

        self.W_in[center] -= lr * grad_v
        self.W_out[context] -= lr * grad_u_pos
        np.add.at(self.W_out, negatives, -lr * grad_u_neg)
        return float(loss)

    def train(
        self,
        pairs: list[tuple[int, int]],
        epochs: int = 50,
        num_negatives: int = 5,
        lr: float = 0.05,
    ) -> list[float]:
        """Returns the mean loss of each epoch."""
        history = []
        for _ in range(epochs):
            order = self.rng.permutation(len(pairs))
            losses = [
                self.loss_and_step(*pairs[i], num_negatives=num_negatives, lr=lr) for i in order
            ]
            history.append(float(np.mean(losses)))
        return history

    def similarity(self, a: int, b: int) -> float:
        """Cosine similarity between two input embeddings."""
        va, vb = self.W_in[a], self.W_in[b]
        return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))
