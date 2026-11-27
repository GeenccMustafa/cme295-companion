"""A vanilla RNN, to see the two problems attention was invented to fix.

    h_t = tanh(W_xh x_t + W_hh h_{t-1} + b)

1. It is sequential: h_t needs h_{t-1}, so the time steps cannot run in parallel.
2. Long-range information fades: the gradient from h_T back to h_1 is a product
   of T Jacobians, which tends to vanish (or explode) as T grows.
"""

from __future__ import annotations

import numpy as np


class VanillaRNN:
    def __init__(self, input_dim: int, hidden_dim: int, seed: int = 0, scale: float = 0.5):
        rng = np.random.default_rng(seed)
        self.W_xh = rng.normal(0, scale / np.sqrt(input_dim), (input_dim, hidden_dim))
        self.W_hh = rng.normal(0, scale / np.sqrt(hidden_dim), (hidden_dim, hidden_dim))
        self.b = np.zeros(hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (n, input_dim) -> all hidden states (n, hidden_dim). One step at a time."""
        h = np.zeros(self.hidden_dim)
        states = []
        for x_t in x:
            h = np.tanh(x_t @ self.W_xh + h @ self.W_hh + self.b)
            states.append(h)
        return np.stack(states)

    def gradient_norms(self, x: np.ndarray) -> np.ndarray:
        """‖∂h_T/∂h_t‖ for every t, computed by chaining the per-step Jacobians backwards.

        For a contractive W_hh these norms shrink geometrically as t moves away from T:
        the vanishing gradient problem.
        """
        states = self.forward(x)
        n = len(states)
        norms = np.zeros(n)
        jac = np.eye(self.hidden_dim)  # ∂h_T/∂h_T
        norms[-1] = np.linalg.norm(jac, 2)
        for t in range(n - 1, 0, -1):
            # ∂h_t/∂h_{t-1} = diag(1 - h_t²) W_hhᵀ
            # (row-vector convention: h_t = tanh(h_{t-1} W_hh + ...))
            step = (1 - states[t] ** 2)[:, None] * self.W_hh.T
            jac = jac @ step
            norms[t - 1] = np.linalg.norm(jac, 2)
        return norms
