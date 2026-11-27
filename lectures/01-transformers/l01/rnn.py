r"""A vanilla recurrent neural network and its vanishing gradients.

Before attention, sequences were modelled with recurrent networks that read
one token at a time and carry a hidden state forward:

.. math::

    h_t = \tanh(x_t W_{xh} + h_{t-1} W_{hh} + b).

This design has two weaknesses that motivate the Transformer:

1. **Sequential computation.** :math:`h_t` depends on :math:`h_{t-1}`, so the
   time steps of a sequence cannot be processed in parallel.
2. **Vanishing gradients.** The gradient of :math:`h_T` with respect to an
   early state :math:`h_t` is a product of :math:`T - t` Jacobians. When
   these are contractive the product shrinks geometrically, so the network
   struggles to learn dependencies between distant tokens. LSTMs [1]_
   mitigate this with gating; attention removes the long path entirely.

This module uses the row-vector convention throughout: inputs are rows and
are multiplied on the right by weight matrices.

References
----------
.. [1] S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory",
   Neural Computation 9(8), 1997.
.. [2] Y. Bengio, P. Simard and P. Frasconi, "Learning Long-Term Dependencies
   with Gradient Descent is Difficult", IEEE Transactions on Neural
   Networks 5(2), 1994.
"""

from __future__ import annotations

import numpy as np

__all__ = ["VanillaRNN"]


class VanillaRNN:
    """Elman-style RNN with a tanh non-linearity (forward pass only).

    Parameters
    ----------
    input_dim : int
        Size of each input vector.
    hidden_dim : int
        Size of the hidden state.
    seed : int, default 0
        Seed for weight initialization.
    scale : float, default 0.5
        Standard deviation multiplier for the weights. Values below 1 make
        the recurrence contractive and the gradients vanish; values well
        above 1 tend to make them explode.

    Attributes
    ----------
    W_xh : numpy.ndarray of shape (input_dim, hidden_dim)
        Input-to-hidden weights.
    W_hh : numpy.ndarray of shape (hidden_dim, hidden_dim)
        Hidden-to-hidden (recurrent) weights.
    b : numpy.ndarray of shape (hidden_dim,)
        Hidden bias.

    Examples
    --------
    >>> rnn = VanillaRNN(input_dim=4, hidden_dim=16)
    >>> x = np.random.default_rng(1).normal(size=(30, 4))
    >>> rnn.forward(x).shape
    (30, 16)
    >>> norms = rnn.gradient_norms(x)
    >>> bool(norms[0] < 1e-3 * norms[-1])  # the first token barely influences h_T
    True
    """

    def __init__(self, input_dim: int, hidden_dim: int, seed: int = 0, scale: float = 0.5):
        rng = np.random.default_rng(seed)
        self.W_xh = rng.normal(0, scale / np.sqrt(input_dim), (input_dim, hidden_dim))
        self.W_hh = rng.normal(0, scale / np.sqrt(hidden_dim), (hidden_dim, hidden_dim))
        self.b = np.zeros(hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run the recurrence over a sequence, starting from a zero state.

        Parameters
        ----------
        x : numpy.ndarray of shape (n, input_dim)
            Input sequence of length ``n``.

        Returns
        -------
        numpy.ndarray of shape (n, hidden_dim)
            The hidden state after every time step.
        """
        h = np.zeros(self.hidden_dim)
        states = []
        for x_t in x:
            h = np.tanh(x_t @ self.W_xh + h @ self.W_hh + self.b)
            states.append(h)
        return np.stack(states)

    def gradient_norms(self, x: np.ndarray) -> np.ndarray:
        r"""Measure how strongly each hidden state influences the last one.

        Parameters
        ----------
        x : numpy.ndarray of shape (n, input_dim)
            Input sequence of length ``n``.

        Returns
        -------
        numpy.ndarray of shape (n,)
            Entry ``t`` is the spectral norm of the Jacobian
            :math:`\partial h_{n-1} / \partial h_t`. The last entry is
            always 1.

        Notes
        -----
        Each step contributes the Jacobian

        .. math::

            \frac{\partial h_t}{\partial h_{t-1}}
                = \operatorname{diag}(1 - h_t^2)\, W_{hh}^\top,

        and the chain rule multiplies them from :math:`t = n - 1` backwards.
        """
        states = self.forward(x)
        n = len(states)
        norms = np.zeros(n)
        jac = np.eye(self.hidden_dim)  # ∂h_{n-1} / ∂h_{n-1}
        norms[-1] = np.linalg.norm(jac, 2)
        for t in range(n - 1, 0, -1):
            step = (1 - states[t] ** 2)[:, None] * self.W_hh.T  # ∂h_t / ∂h_{t-1}
            jac = jac @ step
            norms[t - 1] = np.linalg.norm(jac, 2)
        return norms
