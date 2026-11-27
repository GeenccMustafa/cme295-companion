r"""SwiGLU feed-forward network.

Every Transformer layer contains a position-wise feed-forward network (FFN)
that transforms each token's vector on its own. The 2017 Transformer used two
linear maps with a ReLU in between. Shazeer [1]_ compared *gated* variants, in
which one linear map is passed through an activation and multiplied
element-wise by a second linear map that acts as a learned gate. The SwiGLU
variant gave the best results and has become the standard FFN in LLMs: it is
used by LLaMA [2]_ and by every expert of Mixtral [4]_.

Notes
-----
With :math:`\mathrm{SiLU}(z) = z\, \sigma(z)` (Swish with :math:`\beta = 1`),

.. math::

    \mathrm{FFN}_{\mathrm{SwiGLU}}(x)
        = \big(\mathrm{SiLU}(x W_{\mathrm{gate}}) \odot x W_{\mathrm{up}}\big)\, W_{\mathrm{down}},

with :math:`W_{\mathrm{gate}}, W_{\mathrm{up}} \in \mathbb{R}^{d \times d_{ff}}`
and :math:`W_{\mathrm{down}} \in \mathbb{R}^{d_{ff} \times d}`, and no
biases, as in the paper (equation 6, where the matrices are called
:math:`W`, :math:`V` and :math:`W_2`).

A gated FFN has three weight matrices instead of two. To keep the parameter
count and compute equal to a standard FFN of width :math:`4d`, the paper
shrinks the hidden width by a factor of 2/3; LLaMA accordingly uses
:math:`d_{ff} = \tfrac{2}{3} \cdot 4d` (rounded). Released models do not
always follow this rule: Mistral 7B [3]_ uses :math:`d = 4096` and
:math:`d_{ff} = 14336 = 3.5\, d`.

References
----------
.. [1] N. Shazeer, "GLU Variants Improve Transformer", 2020, Section 2,
   equations 5–6. https://arxiv.org/abs/2002.05202
.. [2] H. Touvron et al., "LLaMA: Open and Efficient Foundation Language
   Models", 2023, Section 2.2. https://arxiv.org/abs/2302.13971
.. [3] Mistral AI, Mistral-7B-v0.1 configuration file.
   https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json
.. [4] A. Q. Jiang et al., "Mixtral of Experts", 2024, Section 2.1.
   https://arxiv.org/abs/2401.04088

Examples
--------
>>> ffn = SwiGLUFeedForward(d_model=16, d_ff=40, rng=np.random.default_rng(0))
>>> ffn(np.ones((5, 16))).shape
(5, 16)
>>> ffn.num_parameters()
1920
"""

from __future__ import annotations

import numpy as np

__all__ = ["SwiGLUFeedForward", "silu"]


def silu(x: np.ndarray) -> np.ndarray:
    """Compute the SiLU (Swish) activation ``x · sigmoid(x)`` element-wise.

    Parameters
    ----------
    x : numpy.ndarray
        Input of any shape.

    Returns
    -------
    numpy.ndarray
        Same shape as `x`. Close to ``x`` for large positive inputs and to 0
        for large negative inputs, with a small negative dip in between.

    Notes
    -----
    The sigmoid is computed as ``(1 + tanh(x / 2)) / 2``, which equals
    ``1 / (1 + exp(-x))`` but cannot overflow for very negative `x`.
    """
    return x * 0.5 * (1.0 + np.tanh(0.5 * x))


class SwiGLUFeedForward:
    """Position-wise SwiGLU feed-forward network without biases.

    Parameters
    ----------
    d_model : int
        Input and output width ``d``.
    d_ff : int
        Hidden width.
    rng : numpy.random.Generator
        Source of randomness for weight initialization.

    Attributes
    ----------
    W_gate : numpy.ndarray of shape (d_model, d_ff)
        Projection passed through SiLU; acts as the gate.
    W_up : numpy.ndarray of shape (d_model, d_ff)
        Projection that is gated.
    W_down : numpy.ndarray of shape (d_ff, d_model)
        Projection back to the model width.
    """

    def __init__(self, d_model: int, d_ff: int, rng: np.random.Generator):
        self.W_gate = rng.normal(0, 1 / np.sqrt(d_model), (d_model, d_ff))
        self.W_up = rng.normal(0, 1 / np.sqrt(d_model), (d_model, d_ff))
        self.W_down = rng.normal(0, 1 / np.sqrt(d_ff), (d_ff, d_model))

    def num_parameters(self) -> int:
        """Return the number of weights, ``3 · d_model · d_ff``."""
        return self.W_gate.size + self.W_up.size + self.W_down.size

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the FFN to every position.

        Parameters
        ----------
        x : numpy.ndarray of shape (n, d_model)
            One vector per position.

        Returns
        -------
        numpy.ndarray of shape (n, d_model)
            Transformed vectors.
        """
        hidden = silu(x @ self.W_gate) * (x @ self.W_up)  # (n, d_ff)
        return hidden @ self.W_down
