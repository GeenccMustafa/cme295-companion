r"""Root mean square layer normalization (RMSNorm).

Normalization keeps the scale of activations under control as they flow
through dozens of layers. The original Transformer used LayerNorm, which
re-centres each vector to zero mean and rescales it to unit variance. Zhang
and Sennrich [1]_ observed that the re-centring step contributes little and
proposed RMSNorm, which only rescales. It needs one statistic instead of two
and one learned vector instead of two, and it has become the default in
modern LLMs (LLaMA [2]_, Mistral, Qwen, DeepSeek).

Notes
-----
For a vector :math:`x \in \mathbb{R}^d`,

.. math::

    \mathrm{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon},
    \qquad
    \mathrm{RMSNorm}(x) = \gamma \odot \frac{x}{\mathrm{RMS}(x)},

with a learned gain :math:`\gamma \in \mathbb{R}^d`. Compared with
LayerNorm, :math:`\mathrm{LN}(x) = \gamma \odot (x - \mu)/\sigma + \beta`,
there is no mean :math:`\mu` and no bias :math:`\beta`. The output therefore
has unit root mean square but is not centred, and it is invariant to
rescaling the input by any positive constant.

Modern LLMs also place the normalization differently from the 2017
Transformer. *Post-norm* computes :math:`\mathrm{Norm}(x + F(x))`, which puts
the normalization on the residual path. *Pre-norm* computes
:math:`x + F(\mathrm{Norm}(x))`, which leaves an identity path from the
input to the output of every block; Xiong et al. [3]_ show that this keeps
gradients well-behaved at initialization and removes the need for
learning-rate warm-up. :class:`l02.model.DecoderBlock` uses pre-norm.

References
----------
.. [1] B. Zhang and R. Sennrich, "Root Mean Square Layer Normalization",
   NeurIPS 2019. https://arxiv.org/abs/1910.07467
.. [2] H. Touvron et al., "LLaMA: Open and Efficient Foundation Language
   Models", 2023, Section 2.2. https://arxiv.org/abs/2302.13971
.. [3] R. Xiong et al., "On Layer Normalization in the Transformer
   Architecture", ICML 2020. https://arxiv.org/abs/2002.04745

Examples
--------
>>> norm = RMSNorm(4, eps=0.0)
>>> norm(np.array([1.0, 2.0, 3.0, 6.0])).round(3).tolist()
[0.283, 0.566, 0.849, 1.697]
"""

from __future__ import annotations

import numpy as np

__all__ = ["RMSNorm"]


class RMSNorm:
    """Root mean square normalization over the last axis.

    Parameters
    ----------
    d_model : int
        Feature dimension ``d``.
    eps : float, default 1e-6
        Added to the mean square for numerical stability. Released models use
        values between ``1e-6`` and ``1e-5`` (``rms_norm_eps`` in their
        configuration files).

    Attributes
    ----------
    gamma : numpy.ndarray of shape (d_model,)
        Learned gain, initialized to ones.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        self.gamma = np.ones(d_model)
        self.eps = eps

    def num_parameters(self) -> int:
        """Return the number of learned values, ``d_model`` (LayerNorm has ``2 · d_model``)."""
        return self.gamma.size

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Normalize every vector along the last axis.

        Parameters
        ----------
        x : numpy.ndarray of shape (..., d_model)
            Input activations; each vector is normalized independently.

        Returns
        -------
        numpy.ndarray of shape (..., d_model)
            ``gamma * x / RMS(x)``.
        """
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return self.gamma * x / rms
