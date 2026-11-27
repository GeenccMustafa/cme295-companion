r"""Sparse mixture-of-experts (MoE) feed-forward layer.

Making a dense model bigger makes every token more expensive. A mixture of
experts [1]_ decouples the two: the feed-forward network of a layer is
replaced by ``N`` independent *experts* (each a SwiGLU FFN), and a small
*router* sends every token to only ``k`` of them. The model then has the
parameters of ``N`` FFNs but the per-token compute of ``k``. Mixtral 8x7B,
for example, has 8 experts per layer and uses 2 per token: about 47B
parameters in total but 13B active per token [3]_.

Routing happens independently for every token in every layer. Left alone,
training tends to collapse onto a few experts that then get all the
gradient signal ("routing collapse"), so an auxiliary *load-balancing loss*
[2]_ is added to the language-modelling loss to keep the experts in use.

Notes
-----
For a token vector :math:`x`, the router computes logits
:math:`\ell = x W_r \in \mathbb{R}^N`. Following Mixtral [3]_, the gate
weights are a softmax over the top-:math:`k` logits only,

.. math::

    G(x) = \operatorname{softmax}(\operatorname{TopK}(\ell)), \qquad
    y = \sum_{i \in \operatorname{TopK}} G(x)_i\, E_i(x),

where :math:`\operatorname{TopK}` sets the other logits to :math:`-\infty`.
(Other models take the softmax over all logits first and then keep the top
:math:`k`, optionally renormalizing; with :math:`k = N` both reduce to the
dense mixture :math:`\sum_i \operatorname{softmax}(\ell)_i E_i(x)`.)

The load-balancing loss of the Switch Transformer [2]_ (equations 4–6),
over a batch of :math:`T` tokens, is

.. math::

    \mathcal{L}_{\text{aux}} = \alpha \cdot N \sum_{i=1}^{N} f_i P_i,

where :math:`P_i` is the mean router probability of expert :math:`i`
(softmax over all :math:`N` logits) and :math:`f_i` is the fraction of
tokens dispatched to expert :math:`i`. The Switch Transformer routes each
token to one expert; for :math:`k > 1` this module counts all :math:`T k`
assignments, so that :math:`\sum_i f_i = 1` as before. Uniform routing gives
:math:`N \sum_i f_i P_i = 1` and complete collapse onto one expert gives
:math:`N`. :math:`f_i` is not differentiable, so gradients flow through
:math:`P_i`. The coefficient :math:`\alpha` is left to the caller; the
Switch Transformer uses :math:`10^{-2}`.

References
----------
.. [1] N. Shazeer et al., "Outrageously Large Neural Networks: The
   Sparsely-Gated Mixture-of-Experts Layer", ICLR 2017.
   https://arxiv.org/abs/1701.06538
.. [2] W. Fedus, B. Zoph and N. Shazeer, "Switch Transformers: Scaling to
   Trillion Parameter Models with Simple and Efficient Sparsity", JMLR
   2022, Section 2.2. https://arxiv.org/abs/2101.03961
.. [3] A. Q. Jiang et al., "Mixtral of Experts", 2024, Section 2.1.
   https://arxiv.org/abs/2401.04088

Examples
--------
>>> moe = MixtureOfExperts(
...     d_model=16, d_ff=32, num_experts=8, top_k=2, rng=np.random.default_rng(0)
... )
>>> out, routing = moe(np.random.default_rng(1).normal(size=(5, 16)))
>>> out.shape, routing.experts.shape
((5, 16), (5, 2))
>>> moe.num_active_parameters() / moe.num_parameters() < 0.3
True
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from l01.attention import softmax

from .feedforward import SwiGLUFeedForward

__all__ = ["MixtureOfExperts", "Routing", "load_balancing_loss"]


@dataclass(frozen=True)
class Routing:
    """Routing decisions of a mixture-of-experts layer for a batch of tokens.

    Attributes
    ----------
    probs : numpy.ndarray of shape (n, num_experts)
        Softmax of the router logits over all experts. Used by the
        load-balancing loss.
    experts : numpy.ndarray of int, shape (n, top_k)
        Indices of the experts each token is sent to, best first.
    gates : numpy.ndarray of shape (n, top_k)
        Weight of each selected expert; every row sums to 1.
    """

    probs: np.ndarray
    experts: np.ndarray
    gates: np.ndarray


def load_balancing_loss(probs: np.ndarray, experts: np.ndarray, num_experts: int) -> float:
    """Compute the Switch Transformer load-balancing loss without its coefficient.

    Parameters
    ----------
    probs : numpy.ndarray of shape (T, num_experts)
        Router probabilities over all experts for each of ``T`` tokens.
    experts : numpy.ndarray of int, shape (T, k)
        Experts each token was dispatched to.
    num_experts : int
        Number of experts ``N``.

    Returns
    -------
    float
        ``N · Σ_i f_i · P_i``: 1 for perfectly balanced routing, ``N`` when
        every token goes to the same expert with probability 1. Multiply by
        a small coefficient (``1e-2`` in the Switch Transformer) and add it
        to the training loss.
    """
    f = np.bincount(experts.ravel(), minlength=num_experts) / experts.size  # (N,)
    P = probs.mean(axis=0)  # (N,)
    return float(num_experts * np.sum(f * P))


class MixtureOfExperts:
    """Top-k routed mixture of SwiGLU experts.

    Parameters
    ----------
    d_model : int
        Input and output width.
    d_ff : int
        Hidden width of each expert.
    num_experts : int
        Number of experts ``N``.
    top_k : int
        Experts used per token ``k``.
    rng : numpy.random.Generator
        Source of randomness for weight initialization.

    Attributes
    ----------
    W_router : numpy.ndarray of shape (d_model, num_experts)
        Router projection.
    experts : list of SwiGLUFeedForward
        The ``N`` expert networks.

    Raises
    ------
    ValueError
        If `top_k` is not between 1 and `num_experts`.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int,
        rng: np.random.Generator,
    ):
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")
        self.num_experts = num_experts
        self.top_k = top_k
        self.W_router = rng.normal(0, 1 / np.sqrt(d_model), (d_model, num_experts))
        self.experts = [SwiGLUFeedForward(d_model, d_ff, rng) for _ in range(num_experts)]

    def num_parameters(self) -> int:
        """Return the total number of weights: the router plus all experts."""
        return self.W_router.size + sum(e.num_parameters() for e in self.experts)

    def num_active_parameters(self) -> int:
        """Return the weights used for one token: the router plus ``k`` experts."""
        return self.W_router.size + self.top_k * self.experts[0].num_parameters()

    def route(self, x: np.ndarray) -> Routing:
        """Choose experts and gate weights for every token.

        Parameters
        ----------
        x : numpy.ndarray of shape (n, d_model)
            Token vectors.

        Returns
        -------
        Routing
            Probabilities over all experts, selected experts and gates.
        """
        logits = x @ self.W_router  # (n, N)
        experts = np.argsort(-logits, axis=-1, kind="stable")[:, : self.top_k]  # (n, k)
        gates = softmax(np.take_along_axis(logits, experts, axis=-1))  # (n, k)
        return Routing(probs=softmax(logits), experts=experts, gates=gates)

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, Routing]:
        """Route every token to its experts and combine their outputs.

        Each expert only processes the tokens routed to it, which is where
        the compute saving comes from.

        Parameters
        ----------
        x : numpy.ndarray of shape (n, d_model)
            Token vectors.

        Returns
        -------
        output : numpy.ndarray of shape (n, d_model)
            Gate-weighted sum of the selected experts' outputs.
        routing : Routing
            The routing decisions, needed for the load-balancing loss.
        """
        routing = self.route(x)
        out = np.zeros_like(x)
        for e in range(self.num_experts):
            tokens, slot = np.nonzero(routing.experts == e)  # tokens routed to expert e
            if tokens.size == 0:
                continue
            gate = routing.gates[tokens, slot][:, None]  # (m, 1)
            out[tokens] += gate * self.experts[e](x[tokens])
        return out, routing
