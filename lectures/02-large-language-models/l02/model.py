r"""A decoder-only language model with the architecture of modern LLMs.

The original Transformer pairs an encoder with a decoder. GPT-style models
keep only the decoder: a stack of causal self-attention blocks trained to
predict the next token. The lecture summarizes today's default refinements
of that block, all of which appear in this module:

- pre-norm residual blocks with RMSNorm (:mod:`l02.normalization`)
- rotary position embeddings instead of absolute ones (:mod:`l02.rope`)
- grouped-query attention, optionally with a sliding window
  (:mod:`l02.attention`)
- a SwiGLU feed-forward network (:mod:`l02.feedforward`), or a sparse
  mixture of SwiGLU experts (:mod:`l02.moe`)

LLaMA [1]_ is the reference description of this recipe; Mistral 7B adds the
sliding window and grouped-query attention [2]_, and Mixtral replaces the
feed-forward network with experts [3]_. As in Lecture 01 the weights are
random, so the model is useful for following shapes and data flow, not for
producing sensible text.

Notes
-----
Each block computes, with :math:`\mathrm{N}` denoting RMSNorm,

.. math::

    h &= x + \mathrm{Attn}(\mathrm{N}(x)) \\
    y &= h + \mathrm{FFN}(\mathrm{N}(h)),

and the model is

.. math::

    \text{logits} = \mathrm{N}(\text{block}_L \circ \dots \circ
                    \text{block}_1(E[\text{ids}]))\, E^\top,

where :math:`E \in \mathbb{R}^{V \times d}` is the token embedding matrix,
shared with the output projection when ``tie_embeddings`` is set. There is
no positional encoding on the input: positions enter only through RoPE
inside attention. Training minimizes the cross-entropy between the logits
at position :math:`t` and the token at position :math:`t + 1`.

Generation here recomputes the whole prefix for every new token. Real
systems cache the keys and values of past tokens instead (Lecture 05).

References
----------
.. [1] H. Touvron et al., "LLaMA: Open and Efficient Foundation Language
   Models", 2023, Section 2.2. https://arxiv.org/abs/2302.13971
.. [2] A. Q. Jiang et al., "Mistral 7B", 2023. https://arxiv.org/abs/2310.06825
.. [3] A. Q. Jiang et al., "Mixtral of Experts", 2024.
   https://arxiv.org/abs/2401.04088

Examples
--------
>>> cfg = LMConfig(vocab_size=100, d_model=32, num_layers=2, num_heads=4, num_kv_heads=2, d_ff=64)
>>> model = DecoderOnlyLM(cfg)
>>> logits, aux_loss = model(np.array([1, 17, 42, 8]))
>>> logits.shape, aux_loss
((4, 100), 0.0)
>>> len(model.generate([1, 17], max_new_tokens=3, rng=np.random.default_rng(0)))
5
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from l01.transformer import cross_entropy

from .attention import GroupedQueryAttention
from .decoding import sample_next_token
from .feedforward import SwiGLUFeedForward
from .moe import MixtureOfExperts, Routing, load_balancing_loss
from .normalization import RMSNorm

__all__ = ["DecoderBlock", "DecoderOnlyLM", "LMConfig"]


@dataclass(frozen=True)
class LMConfig:
    """Hyperparameters of a decoder-only language model.

    Field names follow the meaning of the corresponding keys in released
    model configuration files (``hidden_size``, ``num_key_value_heads``,
    ``rope_theta``, ``sliding_window``, ``num_local_experts``,
    ``num_experts_per_tok``, ``rms_norm_eps``, ``tie_word_embeddings``).

    Attributes
    ----------
    vocab_size : int
        Number of tokens ``V``.
    d_model : int
        Width of the residual stream ``d``.
    num_layers : int
        Number of decoder blocks ``L``.
    num_heads : int
        Query heads per attention layer.
    num_kv_heads : int
        Key/value heads per attention layer.
    d_ff : int
        Hidden width of each feed-forward network or expert.
    rope_base : float
        Base of the RoPE frequencies.
    sliding_window : int or None
        Attention window size; None for full causal attention.
    num_experts : int
        Experts per layer; 0 means a dense SwiGLU feed-forward network.
    top_k : int
        Experts used per token when `num_experts` is positive.
    norm_eps : float
        Epsilon of every RMSNorm.
    tie_embeddings : bool
        Whether the output projection reuses the embedding matrix.
    """

    vocab_size: int = 256
    d_model: int = 64
    num_layers: int = 2
    num_heads: int = 4
    num_kv_heads: int = 2
    d_ff: int = 128
    rope_base: float = 10000.0
    sliding_window: int | None = None
    num_experts: int = 0
    top_k: int = 2
    norm_eps: float = 1e-6
    tie_embeddings: bool = True


class DecoderBlock:
    """One pre-norm decoder block: attention, then a feed-forward network or MoE.

    Parameters
    ----------
    cfg : LMConfig
        Model hyperparameters.
    rng : numpy.random.Generator
        Source of randomness for weight initialization.

    Attributes
    ----------
    attn_norm, ffn_norm : RMSNorm
        Normalizations applied to the input of each sub-layer.
    attn : GroupedQueryAttention
        Causal self-attention.
    ffn : SwiGLUFeedForward or MixtureOfExperts
        Position-wise network.
    """

    def __init__(self, cfg: LMConfig, rng: np.random.Generator):
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GroupedQueryAttention(
            cfg.d_model,
            cfg.num_heads,
            cfg.num_kv_heads,
            rng,
            rope_base=cfg.rope_base,
            window=cfg.sliding_window,
        )
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn: SwiGLUFeedForward | MixtureOfExperts
        if cfg.num_experts:
            self.ffn = MixtureOfExperts(cfg.d_model, cfg.d_ff, cfg.num_experts, cfg.top_k, rng)
        else:
            self.ffn = SwiGLUFeedForward(cfg.d_model, cfg.d_ff, rng)

    def num_parameters(self) -> int:
        """Return all weights of the block."""
        norms = self.attn_norm.num_parameters() + self.ffn_norm.num_parameters()
        return norms + self.attn.num_parameters() + self.ffn.num_parameters()

    def num_active_parameters(self) -> int:
        """Return the weights used for one token (differs from the total only for MoE)."""
        if isinstance(self.ffn, MixtureOfExperts):
            ffn = self.ffn.num_active_parameters()
        else:
            ffn = self.ffn.num_parameters()
        norms = self.attn_norm.num_parameters() + self.ffn_norm.num_parameters()
        return norms + self.attn.num_parameters() + ffn

    def __call__(self, x: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, Routing | None]:
        """Apply the block to the residual stream.

        Parameters
        ----------
        x : numpy.ndarray of shape (n, d_model)
            Residual stream.
        positions : numpy.ndarray of int, shape (n,)
            Token positions, used by RoPE.

        Returns
        -------
        output : numpy.ndarray of shape (n, d_model)
            Updated residual stream.
        routing : Routing or None
            Expert routing when the block uses a mixture of experts.
        """
        attn_out, _ = self.attn(self.attn_norm(x), positions)
        x = x + attn_out
        routing = None
        if isinstance(self.ffn, MixtureOfExperts):
            ffn_out, routing = self.ffn(self.ffn_norm(x))
        else:
            ffn_out = self.ffn(self.ffn_norm(x))
        return x + ffn_out, routing


class DecoderOnlyLM:
    """Decoder-only language model built from :class:`DecoderBlock`.

    Parameters
    ----------
    cfg : LMConfig
        Model hyperparameters.
    seed : int, default 0
        Seed for weight initialization.

    Attributes
    ----------
    embedding : numpy.ndarray of shape (vocab_size, d_model)
        Token embedding matrix.
    blocks : list of DecoderBlock
        The ``L`` decoder blocks.
    final_norm : RMSNorm
        Normalization before the output projection.
    lm_head : numpy.ndarray of shape (d_model, vocab_size)
        Output projection; the transpose of `embedding` when tied.
    """

    def __init__(self, cfg: LMConfig, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.cfg = cfg
        self.embedding = rng.normal(0, 1 / np.sqrt(cfg.d_model), (cfg.vocab_size, cfg.d_model))
        self.blocks = [DecoderBlock(cfg, rng) for _ in range(cfg.num_layers)]
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        if cfg.tie_embeddings:
            self.lm_head = self.embedding.T
        else:
            self.lm_head = rng.normal(0, 1 / np.sqrt(cfg.d_model), (cfg.d_model, cfg.vocab_size))

    def _shared_parameters(self) -> int:
        head = 0 if self.cfg.tie_embeddings else self.lm_head.size
        return self.embedding.size + self.final_norm.num_parameters() + head

    def num_parameters(self) -> int:
        """Return the total number of weights."""
        return self._shared_parameters() + sum(b.num_parameters() for b in self.blocks)

    def num_active_parameters(self) -> int:
        """Return the weights used for one token; smaller than the total for MoE models."""
        return self._shared_parameters() + sum(b.num_active_parameters() for b in self.blocks)

    def __call__(self, ids: np.ndarray) -> tuple[np.ndarray, float]:
        """Compute next-token logits for every position.

        Parameters
        ----------
        ids : numpy.ndarray of int, shape (n,)
            Token ids.

        Returns
        -------
        logits : numpy.ndarray of shape (n, vocab_size)
            Row ``t`` scores the token at position ``t + 1``.
        aux_loss : float
            Mean load-balancing loss over the MoE layers, 0 for dense models.
        """
        positions = np.arange(len(ids))
        x = self.embedding[ids]
        aux_losses = []
        for block in self.blocks:
            x, routing = block(x, positions)
            if routing is not None:
                aux_losses.append(
                    load_balancing_loss(routing.probs, routing.experts, self.cfg.num_experts)
                )
        logits = self.final_norm(x) @ self.lm_head
        return logits, float(np.mean(aux_losses)) if aux_losses else 0.0

    def next_token_loss(self, ids: np.ndarray) -> float:
        """Return the language-modelling loss of a sequence.

        Parameters
        ----------
        ids : numpy.ndarray of int, shape (n,)
            Token ids, ``n >= 2``.

        Returns
        -------
        float
            Mean cross-entropy between the logits at positions ``0..n-2``
            and the tokens at positions ``1..n-1``.
        """
        logits, _ = self(ids)
        return cross_entropy(logits[:-1], ids[1:])

    def generate(
        self,
        prompt: list[int],
        max_new_tokens: int,
        rng: np.random.Generator,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_id: int | None = None,
    ) -> list[int]:
        """Extend a prompt one sampled token at a time.

        Parameters
        ----------
        prompt : list of int
            Token ids to continue.
        max_new_tokens : int
            Maximum number of tokens to add.
        rng : numpy.random.Generator
            Source of randomness for sampling.
        temperature : float, default 1.0
            Sampling temperature; 0 for greedy decoding.
        top_k : int, optional
            Restrict sampling to the `top_k` most likely tokens.
        top_p : float, optional
            Restrict sampling to the nucleus of mass `top_p`.
        eos_id : int, optional
            Stop as soon as this token is generated.

        Returns
        -------
        list of int
            The prompt followed by the generated tokens.
        """
        tokens = list(prompt)
        for _ in range(max_new_tokens):
            logits, _ = self(np.array(tokens))
            next_id = sample_next_token(logits[-1], rng, temperature, top_k, top_p)
            tokens.append(next_id)
            if next_id == eos_id:
                break
        return tokens
