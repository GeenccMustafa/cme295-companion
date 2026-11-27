"""Lecture 02: Large Language Models.

From-scratch NumPy implementations of the architecture and decoding choices
that turn the 2017 Transformer into a modern LLM. Builds on Lecture 01
(:mod:`l01.attention` and :mod:`l01.transformer`).

``normalization``
    RMSNorm, and why pre-norm replaced post-norm.
``feedforward``
    The SwiGLU feed-forward network.
``rope``
    Rotary position embeddings.
``attention``
    Grouped-query attention (MHA, GQA, MQA), sliding windows and KV-cache size.
``moe``
    Sparse mixture of experts with top-k routing and a load-balancing loss.
``decoding``
    Greedy and beam search, temperature, top-k and top-p sampling.
``model``
    A decoder-only language model assembled from the pieces above.
"""
