# Lecture 02: Large Language Models

**Date:** 2026-10-02 · **Length:** 1:43:15 · **Video:** <https://www.youtube.com/watch?v=GaIeu3npx04> · **Slides:** [PDF](https://cme295.stanford.edu/slides/fall26-cme295-lecture2.pdf)

How the 2017 encoder-decoder Transformer became today's decoder-only LLM: which model family won, why bigger keeps working, and the architecture and decoding choices that almost every current model shares.

```
2017 Transformer                         2026 decoder-only LLM
────────────────                         ─────────────────────
encoder + decoder           ──►          decoder only
sinusoidal positions        ──►          RoPE inside attention
multi-head attention        ──►          grouped-query attention (+ sliding window)
post-norm LayerNorm         ──►          pre-norm RMSNorm
ReLU FFN                    ──►          SwiGLU FFN or mixture of experts
greedy / beam search        ──►          temperature + top-k / top-p sampling
```

## Learning objectives

After this lecture you should be able to:

1. Compare encoder-only, decoder-only and encoder-decoder models, and explain why decoder-only models dominate.
2. Explain scaling laws, the 6·N·D compute estimate and compute-optimal training.
3. Implement a sparse mixture of experts with top-k routing, explain routing collapse and compute the load-balancing loss.
4. Implement RoPE and show that it makes attention scores depend only on relative position.
5. Explain sliding-window attention and how depth extends its reach.
6. Implement grouped-query attention and compute the KV-cache size under MHA, GQA and MQA.
7. Explain pre-norm vs post-norm and LayerNorm vs RMSNorm.
8. Implement greedy search, beam search, temperature, top-k and top-p sampling, and say when each is appropriate.
9. Explain prompting, in-context learning, context rot and chain of thought.

## Topics

| # | Time | Topic | Code |
|---|---|---|---|
| 1 | [0:00](https://www.youtube.com/watch?v=GaIeu3npx04&t=0s) | Introduction | |
| 2 | [4:47](https://www.youtube.com/watch?v=GaIeu3npx04&t=287s) | BERT and encoder-only models | [l01/transformer.py](../01-transformers/l01/transformer.py) (encoder) |
| 3 | [13:47](https://www.youtube.com/watch?v=GaIeu3npx04&t=827s) | GPT and decoder-only models | [model.py](l02/model.py) |
| 4 | [16:10](https://www.youtube.com/watch?v=GaIeu3npx04&t=970s) | Encoder-decoder versus decoder-only | [model.py](l02/model.py) |
| 5 | [19:23](https://www.youtube.com/watch?v=GaIeu3npx04&t=1163s) | Scaling laws | [Exercises](EXERCISES.md) |
| 6 | [26:49](https://www.youtube.com/watch?v=GaIeu3npx04&t=1609s) | Large language model definition | |
| 7 | [29:05](https://www.youtube.com/watch?v=GaIeu3npx04&t=1745s) | Mixture of experts (MoE) | [moe.py](l02/moe.py) |
| 8 | [33:11](https://www.youtube.com/watch?v=GaIeu3npx04&t=1991s) | MoE-based LLMs | [moe.py](l02/moe.py) |
| 9 | [36:03](https://www.youtube.com/watch?v=GaIeu3npx04&t=2163s) | Load balancing loss | [moe.py](l02/moe.py) |
| 10 | [45:36](https://www.youtube.com/watch?v=GaIeu3npx04&t=2736s) | Position embeddings | [rope.py](l02/rope.py) |
| 11 | [1:01:58](https://www.youtube.com/watch?v=GaIeu3npx04&t=3718s) | Rotary position embeddings (RoPE) | [rope.py](l02/rope.py) |
| 12 | [1:09:28](https://www.youtube.com/watch?v=GaIeu3npx04&t=4168s) | Sliding window attention (SWA) | [attention.py](l02/attention.py) |
| 13 | [1:13:15](https://www.youtube.com/watch?v=GaIeu3npx04&t=4395s) | Grouped-query attention (GQA) | [attention.py](l02/attention.py) |
| 14 | [1:18:13](https://www.youtube.com/watch?v=GaIeu3npx04&t=4693s) | Pre-norm versus post-norm | [normalization.py](l02/normalization.py), [model.py](l02/model.py) |
| 15 | [1:19:41](https://www.youtube.com/watch?v=GaIeu3npx04&t=4781s) | LayerNorm versus RMSNorm | [normalization.py](l02/normalization.py) |
| 16 | [1:21:09](https://www.youtube.com/watch?v=GaIeu3npx04&t=4869s) | Next token prediction | [decoding.py](l02/decoding.py) |
| 17 | [1:35:22](https://www.youtube.com/watch?v=GaIeu3npx04&t=5722s) | Prompting | [Exercises](EXERCISES.md) |
| 18 | [1:38:38](https://www.youtube.com/watch?v=GaIeu3npx04&t=5918s) | Context rot | [Exercises](EXERCISES.md) |
| 19 | [1:41:42](https://www.youtube.com/watch?v=GaIeu3npx04&t=6102s) | Chain of thought (CoT) | [Exercises](EXERCISES.md) |

The SwiGLU feed-forward network ([feedforward.py](l02/feedforward.py)) is not a chapter of its own, but every modern block and every MoE expert uses it.

## Key formulas

**Training compute (rule of thumb):**
C ≈ 6 · N · D floating-point operations for N parameters and D training tokens

**Mixture of experts (Mixtral-style gating):**
G(x) = softmax(TopK(x W_r)), y = Σ_{i ∈ TopK} G(x)_i E_i(x)

**Load-balancing loss (Switch Transformer):**
L_aux = α · N · Σᵢ fᵢ Pᵢ, fᵢ = fraction of tokens routed to expert i, Pᵢ = mean router probability of expert i

**RoPE:**
rotate coordinate pair i by m·θᵢ at position m, θᵢ = 10000^(−2i/d), so (R_m q)ᵀ(R_n k) = qᵀ R_{n−m} k

**Sliding-window reach:**
one layer sees W positions; L layers reach about L·W positions back

**KV cache per token:**
2 · layers · KV heads · head dim · bytes per value

**RMSNorm:**
RMSNorm(x) = γ ⊙ x / √(mean(x²) + ε)

**Pre-norm block:**
h = x + Attn(Norm(x)), y = h + FFN(Norm(h))

**SwiGLU:**
FFN(x) = (SiLU(x W_gate) ⊙ x W_up) W_down

**Temperature:**
P_T(wᵢ) = exp(xᵢ / T) / Σⱼ exp(xⱼ / T)

## Real model configurations

The fields of [LMConfig](l02/model.py) correspond to keys in released `config.json` files:

| Concept | Config key | Mistral 7B | Mixtral 8x7B |
|---|---|---|---|
| Layers | `num_hidden_layers` | 32 | 32 |
| Width | `hidden_size` | 4096 | 4096 |
| FFN width | `intermediate_size` | 14336 | 14336 |
| Query heads | `num_attention_heads` | 32 | 32 |
| KV heads (GQA) | `num_key_value_heads` | 8 | 8 |
| RoPE base | `rope_theta` | 10000 | 1000000 |
| Sliding window | `sliding_window` | 4096 | none |
| Experts | `num_local_experts` | dense | 8 |
| Experts per token | `num_experts_per_tok` | dense | 2 |
| Load-balancing coefficient | `router_aux_loss_coef` | | 0.02 |

Sources: [Mistral-7B-v0.1 config](https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json), [Mixtral-8x7B-v0.1 config](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json).

## Code in this folder

This lecture reuses `softmax`, `causal_mask` and `scaled_dot_product_attention` from [l01/attention.py](../01-transformers/l01/attention.py), and `cross_entropy` from [l01/transformer.py](../01-transformers/l01/transformer.py).

| File | What it implements |
|---|---|
| [l02/normalization.py](l02/normalization.py) | RMSNorm; pre-norm vs post-norm explained in the docstring |
| [l02/feedforward.py](l02/feedforward.py) | SwiGLU feed-forward network |
| [l02/rope.py](l02/rope.py) | Rotary position embeddings and the explicit rotation matrix |
| [l02/attention.py](l02/attention.py) | Grouped-query attention (MHA, GQA, MQA) with RoPE, sliding-window masks, receptive field, KV-cache size |
| [l02/moe.py](l02/moe.py) | Top-k mixture of experts and the load-balancing loss |
| [l02/decoding.py](l02/decoding.py) | Greedy search, beam search, temperature, top-k and top-p sampling |
| [l02/model.py](l02/model.py) | Pre-norm decoder-only LM, dense or MoE, with generation |
| [demo.py](demo.py) | Runs everything above and prints intermediate results |
| [tests/](tests/) | Unit, property-based and differential tests (GQA is checked against Lecture 01's MHA) |

```bash
uv run python lectures/02-large-language-models/demo.py
uv run pytest lectures/02-large-language-models
```

## Exercises

See [EXERCISES.md](EXERCISES.md).

## Further reading

- Devlin et al., 2018. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- Radford et al., 2019. [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2)
- Kaplan et al., 2020. [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- Hoffmann et al., 2022. [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Chinchilla)
- Shazeer et al., 2017. [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
- Fedus et al., 2021. [Switch Transformers](https://arxiv.org/abs/2101.03961)
- Jiang et al., 2024. [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- Su et al., 2021. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- Jiang et al., 2023. [Mistral 7B](https://arxiv.org/abs/2310.06825) (sliding-window attention)
- Shazeer, 2019. [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) (MQA)
- Ainslie et al., 2023. [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- Xiong et al., 2020. [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
- Zhang and Sennrich, 2019. [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- Shazeer, 2020. [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- Holtzman et al., 2019. [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) (nucleus sampling)
- Wei et al., 2022. [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
