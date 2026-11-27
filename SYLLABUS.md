# Syllabus

Stanford CME295, Autumn 2026. Nine lectures, a midterm on 2026-10-23 and a final on 2026-12-09.

For every lecture this file lists the official topics, what you should be able to do afterwards, and what this repository builds in code. A lecture is "done" here when its guide, code, tests and exercises are all in place.

Legend: 🟢 in progress · ✅ done · ⚪ planned

---

## Lecture 01: Transformers 🟢

**Date:** 2026-09-25 · **Folder:** [lectures/01-transformers](lectures/01-transformers/)

**Topics:** NLP background and tasks · tokenization · embeddings · word2vec · RNN, LSTM · attention · the Transformer architecture · end-to-end example

**Objectives:**
- Explain why subword tokenization wins over word and character tokenization, and train a BPE tokenizer by hand
- Explain how word2vec learns embeddings and why static embeddings are not enough
- Explain the RNN bottleneck and how attention removes it
- Derive scaled dot-product attention, including the √d_k factor and causal masking
- Trace tensor shapes through a full encoder-decoder Transformer

**Code:**
- [x] Byte-pair encoding tokenizer (train, encode, decode)
- [x] Skip-gram word2vec with negative sampling
- [x] Vanilla RNN cell
- [x] Scaled dot-product attention, causal and padding masks
- [x] Multi-head attention
- [x] Sinusoidal positional encoding
- [x] Encoder layer, decoder layer, full encoder-decoder forward pass
- [ ] PyTorch version trained on a toy translation task

---

## Lecture 02: Large Language Models ✅

**Date:** 2026-10-02 · **Folder:** [lectures/02-large-language-models](lectures/02-large-language-models/)

**Topics:** Transformer model families · LLM definition and architecture · Mixture of Experts (MoE) · MHA, MQA, GQA · position embeddings (RoPE and variants) · context length, temperature · sampling strategies

**Objectives:**
- Compare encoder-only, decoder-only and encoder-decoder models
- Explain scaling laws and what "large" means
- Implement MoE routing with a load-balancing loss
- Implement RoPE and explain why it encodes relative position
- Explain sliding window attention, GQA, pre-norm vs post-norm, LayerNorm vs RMSNorm
- Implement greedy, temperature, top-k and top-p sampling

**Code:**
- [x] RMSNorm
- [x] SwiGLU feed-forward network
- [x] Rotary position embeddings
- [x] Grouped-query attention (MHA, GQA, MQA) with sliding-window masks and KV-cache sizing
- [x] Top-k mixture of experts with the load-balancing loss
- [x] Greedy search, beam search, temperature, top-k and top-p sampling
- [x] Pre-norm decoder-only LM, dense or MoE, with generation

---

## Lecture 03: LLM Training ⚪

**Date:** 2026-10-09

**Topics:** pretraining · supervised finetuning (SFT) · parameter-efficient finetuning (LoRA) · preference tuning (RLHF, DPO) · reasoning · on-policy distillation · distillation to small models

**Objectives:**
- Write the cross-entropy pretraining objective and explain data mixtures and midtraining
- Explain long-context training
- Implement LoRA and count its trainable parameters
- Derive the Bradley-Terry model and the DPO loss
- Explain GRPO and on-policy distillation at a high level

**Code (planned):** tiny pretraining loop · SFT with loss masking · LoRA layer · reward model with Bradley-Terry loss · DPO loss

---

## Lecture 04: Reinforcement Learning with LLMs ⚪

**Date:** 2026-10-16

**Topics:** mathematical notation · reward design · policy gradient methods · limitations · preference tuning with PPO (RLHF) · reasoning with GRPO (RLVR) · on-policy distillation

**Objectives:**
- Map LLM generation onto states, actions, policies and rewards
- Derive REINFORCE and explain baselines and advantages
- Explain the PPO clipped objective and KL penalty
- Implement the GRPO group-relative advantage
- Design verifiable rewards

**Code (planned):** REINFORCE on a toy policy · PPO clipped loss · GRPO advantage and loss · verifiable reward functions

---

## Midterm: 2026-10-23

Covers lectures 01 to 04.

---

## Lecture 05: LLM Systems ⚪

**Date:** 2026-10-30

**Topics:** distributed training · inference optimizations · KV caching · speculative decoding · efficient kernels · FlashAttention · hardware trade-offs

**Objectives:**
- Compare data, tensor, pipeline parallelism and ZeRO/FSDP
- Estimate memory for weights, activations, optimizer state and the KV cache
- Implement a KV cache and measure the speedup
- Implement speculative decoding and explain why it is lossless
- Explain the tiling idea behind FlashAttention

**Code (planned):** memory calculator · KV cache · speculative decoding · tiled (online-softmax) attention

---

## Lecture 06: AI Agents ⚪

**Date:** 2026-11-06

**Topics:** tool calling · MCP · memory and retrieval · context compression · harness optimization · coding agents · skills and plugins

**Objectives:**
- Explain the tool-calling loop and structured outputs
- Describe the Model Context Protocol
- Build a small retrieval pipeline
- Explain context compression strategies and harness design

**Code (planned):** minimal agent loop with tools · toy MCP-style tool server · embedding-based retriever · context compaction

---

## Lecture 07: LLM Evaluation ⚪

**Date:** 2026-11-13

**Topics:** LLM-as-a-judge overview · best practices and benefits · biases and pitfalls · agent evaluation · benchmarks

**Objectives:**
- Design an LLM-as-a-judge rubric and measure agreement
- Recognize position, verbosity and self-preference bias
- Evaluate agents on trajectories, not only final answers
- Read benchmark results critically

**Code (planned):** pairwise judge harness with position swapping · agreement metrics · pass@k estimator

---

## Lecture 08: Diffusion LLMs ⚪

**Date:** 2026-11-20

**Topics:** continuous diffusion · discrete diffusion · masked diffusion · training · inference

**Objectives:**
- Explain the forward and reverse processes in continuous diffusion
- Explain how masked diffusion adapts diffusion to tokens
- Compare diffusion decoding with autoregressive decoding

**Code (planned):** 1D continuous diffusion toy · masked diffusion training objective · iterative unmasking sampler

---

## Lecture 09: Trending Topics ⚪

**Date:** 2026-12-04

**Topics:** general review · multimodality · closing thoughts

**Code (planned):** to be decided once the lecture is published

---

## Final exam: 2026-12-09

Covers the whole course.
