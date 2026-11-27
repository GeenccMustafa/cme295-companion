# Lecture 02 exercises

Pen-and-paper questions first, then coding tasks that extend the code in [l02/](l02/). Many concept answers can be checked with the code.

## Concepts

1. **Model families.** In "I sat by the bank of the river", the word "river" disambiguates "bank". Which model family can use it when building the representation of "bank": encoder-only, decoder-only, or both? Why did decoder-only models win anyway?

2. **Training compute.** Estimate the training compute of a 7B-parameter model trained on 2T tokens with C ≈ 6·N·D. Compute-optimal training (Chinchilla) suggests roughly 20 tokens per parameter. How many tokens would that be for 7B parameters, and by what factor is the 2T-token model "over-trained"? Why might a lab do this on purpose?

3. **MoE parameters.** Mixtral 8x7B has 8 experts and routes each token to 2. Explain why it has about 47B parameters rather than 56B, and why only about 13B are active per token. Which parts of the model are not replicated per expert?

4. **Routing collapse.** For N = 4 experts, compute N · Σ fᵢ Pᵢ when (a) routing is perfectly uniform and (b) every token goes to expert 0 with probability 1. Check your answers with `load_balancing_loss`. Why does the loss use Pᵢ at all, given that fᵢ already measures the imbalance?

5. **Why experts replace the FFN.** Why is the mixture of experts placed in the feed-forward network and not in attention?

6. **RoPE and shifts.** A query at position 10 and a key at position 7 give score s. What is the score for the same vectors at positions 110 and 107? What changes if only the key moves? Verify with `apply_rope`.

7. **Sliding windows.** Each token attends to itself and the two previous tokens, and there are two such layers. What is the earliest input position that can influence position 6? How many layers would position 1 need? Check with `receptive_field`.

8. **KV cache.** A model has 40 layers, 8 KV heads of size 128 and stores the cache in bfloat16. How much memory does a 64K-token context need? How much would it need with 32 KV heads (MHA), and with 1 (MQA)?

9. **Normalization.** Compute LayerNorm and RMSNorm of x = [1, 2, 3, 6] by hand (ε = 0, γ = 1, β = 0). Why does pre-norm make deep stacks easier to train than post-norm?

10. **Temperature and filtering.** For probabilities soft 0.60, brown 0.25, small 0.10, worn 0.05: which tokens survive top-p with p = 0.8? What happens to greedy decoding and to sampling when the temperature goes from 1 to 2?

11. **Greedy vs beam.** With P(a) = 0.6, P(b) = 0.4, P(u|a) = P(v|a) = 0.5, P(u|b) = 0.9 and P(v|b) = 0.1, find the greedy sequence, the most probable two-token sequence, and the result of beam search with 2 beams.

12. **Determinism.** Why can an LLM API give different answers at temperature 0?

13. **Prompting and context.** What is the difference between few-shot prompting and fine-tuning? Why can a model with a 1M-token window still answer worse when the relevant fact sits in the middle of a long context?

14. **Chain of thought.** Give two reasons why asking a model to write intermediate reasoning steps can improve its answer, in terms of how a decoder-only model computes. What does self-consistency add, and when can it fail?

## Coding

1. **Shared experts.** Add `num_shared_experts` to `MixtureOfExperts`: shared experts process every token and are added to the routed output (DeepSeekMoE). Update the parameter counts and tests.

2. **Alternative gating.** Add a `gating="softmax_then_topk"` option that takes the softmax over all experts first and renormalizes the selected ones. Show that for k = 1 the two gatings give different gate values but the same chosen expert.

3. **Interleaved attention.** Let `LMConfig` describe a per-layer pattern of local (sliding-window) and global layers, as in models that alternate the two. Measure the receptive field of the stack.

4. **The `rotate_half` convention.** Implement RoPE with the half-split pairing used by many released checkpoints. Write a differential test showing that it equals `apply_rope` applied to a permuted vector.

5. **RoPE long-term decay.** Plot the RoPE attention score between a fixed query and key vector of ones as a function of their distance, for bases 10⁴ and 10⁶. Relate the plot to raising `rope_theta` for long contexts.

6. **ALiBi.** Implement ALiBi (a fixed per-head linear penalty on the distance) as an alternative to RoPE in `GroupedQueryAttention`, and test that it is shift-invariant as well.

7. **Vectorized GQA.** Replace the loop over heads with batched matrix products, using `np.repeat` on the key/value heads. Keep the differential test against Lecture 01 passing.

8. **Beam search length penalty.** Add the length penalty of Wu et al. (2016), as used with α = 0.6 by the original Transformer, and construct a toy distribution where it changes the result.

9. **Min-p sampling.** Implement min-p filtering (keep tokens whose probability is at least a fraction of the top token's) and compare the number of kept tokens with top-p on peaked and flat distributions.

10. **Constrained decoding.** Write a sampler that only allows tokens accepted by a small finite-state machine (for example digits followed by a closing bracket) by setting the logits of all other tokens to −∞.

11. **Scaling-law calculator.** Write a function that, given a compute budget C, returns the compute-optimal parameter count and token count under C = 6·N·D and D ≈ 20·N, and tabulate it for budgets from 10²⁰ to 10²⁵ FLOPs.
