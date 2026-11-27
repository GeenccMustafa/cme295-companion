# Lecture 01: Transformers

**Date:** 2026-09-25 · **Length:** 1:44:28 · **Video:** <https://www.youtube.com/watch?v=114i2Kz-LZA> · **Slides:** [PDF](https://cme295.stanford.edu/slides/fall26-cme295-lecture1.pdf)

From "what is a token?" to a full encoder-decoder Transformer. The lecture follows the history of NLP: each new idea fixes a specific problem of the one before it.

```
words ──► tokens ──► embeddings ──► RNN / LSTM ──► attention ──► Transformer
          (BPE)      (word2vec)     (sequential,   (look at      (attention only,
                                     forgetful)     everything)   parallel)
```

## Learning objectives

After this lecture you should be able to:

1. Explain the trade-offs between word, character and subword tokenization, and run BPE by hand on a toy corpus.
2. Explain how word2vec learns embeddings from a proxy task, and why one vector per word is not enough.
3. Explain the two weaknesses of RNNs: sequential computation and vanishing gradients over long ranges.
4. Write scaled dot-product attention from memory, and justify the √d_k factor and the causal mask.
5. Draw the encoder and decoder layers and give the shape of every tensor in the end-to-end example.
6. Explain the training tricks: residual connections, layer normalization, dropout, label smoothing.

## Topics

| # | Time | Topic | Code |
|---|---|---|---|
| 1 | [0:00](https://www.youtube.com/watch?v=114i2Kz-LZA&t=0s) | Welcome | |
| 2 | [3:43](https://www.youtube.com/watch?v=114i2Kz-LZA&t=223s) | Difference from last year's edition | |
| 3 | [5:01](https://www.youtube.com/watch?v=114i2Kz-LZA&t=301s) | Class logistics | |
| 4 | [12:42](https://www.youtube.com/watch?v=114i2Kz-LZA&t=762s) | Timeline | |
| 5 | [16:56](https://www.youtube.com/watch?v=114i2Kz-LZA&t=1016s) | Tokenization | [tokenization.py](l01/tokenization.py) |
| 6 | [37:27](https://www.youtube.com/watch?v=114i2Kz-LZA&t=2247s) | Word2vec (CBOW, skip-gram) | [word2vec.py](l01/word2vec.py) |
| 7 | [48:26](https://www.youtube.com/watch?v=114i2Kz-LZA&t=2906s) | RNN, LSTM | [rnn.py](l01/rnn.py) |
| 8 | [55:18](https://www.youtube.com/watch?v=114i2Kz-LZA&t=3318s) | Attention | [attention.py](l01/attention.py) |
| 9 | [57:16](https://www.youtube.com/watch?v=114i2Kz-LZA&t=3436s) | Self-attention mechanism | [attention.py](l01/attention.py) |
| 10 | [59:54](https://www.youtube.com/watch?v=114i2Kz-LZA&t=3594s) | Query, key, value | [attention.py](l01/attention.py) |
| 11 | [1:09:35](https://www.youtube.com/watch?v=114i2Kz-LZA&t=4175s) | Transformer architecture | [transformer.py](l01/transformer.py) |
| 12 | [1:12:19](https://www.youtube.com/watch?v=114i2Kz-LZA&t=4339s) | Input | [positional_encoding.py](l01/positional_encoding.py) |
| 13 | [1:16:27](https://www.youtube.com/watch?v=114i2Kz-LZA&t=4587s) | Transformer encoder | [transformer.py](l01/transformer.py) |
| 14 | [1:18:19](https://www.youtube.com/watch?v=114i2Kz-LZA&t=4699s) | Transformer decoder | [transformer.py](l01/transformer.py) |
| 15 | [1:22:16](https://www.youtube.com/watch?v=114i2Kz-LZA&t=4936s) | Computational tricks | [transformer.py](l01/transformer.py) |
| 16 | [1:27:35](https://www.youtube.com/watch?v=114i2Kz-LZA&t=5255s) | End-to-end example | [demo.py](demo.py) |

## Key formulas

**Language modeling (chain rule):**
P(w₁, …, wₙ) = ∏ₜ P(wₜ | w₁, …, wₜ₋₁)

**Skip-gram with negative sampling:**
L = −log σ(u_o · v_c) − Σₖ log σ(−u_k · v_c)

**RNN:**
hₜ = tanh(W_xh xₜ + W_hh hₜ₋₁ + b)

**Scaled dot-product attention:**
Attention(Q, K, V) = softmax(QKᵀ / √d_k) V

**Multi-head attention:**
MultiHead(X) = Concat(head₁, …, head_h) W_O, with headᵢ = Attention(XW_Qⁱ, XW_Kⁱ, XW_Vⁱ)

**Sinusoidal positional encoding:**
PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

**Layer normalization:**
LN(x) = γ ⊙ (x − μ) / √(σ² + ε) + β

**Label smoothing:**
q′(k) = (1 − ε) · 1[k = y] + ε / V

## Shapes in the end-to-end example

"Transformer base": d_model = 512, h = 8 heads, d_k = d_v = 64, d_ff = 2048, N = 6 layers.

| Step | Shape |
|---|---|
| Source sentence, 8 tokens including [BOS] and [EOS] | (8,) |
| Token embeddings + positional encodings | (8, 512) |
| Q, K, V for one head | (8, 64) each |
| Attention scores QKᵀ for one head | (8, 8) |
| Head output | (8, 64) |
| 8 heads concatenated, then W_O | (8, 512) |
| FFN hidden layer | (8, 2048) |
| Encoder output (after 6 layers) | (8, 512) |
| Decoder cross-attention scores at step t | (t, 8) |
| Output logits at step t | (t, V) |

## Code in this folder

| File | What it implements |
|---|---|
| [l01/tokenization.py](l01/tokenization.py) | BPE: train merges, tokenize, encode, decode |
| [l01/word2vec.py](l01/word2vec.py) | Skip-gram with negative sampling, trained with hand-written gradients |
| [l01/rnn.py](l01/rnn.py) | Vanilla RNN and the norm of ∂h_T/∂h_t, to see gradients vanish |
| [l01/attention.py](l01/attention.py) | Softmax, scaled dot-product attention, causal and padding masks, multi-head attention |
| [l01/positional_encoding.py](l01/positional_encoding.py) | Sinusoidal positional encoding |
| [l01/transformer.py](l01/transformer.py) | LayerNorm, FFN, encoder and decoder layers, full model, greedy decoding, cross-entropy with label smoothing |
| [demo.py](demo.py) | Runs everything above and prints intermediate results |
| [tests/](tests/) | One test file per module |

```bash
uv run python lectures/01-transformers/demo.py
uv run pytest lectures/01-transformers
```

## Exercises

See [EXERCISES.md](EXERCISES.md).

## Further reading

- Vaswani et al., 2017. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Sennrich et al., 2016. [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) (BPE)
- Mikolov et al., 2013. [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) (word2vec, negative sampling)
- Hochreiter and Schmidhuber, 1997. Long Short-Term Memory
- Bahdanau et al., 2014. [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) (attention)
- Ba et al., 2016. [Layer Normalization](https://arxiv.org/abs/1607.06450)
- Szegedy et al., 2015. [Rethinking the Inception Architecture](https://arxiv.org/abs/1512.00567) (label smoothing)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
