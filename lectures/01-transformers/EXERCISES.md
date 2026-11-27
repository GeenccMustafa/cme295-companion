# Lecture 01 exercises

Pen-and-paper questions first, then coding tasks that extend the code in [l01/](l01/).

## Concepts

1. **Tokenization trade-offs.** For word, character and subword tokenization, what happens to (a) the vocabulary size, (b) the sequence length, and (c) unknown words? Why does attention cost make sequence length matter?

2. **BPE by hand.** Corpus: `low ×5, lower ×2, newest ×6, widest ×3`, each word ending in `</w>`. List the counts of all adjacent pairs and perform the first three merges. Check your answer against `BPETokenizer.train(corpus, 3).merges`.

3. **Embedding size.** A vocabulary of 128,000 tokens with d_model = 4,096. How many parameters does the embedding matrix have? What changes if the output projection shares these weights?

4. **The √d_k factor.** If q and k have independent entries with mean 0 and variance 1, show that Var(q · k) = d_k. What is the standard deviation of the raw score when d_k = 64, and what goes wrong in the softmax without scaling?

5. **Masked softmax.** Scores for a 3-token sequence are `[[1, 2, 3], [1, 2, 3], [1, 2, 3]]`. Apply a causal mask and compute the softmax of each row.

6. **Permutation.** Without positional encodings, show that self-attention is permutation-equivariant. Does the same hold for a decoder with a causal mask?

7. **Multi-head parameters.** How many weights are in one multi-head attention block with d_model = 512 and h = 8 (no biases)? Does the number depend on h?

8. **Cross-attention shapes.** The encoder has seen 5 source tokens and the decoder has produced 3 tokens. Where do Q, K and V come from in cross-attention, and what is the shape of the attention matrix?

9. **Label smoothing.** With V = 5 classes and ε = 0.1, what target distribution does the true class get? Why does label smoothing hurt perplexity but can improve BLEU?

10. **Sinusoidal vs learned positions.** Give one advantage of sinusoidal encodings over learned position embeddings when the test sequence is longer than any training sequence.

## Coding

1. **BPE on real text.** Train `BPETokenizer` on a few paragraphs of English and of Turkish. Plot tokens per word against the number of merges for each language. Which language needs more tokens per word, and why?

2. **Byte-level BPE.** Change `tokenization.py` to start from UTF-8 bytes instead of characters, so that no input can ever raise `KeyError`.

3. **CBOW.** Add a `CBOW` class to `word2vec.py` that predicts the center word from the average of its context vectors. Compare it with skip-gram on the toy corpus from the tests.

4. **LSTM.** Add an LSTM cell to `rnn.py` and compare its `gradient_norms` curve with the vanilla RNN.

5. **Vectorized multi-head attention.** Replace the Python loop over heads in `MultiHeadAttention.__call__` with a single batched matrix multiply. Keep the tests passing.

6. **Pre-norm.** Add a `pre_norm: bool` flag to `TransformerConfig` that switches the layers to `x + Sublayer(LN(x))`. Which variant do modern LLMs use?

7. **Attention heatmap.** Plot the attention weights of the first encoder layer for a short input, one subplot per head.

8. **Train it.** Port `transformer.py` to PyTorch and train it to reverse or copy short sequences of digits. Report the loss curve and a few greedy decodes.
