"""Byte-pair encoding (BPE), the subword tokenizer behind GPT-style models.

Training starts from single characters and repeatedly merges the most frequent
adjacent pair of symbols into a new symbol, until the vocabulary reaches the
target size. Encoding replays the learned merges in the order they were learned.

This is the classic word-level BPE of Sennrich et al. (2016): text is split on
whitespace first, and each word ends with an end-of-word marker so that "low"
at the end of a word and "low" inside "lower" can be told apart.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import pairwise

END_OF_WORD = "</w>"

Pair = tuple[str, str]


def _word_to_symbols(word: str) -> tuple[str, ...]:
    return (*word, END_OF_WORD)


def count_pairs(word_freqs: dict[tuple[str, ...], int]) -> Counter[Pair]:
    """Count every adjacent symbol pair, weighted by how often its word occurs."""
    pairs: Counter[Pair] = Counter()
    for symbols, freq in word_freqs.items():
        for a, b in pairwise(symbols):
            pairs[(a, b)] += freq
    return pairs


def merge_pair(symbols: tuple[str, ...], pair: Pair) -> tuple[str, ...]:
    """Replace every occurrence of `pair` in `symbols` with the joined symbol."""
    out: list[str] = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            out.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


@dataclass
class BPETokenizer:
    merges: list[Pair] = field(default_factory=list)
    vocab: dict[str, int] = field(default_factory=dict)

    @classmethod
    def train(cls, corpus: str, num_merges: int) -> BPETokenizer:
        word_freqs: dict[tuple[str, ...], int] = Counter(
            _word_to_symbols(w) for w in corpus.split()
        )
        symbols = {s for word in word_freqs for s in word}
        merges: list[Pair] = []

        for _ in range(num_merges):
            pairs = count_pairs(word_freqs)
            if not pairs:
                break
            # Highest count wins; ties are broken lexicographically so training is deterministic.
            best = max(pairs, key=lambda p: (pairs[p], p))
            merges.append(best)
            symbols.add(best[0] + best[1])
            word_freqs = {merge_pair(w, best): f for w, f in word_freqs.items()}

        vocab = {s: i for i, s in enumerate(sorted(symbols))}
        return cls(merges=merges, vocab=vocab)

    def tokenize(self, text: str) -> list[str]:
        """Split text into subword strings."""
        tokens: list[str] = []
        for word in text.split():
            symbols = _word_to_symbols(word)
            for pair in self.merges:
                symbols = merge_pair(symbols, pair)
            tokens.extend(symbols)
        return tokens

    def encode(self, text: str) -> list[int]:
        """Map text to token ids. Characters never seen in training raise KeyError."""
        return [self.vocab[t] for t in self.tokenize(text)]

    def decode(self, ids: list[int]) -> str:
        inverse = {i: s for s, i in self.vocab.items()}
        text = "".join(inverse[i] for i in ids)
        return text.replace(END_OF_WORD, " ").strip()
