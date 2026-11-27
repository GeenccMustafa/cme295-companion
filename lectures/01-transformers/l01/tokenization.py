"""Byte-pair encoding (BPE) subword tokenization.

Language models do not read characters or words; they read *tokens*, integer
ids drawn from a fixed vocabulary. Word-level vocabularies explode in size and
cannot represent unseen words, while character-level sequences become very
long. Subword tokenization sits between the two: frequent words become a
single token, rare words are spelled out from smaller, reusable pieces.

BPE builds such a vocabulary bottom-up. Training starts from individual
characters and repeatedly merges the most frequent pair of adjacent symbols
into a new symbol, recording each merge. Tokenization replays the recorded
merges, in order, on new text.

This module implements the word-level variant introduced for neural machine
translation by Sennrich et al. [1]_. Text is pre-tokenized on whitespace and
every word is terminated by an end-of-word marker, so that a suffix such as
``"est</w>"`` is distinguished from the same characters inside a word.

References
----------
.. [1] R. Sennrich, B. Haddow and A. Birch, "Neural Machine Translation of
   Rare Words with Subword Units", ACL 2016. https://arxiv.org/abs/1508.07909

Examples
--------
>>> corpus = "low " * 5 + "lower " * 2 + "newest " * 6 + "widest " * 3
>>> tokenizer = BPETokenizer.train(corpus, num_merges=10)
>>> tokenizer.tokenize("lowest")
['low', 'est</w>']
>>> tokenizer.decode(tokenizer.encode("lowest newer"))
'lowest newer'
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import pairwise

__all__ = ["END_OF_WORD", "BPETokenizer", "Pair", "count_pairs", "merge_pair"]

END_OF_WORD = "</w>"
"""Marker appended to every word so that word-final subwords get their own tokens."""

Pair = tuple[str, str]
"""Two adjacent symbols that are candidates for a merge."""


def _word_to_symbols(word: str) -> tuple[str, ...]:
    """Split a word into characters and append the end-of-word marker."""
    return (*word, END_OF_WORD)


def count_pairs(word_freqs: dict[tuple[str, ...], int]) -> Counter[Pair]:
    """Count adjacent symbol pairs across a corpus, weighted by word frequency.

    Parameters
    ----------
    word_freqs : dict of tuple of str to int
        Each key is a word, already split into its current symbols; each
        value is how many times that word occurs in the corpus.

    Returns
    -------
    collections.Counter
        Maps every adjacent pair ``(a, b)`` to its total number of
        occurrences.

    Examples
    --------
    >>> count_pairs({("l", "o", "w"): 5, ("l", "o", "g"): 2})
    Counter({('l', 'o'): 7, ('o', 'w'): 5, ('o', 'g'): 2})
    """
    pairs: Counter[Pair] = Counter()
    for symbols, freq in word_freqs.items():
        for a, b in pairwise(symbols):
            pairs[(a, b)] += freq
    return pairs


def merge_pair(symbols: tuple[str, ...], pair: Pair) -> tuple[str, ...]:
    """Merge every non-overlapping occurrence of a pair into a single symbol.

    The sequence is scanned left to right, so for ``("a", "a", "a")`` and the
    pair ``("a", "a")`` only the first two symbols are merged.

    Parameters
    ----------
    symbols : tuple of str
        The current segmentation of one word.
    pair : tuple of (str, str)
        The pair to merge.

    Returns
    -------
    tuple of str
        The new segmentation, never longer than `symbols`.

    Examples
    --------
    >>> merge_pair(("a", "b", "a", "b", "c"), ("a", "b"))
    ('ab', 'ab', 'c')
    >>> merge_pair(("a", "a", "a"), ("a", "a"))
    ('aa', 'a')
    """
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
    """A trained byte-pair encoding tokenizer.

    Instances are normally created with :meth:`train` rather than directly.

    Attributes
    ----------
    merges : list of tuple of (str, str)
        Learned merge rules, in the order they were learned. The order
        matters: tokenization applies them in exactly this sequence.
    vocab : dict of str to int
        Maps every symbol (initial characters, the end-of-word marker and
        every merged symbol) to its integer id.
    """

    merges: list[Pair] = field(default_factory=list)
    vocab: dict[str, int] = field(default_factory=dict)

    @classmethod
    def train(cls, corpus: str, num_merges: int) -> BPETokenizer:
        """Learn merge rules from a corpus.

        Parameters
        ----------
        corpus : str
            Training text. It is split on whitespace into words.
        num_merges : int
            Maximum number of merges to learn. The final vocabulary size is
            the number of distinct characters plus one (the end-of-word
            marker) plus the number of merges actually performed. Training
            stops early if every word has become a single symbol.

        Returns
        -------
        BPETokenizer
            The trained tokenizer.

        Notes
        -----
        At each step the most frequent pair is merged. Ties are broken by
        taking the lexicographically largest pair, which makes training
        fully deterministic.
        """
        word_freqs: dict[tuple[str, ...], int] = Counter(
            _word_to_symbols(w) for w in corpus.split()
        )
        symbols = {s for word in word_freqs for s in word}
        merges: list[Pair] = []

        for _ in range(num_merges):
            pairs = count_pairs(word_freqs)
            if not pairs:
                break
            best = max(pairs, key=lambda p: (pairs[p], p))
            merges.append(best)
            symbols.add(best[0] + best[1])
            word_freqs = {merge_pair(w, best): f for w, f in word_freqs.items()}

        vocab = {s: i for i, s in enumerate(sorted(symbols))}
        return cls(merges=merges, vocab=vocab)

    def tokenize(self, text: str) -> list[str]:
        """Split text into subword strings.

        Parameters
        ----------
        text : str
            Input text. It is split on whitespace into words.

        Returns
        -------
        list of str
            Subword tokens. The last token of every word ends with
            :data:`END_OF_WORD`.
        """
        tokens: list[str] = []
        for word in text.split():
            symbols = _word_to_symbols(word)
            for pair in self.merges:
                symbols = merge_pair(symbols, pair)
            tokens.extend(symbols)
        return tokens

    def encode(self, text: str) -> list[int]:
        """Convert text to token ids.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        list of int
            One id per subword token.

        Raises
        ------
        KeyError
            If the text contains a character that never appeared in the
            training corpus. Byte-level BPE avoids this by starting from the
            256 possible byte values instead of characters.
        """
        return [self.vocab[t] for t in self.tokenize(text)]

    def decode(self, ids: list[int]) -> str:
        """Convert token ids back to text.

        Parameters
        ----------
        ids : list of int
            Token ids, as produced by :meth:`encode`.

        Returns
        -------
        str
            The reconstructed text, with words separated by single spaces.
        """
        inverse = {i: s for s, i in self.vocab.items()}
        text = "".join(inverse[i] for i in ids)
        return text.replace(END_OF_WORD, " ").strip()
