from hypothesis import given
from hypothesis import strategies as st

from l01.tokenization import END_OF_WORD, BPETokenizer

CORPUS = "low " * 5 + "lower " * 2 + "newest " * 6 + "widest " * 3
TOKENIZER = BPETokenizer.train(CORPUS, num_merges=10)
ALPHABET = sorted(set(CORPUS.replace(" ", "")))

words = st.text(alphabet=ALPHABET, min_size=1, max_size=12)
texts = st.lists(words, min_size=1, max_size=6).map(" ".join)


@given(texts)
def test_decode_inverts_encode_for_any_text_over_the_training_alphabet(text):
    assert TOKENIZER.decode(TOKENIZER.encode(text)) == text


@given(words)
def test_a_word_never_needs_more_tokens_than_characters_plus_end_marker(word):
    assert len(TOKENIZER.tokenize(word)) <= len(word) + 1


@given(words)
def test_every_word_ends_with_exactly_one_end_of_word_marker(word):
    tokens = TOKENIZER.tokenize(word)
    assert tokens[-1].endswith(END_OF_WORD)
    assert sum(t.count(END_OF_WORD) for t in tokens) == 1
