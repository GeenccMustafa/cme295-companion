from l01.tokenization import END_OF_WORD, BPETokenizer, count_pairs, merge_pair

CORPUS = "low " * 5 + "lower " * 2 + "newest " * 6 + "widest " * 3


def test_merge_pair_replaces_all_occurrences():
    assert merge_pair(("a", "b", "a", "b", "c"), ("a", "b")) == ("ab", "ab", "c")


def test_first_merge_is_most_frequent_pair():
    tok = BPETokenizer.train(CORPUS, num_merges=1)
    # "es", "st" and "t</w>" each appear 9 times (newest x6 + widest x3).
    pairs = count_pairs({(*w, END_OF_WORD): f for w, f in [("newest", 6), ("widest", 3)]})
    assert pairs[("e", "s")] == pairs[("s", "t")] == pairs[("t", END_OF_WORD)] == 9
    # Ties are broken by taking the lexicographically largest pair.
    assert tok.merges[0] == ("t", END_OF_WORD)


def test_round_trip():
    tok = BPETokenizer.train(CORPUS, num_merges=10)
    text = "lowest newer"
    assert tok.decode(tok.encode(text)) == text


def test_frequent_word_becomes_single_token():
    tok = BPETokenizer.train(CORPUS, num_merges=20)
    assert tok.tokenize("newest") == ["newest" + END_OF_WORD]


def test_more_merges_means_fewer_tokens():
    few = BPETokenizer.train(CORPUS, num_merges=2)
    many = BPETokenizer.train(CORPUS, num_merges=15)
    assert len(many.tokenize("lowest widest")) < len(few.tokenize("lowest widest"))
