from l01.word2vec import SkipGram, skipgram_pairs


def test_skipgram_pairs_window_one():
    assert skipgram_pairs([0, 1, 2], window=1) == [(0, 1), (1, 0), (1, 2), (2, 1)]


def test_training_reduces_loss_and_groups_similar_words():
    # Two "topics" that never share context: {0,1,2} and {3,4,5}.
    sentences = [[0, 1, 2, 0, 2, 1], [3, 4, 5, 3, 5, 4]] * 20
    pairs = [p for s in sentences for p in skipgram_pairs(s, window=2)]
    model = SkipGram(vocab_size=6, dim=8, seed=0)
    history = model.train(pairs, epochs=15, num_negatives=2, lr=0.05)
    assert history[-1] < history[0]
    assert model.similarity(0, 1) > model.similarity(0, 4)
