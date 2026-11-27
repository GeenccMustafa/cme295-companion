import numpy as np

from l01.positional_encoding import sinusoidal_positional_encoding
from l01.transformer import Transformer, TransformerConfig, cross_entropy

CFG = TransformerConfig(vocab_size=50, d_model=32, num_heads=4, d_ff=64, num_layers=2, max_len=20)


def test_positional_encoding_values():
    pe = sinusoidal_positional_encoding(10, 4)
    assert pe.shape == (10, 4)
    assert np.allclose(pe[0], [0, 1, 0, 1])
    assert np.isclose(pe[1, 0], np.sin(1.0))
    assert np.isclose(pe[1, 2], np.sin(1.0 / 100))


def test_encoder_decoder_shapes():
    model = Transformer(CFG)
    src, tgt = np.array([1, 5, 7, 9, 2]), np.array([1, 4, 6])
    memory = model.encode(src)
    assert memory.shape == (5, 32)
    assert model.decode(tgt, memory).shape == (3, 50)


def test_decoder_is_causal():
    """Changing a later target token must not change the logits at earlier positions."""
    model = Transformer(CFG)
    src = np.array([1, 5, 7, 2])
    a = model(src, np.array([1, 4, 6, 8]))
    b = model(src, np.array([1, 4, 9, 3]))
    assert np.allclose(a[:2], b[:2])
    assert not np.allclose(a[2:], b[2:])


def test_greedy_decode_stops_at_max_len():
    model = Transformer(CFG)
    out = model.greedy_decode(np.array([1, 5, 2]), bos_id=1, eos_id=2, max_new=5)
    assert out[0] == 1
    assert 2 <= len(out) <= 6


def test_cross_entropy_uniform_logits_is_log_vocab():
    loss = cross_entropy(np.zeros((4, 50)), np.array([0, 1, 2, 3]))
    assert np.isclose(loss, np.log(50))


def test_label_smoothing_penalizes_overconfidence():
    confident = np.full((1, 5), -20.0)
    confident[0, 0] = 20.0
    hedged = np.log(np.array([[0.92, 0.02, 0.02, 0.02, 0.02]]))
    target = np.array([0])
    assert cross_entropy(confident, target) < cross_entropy(hedged, target)
    assert cross_entropy(confident, target, 0.1) > cross_entropy(hedged, target, 0.1)
