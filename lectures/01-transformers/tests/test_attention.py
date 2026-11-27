import numpy as np

from l01.attention import (
    MultiHeadAttention,
    causal_mask,
    padding_mask,
    scaled_dot_product_attention,
    softmax,
)

rng = np.random.default_rng(0)


def test_softmax_rows_sum_to_one_and_handle_large_values():
    p = softmax(np.array([[1000.0, 1000.0], [1.0, 2.0]]))
    assert np.allclose(p.sum(axis=-1), 1.0)
    assert np.allclose(p[0], [0.5, 0.5])


def test_attention_shapes_and_weights():
    Q, K, V = rng.normal(size=(3, 8)), rng.normal(size=(5, 8)), rng.normal(size=(5, 4))
    out, w = scaled_dot_product_attention(Q, K, V)
    assert out.shape == (3, 4)
    assert w.shape == (3, 5)
    assert np.allclose(w.sum(axis=-1), 1.0)


def test_causal_mask_blocks_the_future():
    x = rng.normal(size=(4, 8))
    _, w = scaled_dot_product_attention(x, x, x, causal_mask(4))
    assert np.allclose(np.triu(w, k=1), 0.0)
    assert np.isclose(w[0, 0], 1.0)  # the first token can only see itself


def test_padding_mask_ignores_pad_keys():
    x = rng.normal(size=(5, 8))
    _, w = scaled_dot_product_attention(x, x, x, padding_mask(3, 5))
    assert np.allclose(w[:, 3:], 0.0)


def test_scaling_keeps_score_variance_near_one():
    d_k = 512
    q, k = rng.normal(size=(2000, d_k)), rng.normal(size=(2000, d_k))
    raw = (q * k).sum(axis=-1)
    assert 0.8 * d_k < raw.var() < 1.2 * d_k
    assert 0.8 < (raw / np.sqrt(d_k)).var() < 1.2


def test_self_attention_without_positions_is_permutation_equivariant():
    mha = MultiHeadAttention(d_model=16, num_heads=4, rng=np.random.default_rng(1))
    x = rng.normal(size=(6, 16))
    perm = rng.permutation(6)
    out, _ = mha(x)
    out_perm, _ = mha(x[perm])
    assert np.allclose(out[perm], out_perm)


def test_multi_head_shapes_and_parameter_count():
    mha = MultiHeadAttention(d_model=512, num_heads=8, rng=np.random.default_rng(0))
    out, w = mha(rng.normal(size=(7, 512)), rng.normal(size=(9, 512)))
    assert out.shape == (7, 512)
    assert w.shape == (8, 7, 9)
    assert mha.num_parameters() == 4 * 512**2 == 1_048_576
