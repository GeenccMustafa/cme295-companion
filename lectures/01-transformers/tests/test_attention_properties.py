import hypothesis.extra.numpy as hnp
import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from l01.attention import MultiHeadAttention, causal_mask, scaled_dot_product_attention, softmax
from l01.positional_encoding import sinusoidal_positional_encoding
from l01.transformer import LayerNorm

finite = st.floats(-50, 50, allow_nan=False, allow_infinity=False, width=64)
matrices = hnp.arrays(
    np.float64, hnp.array_shapes(min_dims=2, max_dims=2, max_side=8), elements=finite
)
sequences = st.integers(1, 6).flatmap(
    lambda n: hnp.arrays(np.float64, (n, 8), elements=st.floats(-3, 3, width=64))
)


@given(matrices)
def test_softmax_rows_are_probability_distributions(x):
    p = softmax(x)
    assert np.all(p >= 0)
    np.testing.assert_allclose(p.sum(axis=-1), 1.0, rtol=1e-12)


@given(matrices, finite)
def test_softmax_is_invariant_to_adding_a_constant(x, c):
    np.testing.assert_allclose(softmax(x + c), softmax(x), atol=1e-12)


@given(sequences, st.data())
def test_causal_attention_ignores_changes_to_later_positions(x, data):
    n = x.shape[0]
    j = data.draw(st.integers(0, n - 1))
    changed = x.copy()
    changed[j:] += 1.0
    out, _ = scaled_dot_product_attention(x, x, x, causal_mask(n))
    out_changed, _ = scaled_dot_product_attention(changed, changed, changed, causal_mask(n))
    np.testing.assert_allclose(out[:j], out_changed[:j], atol=1e-12)


@given(sequences, st.randoms(use_true_random=False))
def test_self_attention_without_positions_is_permutation_equivariant(x, random):
    mha = MultiHeadAttention(d_model=8, num_heads=2, rng=np.random.default_rng(0))
    perm = np.array(random.sample(range(len(x)), len(x)))
    out, _ = mha(x)
    out_perm, _ = mha(x[perm])
    np.testing.assert_allclose(out[perm], out_perm, atol=1e-10)


@given(sequences.filter(lambda x: np.all(x.std(axis=-1) > 1e-3)))
def test_layer_norm_output_has_zero_mean_and_unit_variance(x):
    y = LayerNorm(x.shape[-1], eps=0.0)(x)
    np.testing.assert_allclose(y.mean(axis=-1), 0.0, atol=1e-10)
    np.testing.assert_allclose(y.var(axis=-1), 1.0, rtol=1e-8)


@given(st.integers(1, 200), st.integers(1, 32).map(lambda k: 2 * k))
def test_positional_encoding_values_lie_in_minus_one_to_one(max_len, d_model):
    pe = sinusoidal_positional_encoding(max_len, d_model)
    assert pe.shape == (max_len, d_model)
    assert np.all(np.abs(pe) <= 1.0)
