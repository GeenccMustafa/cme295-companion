import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from l01.transformer import LayerNorm
from l02.normalization import RMSNorm

vectors = hnp.arrays(
    np.float64,
    st.integers(2, 16),
    elements=st.floats(-10, 10, allow_nan=False, width=64),
).filter(lambda x: np.sqrt(np.mean(x**2)) > 1e-3)


def test_rms_norm_matches_worked_example():
    out = RMSNorm(4, eps=0.0)(np.array([1.0, 2.0, 3.0, 6.0]))
    np.testing.assert_allclose(out, [0.28284, 0.56569, 0.84853, 1.69706], atol=1e-5)


def test_rms_norm_does_not_center_unlike_layer_norm():
    x = np.array([1.0, 2.0, 3.0, 6.0])
    assert abs(RMSNorm(4, eps=0.0)(x).mean()) > 0.5
    assert abs(LayerNorm(4, eps=0.0)(x).mean()) < 1e-12


def test_rms_norm_has_half_the_parameters_of_layer_norm():
    assert RMSNorm(512).num_parameters() == 512


def test_rms_norm_normalizes_each_row_independently():
    x = np.array([[1.0, 2.0, 3.0, 6.0], [10.0, 0.0, 0.0, 0.0]])
    out = RMSNorm(4, eps=0.0)(x)
    np.testing.assert_allclose(out[0], RMSNorm(4, eps=0.0)(x[0]))
    np.testing.assert_allclose(out[1], [2.0, 0.0, 0.0, 0.0])


@given(vectors)
def test_rms_norm_output_has_unit_root_mean_square(x):
    out = RMSNorm(len(x), eps=0.0)(x)
    np.testing.assert_allclose(np.sqrt(np.mean(out**2)), 1.0, rtol=1e-10)


@given(vectors, st.floats(1e-2, 1e2))
def test_rms_norm_is_invariant_to_positive_rescaling(x, c):
    norm = RMSNorm(len(x), eps=0.0)
    np.testing.assert_allclose(norm(c * x), norm(x), rtol=1e-9, atol=1e-12)
