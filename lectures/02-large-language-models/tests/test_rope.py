import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from l02.rope import apply_rope, rope_frequencies, rotation_matrix

rng = np.random.default_rng(0)


def test_frequencies_follow_the_geometric_schedule():
    np.testing.assert_allclose(rope_frequencies(8), [1.0, 10000**-0.25, 10000**-0.5, 10000**-0.75])


def test_odd_head_dim_is_rejected():
    with pytest.raises(ValueError):
        rope_frequencies(7)


def test_position_zero_is_the_identity():
    x = rng.normal(size=(1, 8))
    np.testing.assert_allclose(apply_rope(x, np.array([0])), x)


def test_two_dimensional_case_is_a_rotation_by_m_radians():
    x = np.array([[1.0, 0.0]])
    out = apply_rope(x, np.array([3]))
    np.testing.assert_allclose(out, [[np.cos(3.0), np.sin(3.0)]])


def test_matches_the_explicit_block_diagonal_rotation_matrix():
    x = rng.normal(size=(5, 8))
    positions = np.arange(5)
    expected = np.stack([rotation_matrix(m, 8) @ x[m] for m in positions])
    np.testing.assert_allclose(apply_rope(x, positions), expected)


def test_rotation_matrix_is_orthogonal():
    R = rotation_matrix(7, 16)
    np.testing.assert_allclose(R.T @ R, np.eye(16), atol=1e-12)


def test_heads_are_rotated_independently():
    x = rng.normal(size=(3, 4, 8))  # (heads, n, d_head)
    out = apply_rope(x, np.arange(4))
    for h in range(3):
        np.testing.assert_allclose(out[h], apply_rope(x[h], np.arange(4)))


head_vectors = hnp.arrays(np.float64, 16, elements=st.floats(-3, 3, width=64))
positions = st.integers(0, 2000)


@given(head_vectors, head_vectors, positions, positions, st.integers(0, 2000))
def test_score_depends_only_on_relative_position(q, k, m, n, shift):
    def score(a, b):
        return apply_rope(q[None], np.array([a]))[0] @ apply_rope(k[None], np.array([b]))[0]

    np.testing.assert_allclose(score(m + shift, n + shift), score(m, n), rtol=1e-7, atol=1e-7)


@given(head_vectors, positions)
def test_rotation_preserves_the_norm(x, m):
    out = apply_rope(x[None], np.array([m]))[0]
    np.testing.assert_allclose(np.linalg.norm(out), np.linalg.norm(x), rtol=1e-10, atol=1e-12)
