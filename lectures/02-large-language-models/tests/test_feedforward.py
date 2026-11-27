import numpy as np

from l02.feedforward import SwiGLUFeedForward, silu


def test_silu_matches_definition():
    x = np.array([-2.0, 0.0, 3.0])
    np.testing.assert_allclose(silu(x), x / (1 + np.exp(-x)))


def test_silu_is_close_to_identity_for_large_inputs_and_to_zero_for_very_negative_ones():
    np.testing.assert_allclose(silu(np.array([30.0, -30.0])), [30.0, 0.0], atol=1e-10)


def test_swiglu_output_shape():
    ffn = SwiGLUFeedForward(d_model=16, d_ff=40, rng=np.random.default_rng(0))
    assert ffn(np.ones((5, 16))).shape == (5, 16)


def test_swiglu_has_three_weight_matrices():
    ffn = SwiGLUFeedForward(d_model=4096, d_ff=14336, rng=np.random.default_rng(0))
    assert ffn.num_parameters() == 3 * 4096 * 14336


def test_swiglu_is_applied_to_each_position_independently():
    ffn = SwiGLUFeedForward(d_model=8, d_ff=24, rng=np.random.default_rng(0))
    x = np.random.default_rng(1).normal(size=(4, 8))
    rows = np.stack([ffn(row[None])[0] for row in x])
    np.testing.assert_allclose(ffn(x), rows)


def test_swiglu_gate_closes_when_gate_preactivation_is_very_negative():
    ffn = SwiGLUFeedForward(d_model=2, d_ff=2, rng=np.random.default_rng(0))
    ffn.W_gate = -100.0 * np.eye(2)
    out = ffn(np.array([[1.0, 1.0]]))
    np.testing.assert_allclose(out, 0.0, atol=1e-30)


def test_silu_does_not_overflow_for_very_negative_inputs():
    with np.errstate(all="raise"):
        out = silu(np.array([-1000.0, 1000.0]))
    np.testing.assert_allclose(out, [0.0, 1000.0])
