import numpy as np
import pytest

from l01.attention import MultiHeadAttention, causal_mask
from l02.attention import (
    GroupedQueryAttention,
    kv_cache_bytes,
    receptive_field,
    sliding_window_mask,
)

rng = np.random.default_rng(0)


def test_sliding_window_mask_keeps_the_last_w_positions_including_itself():
    expected = np.array(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(sliding_window_mask(5, window=3), expected)


def test_sliding_window_as_wide_as_the_sequence_is_plain_causal_attention():
    np.testing.assert_array_equal(sliding_window_mask(6, window=6), causal_mask(6))


def test_two_local_layers_reach_back_two_window_steps():
    # Midterm II.10: window t, t-1, t-2 and two layers. The earliest input that can
    # influence the 6th position is the 2nd (0-based: position 5 is reached from 1).
    reach = receptive_field(sliding_window_mask(8, window=3), num_layers=2)
    assert reach[5].nonzero()[0].min() == 1


def test_receptive_field_grows_linearly_with_depth():
    mask = sliding_window_mask(64, window=4)
    for layers in range(1, 6):
        earliest = receptive_field(mask, layers)[63].nonzero()[0].min()
        assert 63 - earliest == layers * (4 - 1)


def test_head_count_must_be_divisible_by_kv_head_count():
    with pytest.raises(ValueError):
        GroupedQueryAttention(d_model=32, num_heads=4, num_kv_heads=3, rng=rng)


def test_gqa_output_shapes():
    attn = GroupedQueryAttention(d_model=32, num_heads=8, num_kv_heads=2, rng=rng)
    out, weights = attn(rng.normal(size=(5, 32)))
    assert out.shape == (5, 32)
    assert weights.shape == (8, 5, 5)


def test_gqa_is_causal():
    attn = GroupedQueryAttention(d_model=16, num_heads=4, num_kv_heads=2, rng=rng)
    _, weights = attn(rng.normal(size=(6, 16)))
    assert np.allclose(np.triu(weights, k=1), 0.0)


def test_sliding_window_blocks_distant_keys():
    attn = GroupedQueryAttention(d_model=16, num_heads=4, num_kv_heads=1, rng=rng, window=2)
    _, weights = attn(rng.normal(size=(6, 16)))
    assert np.allclose(weights[:, 5, :4], 0.0)


def test_parameter_count_shrinks_with_fewer_kv_heads():
    d, h, d_head = 4096, 32, 128
    mha = GroupedQueryAttention(d, h, num_kv_heads=32, rng=rng)
    gqa = GroupedQueryAttention(d, h, num_kv_heads=8, rng=rng)
    mqa = GroupedQueryAttention(d, h, num_kv_heads=1, rng=rng)
    assert mha.num_parameters() == 4 * d * d
    assert gqa.num_parameters() == 2 * d * d + 2 * d * 8 * d_head
    assert mqa.num_parameters() == 2 * d * d + 2 * d * d_head


def test_kv_cache_matches_mistral_7b():
    # 32 layers, 8 KV heads of size 128, bf16: 128 KiB per token, 4 GiB for 32K tokens.
    assert kv_cache_bytes(32, 8, 128, num_tokens=1) == 128 * 1024
    assert kv_cache_bytes(32, 8, 128, num_tokens=32 * 1024) == 4 * 1024**3


def test_kv_cache_scalars_per_token_match_midterm_question():
    # Midterm II.3: 8 query heads, d_head 64, 2 KV groups -> 256 K+V values per token per layer.
    assert kv_cache_bytes(1, 2, 64, num_tokens=1, bytes_per_value=1) == 256


# Differential tests against Lecture 01's multi-head attention.


def _mha_with_weights_of(gqa: GroupedQueryAttention) -> MultiHeadAttention:
    """Build an l01 MHA that computes the same function as `gqa` without RoPE.

    Each KV head of `gqa` is repeated for every query head in its group.
    """
    mha = MultiHeadAttention(gqa.d_model, gqa.num_heads, np.random.default_rng(99))
    group = gqa.num_heads // gqa.num_kv_heads
    d_head = gqa.head_dim
    kv_heads_k = gqa.W_K.reshape(gqa.d_model, gqa.num_kv_heads, d_head)
    kv_heads_v = gqa.W_V.reshape(gqa.d_model, gqa.num_kv_heads, d_head)
    mha.W_Q, mha.W_O = gqa.W_Q, gqa.W_O
    mha.W_K = np.repeat(kv_heads_k, group, axis=1).reshape(gqa.d_model, gqa.d_model)
    mha.W_V = np.repeat(kv_heads_v, group, axis=1).reshape(gqa.d_model, gqa.d_model)
    return mha


@pytest.mark.parametrize("num_kv_heads", [4, 2, 1])
def test_gqa_without_rope_equals_mha_with_shared_kv_heads(num_kv_heads):
    gqa = GroupedQueryAttention(32, 4, num_kv_heads, rng=np.random.default_rng(1), rope_base=None)
    mha = _mha_with_weights_of(gqa)
    x = rng.normal(size=(7, 32))
    ours, w_ours = gqa(x)
    ref, w_ref = mha(x, mask=causal_mask(7))
    np.testing.assert_allclose(ours, ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(w_ours, w_ref, rtol=1e-12, atol=1e-12)


def test_rope_makes_attention_scores_shift_invariant():
    attn = GroupedQueryAttention(16, 4, 2, rng=np.random.default_rng(2))
    x = rng.normal(size=(5, 16))
    _, w0 = attn(x, positions=np.arange(5))
    _, w1 = attn(x, positions=np.arange(5) + 1000)
    np.testing.assert_allclose(w0, w1, atol=1e-9)
