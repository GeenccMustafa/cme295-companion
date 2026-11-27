from dataclasses import replace

import numpy as np
import pytest

from l01.transformer import cross_entropy
from l02.model import DecoderBlock, DecoderOnlyLM, LMConfig

DENSE = LMConfig(vocab_size=50, d_model=32, num_layers=2, num_heads=4, num_kv_heads=2, d_ff=64)
MOE = replace(DENSE, num_experts=4, top_k=2)
IDS = np.array([1, 7, 3, 9, 4, 2])


@pytest.mark.parametrize("cfg", [DENSE, MOE], ids=["dense", "moe"])
def test_logits_have_one_row_per_position(cfg):
    logits, _ = DecoderOnlyLM(cfg)(IDS)
    assert logits.shape == (6, 50)


@pytest.mark.parametrize("cfg", [DENSE, MOE], ids=["dense", "moe"])
def test_model_is_causal(cfg):
    model = DecoderOnlyLM(cfg)
    a, _ = model(np.array([1, 7, 3, 9, 4, 2]))
    b, _ = model(np.array([1, 7, 3, 5, 8, 8]))
    np.testing.assert_allclose(a[:3], b[:3], atol=1e-12)
    assert not np.allclose(a[3:], b[3:])


def test_dense_model_has_no_auxiliary_loss_and_moe_model_has_one():
    assert DecoderOnlyLM(DENSE)(IDS)[1] == 0.0
    assert DecoderOnlyLM(MOE)(IDS)[1] > 0.0


def test_next_token_loss_compares_each_position_with_the_following_token():
    model = DecoderOnlyLM(DENSE)
    logits, _ = model(IDS)
    assert model.next_token_loss(IDS) == pytest.approx(cross_entropy(logits[:-1], IDS[1:]))


def test_pre_norm_block_is_the_identity_when_sublayers_output_zero():
    block = DecoderBlock(DENSE, np.random.default_rng(0))
    block.attn.W_O[:] = 0.0
    block.ffn.W_down[:] = 0.0
    x = np.random.default_rng(1).normal(size=(5, 32))
    out, _ = block(x, np.arange(5))
    np.testing.assert_allclose(out, x)


def test_dense_parameter_count_matches_the_formula():
    d, V, L, g, d_head, d_ff = 32, 50, 2, 2, 8, 64
    attention = 2 * d * d + 2 * d * g * d_head
    ffn = 3 * d * d_ff
    norms = 2 * d
    expected = V * d + L * (attention + ffn + norms) + d  # tied embeddings, final norm
    assert DecoderOnlyLM(DENSE).num_parameters() == expected


def test_untied_output_projection_adds_a_vocabulary_matrix():
    untied = replace(DENSE, tie_embeddings=False)
    extra = DecoderOnlyLM(untied).num_parameters() - DecoderOnlyLM(DENSE).num_parameters()
    assert extra == 32 * 50


def test_moe_model_activates_fewer_parameters_than_it_stores():
    model = DecoderOnlyLM(MOE)
    assert model.num_active_parameters() < model.num_parameters()
    assert DecoderOnlyLM(DENSE).num_active_parameters() == DecoderOnlyLM(DENSE).num_parameters()


def test_sliding_window_reaches_every_attention_layer():
    model = DecoderOnlyLM(replace(DENSE, sliding_window=3))
    assert all(block.attn.window == 3 for block in model.blocks)


def test_generate_appends_new_tokens_after_the_prompt():
    out = DecoderOnlyLM(DENSE).generate([1, 7, 3], max_new_tokens=4, rng=np.random.default_rng(0))
    assert out[:3] == [1, 7, 3]
    assert len(out) == 7


def test_generation_at_zero_temperature_is_deterministic():
    model = DecoderOnlyLM(DENSE)
    runs = {
        tuple(model.generate([1, 7], 5, rng=np.random.default_rng(s), temperature=0.0))
        for s in range(3)
    }
    assert len(runs) == 1


def test_generation_stops_at_end_of_sequence():
    model = DecoderOnlyLM(DENSE)
    greedy = model.generate([1, 7], 1, rng=np.random.default_rng(0), temperature=0.0)
    eos = greedy[-1]
    out = model.generate([1, 7], 10, rng=np.random.default_rng(0), temperature=0.0, eos_id=eos)
    assert out == [1, 7, eos]
