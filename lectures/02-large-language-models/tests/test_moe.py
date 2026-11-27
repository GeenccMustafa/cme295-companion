import numpy as np
import pytest

from l01.attention import softmax
from l02.moe import MixtureOfExperts, load_balancing_loss

rng = np.random.default_rng(0)


def make_moe(num_experts=4, top_k=2, d_model=8, d_ff=16, seed=0):
    return MixtureOfExperts(d_model, d_ff, num_experts, top_k, rng=np.random.default_rng(seed))


def test_top_k_cannot_exceed_number_of_experts():
    with pytest.raises(ValueError):
        make_moe(num_experts=4, top_k=5)


def test_output_shape_and_routing_shapes():
    out, routing = make_moe()(rng.normal(size=(6, 8)))
    assert out.shape == (6, 8)
    assert routing.probs.shape == (6, 4)
    assert routing.experts.shape == routing.gates.shape == (6, 2)


def test_each_token_picks_k_distinct_experts_with_gates_summing_to_one():
    _, routing = make_moe(num_experts=8, top_k=3)(rng.normal(size=(10, 8)))
    assert all(len(set(row)) == 3 for row in routing.experts.tolist())
    assert np.all(routing.gates > 0)
    np.testing.assert_allclose(routing.gates.sum(axis=-1), 1.0)


def test_selected_experts_are_those_with_the_highest_router_scores():
    moe = make_moe(num_experts=6, top_k=2)
    x = rng.normal(size=(5, 8))
    _, routing = moe(x)
    best_two = np.argsort(-(x @ moe.W_router), axis=-1)[:, :2]
    np.testing.assert_array_equal(np.sort(routing.experts, axis=-1), np.sort(best_two, axis=-1))


def test_output_is_the_gate_weighted_sum_of_selected_experts():
    moe = make_moe()
    x = rng.normal(size=(3, 8))
    out, routing = moe(x)
    for t in range(3):
        expected = sum(
            g * moe.experts[e](x[t : t + 1])[0]
            for e, g in zip(routing.experts[t], routing.gates[t], strict=True)
        )
        np.testing.assert_allclose(out[t], expected)


def test_routing_to_every_expert_is_a_dense_mixture():
    moe = make_moe(num_experts=4, top_k=4)
    x = rng.normal(size=(5, 8))
    out, _ = moe(x)
    weights = softmax(x @ moe.W_router)  # (n, N)
    dense = sum(weights[:, [e]] * moe.experts[e](x) for e in range(4))
    np.testing.assert_allclose(out, dense)


class CountingExpert:
    """Wraps an expert and records how many token vectors it processes."""

    def __init__(self, expert):
        self.expert = expert
        self.rows = 0

    def __call__(self, z):
        self.rows += len(z)
        return self.expert(z)


def test_only_k_experts_run_per_token():
    moe = make_moe(num_experts=8, top_k=2)
    moe.experts = [CountingExpert(e) for e in moe.experts]
    moe(rng.normal(size=(12, 8)))
    assert sum(e.rows for e in moe.experts) == 12 * 2


def test_active_parameters_count_only_the_selected_experts():
    moe = MixtureOfExperts(d_model=64, d_ff=128, num_experts=8, top_k=2, rng=rng)
    expert = 3 * 64 * 128
    router = 64 * 8
    assert moe.num_parameters() == router + 8 * expert
    assert moe.num_active_parameters() == router + 2 * expert


def test_load_balancing_loss_is_one_for_perfectly_uniform_routing():
    n, num_experts = 8, 4
    probs = np.full((n, num_experts), 1 / num_experts)
    experts = np.arange(n).reshape(n, 1) % num_experts
    assert load_balancing_loss(probs, experts, num_experts) == pytest.approx(1.0)


def test_load_balancing_loss_is_n_when_routing_collapses_onto_one_expert():
    n, num_experts = 8, 4
    probs = np.zeros((n, num_experts))
    probs[:, 0] = 1.0
    experts = np.zeros((n, 1), dtype=int)
    assert load_balancing_loss(probs, experts, num_experts) == pytest.approx(num_experts)


def test_load_balancing_loss_counts_every_top_k_slot():
    probs = np.full((2, 4), 0.25)
    experts = np.array([[0, 1], [2, 3]])  # all four experts used once
    assert load_balancing_loss(probs, experts, 4) == pytest.approx(1.0)
