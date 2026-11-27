import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from l02.decoding import (
    beam_search,
    greedy_search,
    sample_next_token,
    temperature_softmax,
    top_k_filter,
    top_p_filter,
)

LOGITS = np.array([3.0, 2.0, 1.0, 0.0])


@pytest.mark.parametrize(
    ("temperature", "expected"),
    [
        (0.5, [0.865, 0.117, 0.016, 0.002]),
        (1.0, [0.644, 0.237, 0.087, 0.032]),
        (2.0, [0.455, 0.276, 0.167, 0.102]),
    ],
)
def test_temperature_matches_worked_example(temperature, expected):
    np.testing.assert_allclose(temperature_softmax(LOGITS, temperature), expected, atol=1e-3)


def test_temperature_must_be_positive():
    with pytest.raises(ValueError):
        temperature_softmax(LOGITS, 0.0)


def test_top_k_keeps_the_k_most_likely_tokens_and_renormalizes():
    probs = np.array([0.50, 0.25, 0.15, 0.07, 0.03])
    np.testing.assert_allclose(top_k_filter(probs, 2), [2 / 3, 1 / 3, 0, 0, 0])


def test_top_p_keeps_the_smallest_set_reaching_p():
    probs = np.array([0.50, 0.25, 0.15, 0.07, 0.03])
    np.testing.assert_allclose(top_p_filter(probs, 0.85), [0.5, 0.25, 0.15, 0, 0] / np.float64(0.9))


def test_top_p_matches_midterm_question():
    # Midterm II.5: soft .60, brown .25, small .10, worn .05 with p = 0.8 keeps soft and brown.
    kept = top_p_filter(np.array([0.60, 0.25, 0.10, 0.05]), 0.8) > 0
    assert kept.tolist() == [True, True, False, False]


def test_top_p_adapts_to_the_shape_of_the_distribution():
    peaked = top_p_filter(np.array([0.92, 0.05, 0.02, 0.01]), 0.85)
    flat = top_p_filter(np.full(10, 0.1), 0.85)
    assert np.count_nonzero(peaked) == 1
    assert np.count_nonzero(flat) == 9


def test_zero_temperature_sampling_is_greedy():
    rng = np.random.default_rng(0)
    assert {sample_next_token(LOGITS, rng, temperature=0.0) for _ in range(20)} == {0}


def test_sampling_frequencies_match_the_probabilities():
    rng = np.random.default_rng(0)
    draws = [sample_next_token(LOGITS, rng) for _ in range(20_000)]
    freqs = np.bincount(draws, minlength=4) / len(draws)
    np.testing.assert_allclose(freqs, temperature_softmax(LOGITS, 1.0), atol=0.01)


def test_sampling_never_returns_a_filtered_token():
    rng = np.random.default_rng(0)
    draws = {sample_next_token(LOGITS, rng, top_k=2) for _ in range(500)}
    assert draws == {0, 1}


# Midterm II.9: soft (a) .6, brown (b) .4; then P(u|a) = P(v|a) = .5, P(u|b) = .9, P(v|b) = .1.
BOS, A, B, U, V, EOS = range(6)
NEXT = {
    (BOS,): {A: 0.6, B: 0.4},
    (BOS, A): {U: 0.5, V: 0.5},
    (BOS, B): {U: 0.9, V: 0.1},
}


def toy_log_probs(prefix):
    probs = np.zeros(6)
    for token, p in NEXT.get(tuple(prefix), {EOS: 1.0}).items():
        probs[token] = p
    with np.errstate(divide="ignore"):
        return np.log(probs)


def test_greedy_search_follows_the_locally_best_token():
    tokens, log_prob = greedy_search(toy_log_probs, BOS, EOS, max_len=5)
    assert tokens == [BOS, A, U, EOS]
    assert np.exp(log_prob) == pytest.approx(0.30)


def test_beam_search_recovers_the_globally_better_sequence():
    tokens, log_prob = beam_search(toy_log_probs, BOS, EOS, beam_size=2, max_len=5)
    assert tokens == [BOS, B, U, EOS]
    assert np.exp(log_prob) == pytest.approx(0.36)


probability_vectors = hnp.arrays(
    np.float64, st.integers(2, 20), elements=st.floats(0.01, 1.0, width=64)
).map(lambda x: x / x.sum())


@given(probability_vectors, st.floats(0.05, 1.0))
def test_top_p_keeps_at_least_p_of_the_mass_and_renormalizes(probs, p):
    filtered = top_p_filter(probs, p)
    assert probs[filtered > 0].sum() >= p - 1e-12
    np.testing.assert_allclose(filtered.sum(), 1.0)


@given(probability_vectors, st.floats(0.05, 1.0))
def test_top_p_drops_only_tokens_less_likely_than_every_kept_token(probs, p):
    filtered = top_p_filter(probs, p)
    kept, dropped = probs[filtered > 0], probs[filtered == 0]
    if dropped.size:
        assert dropped.max() <= kept.min()


@given(hnp.arrays(np.float64, 6, elements=st.floats(-5, 5, width=64)), st.floats(0.1, 5.0))
def test_lower_temperature_never_flattens_the_distribution(logits, temperature):
    sharper = temperature_softmax(logits, temperature / 2).max()
    flatter = temperature_softmax(logits, temperature).max()
    assert sharper >= flatter - 1e-12
