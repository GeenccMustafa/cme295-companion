import numpy as np

from l01.rnn import VanillaRNN


def test_forward_shape():
    rnn = VanillaRNN(input_dim=3, hidden_dim=5)
    assert rnn.forward(np.ones((7, 3))).shape == (7, 5)


def test_gradient_vanishes_with_distance():
    rnn = VanillaRNN(input_dim=4, hidden_dim=16, scale=0.5)
    x = np.random.default_rng(1).normal(size=(30, 4))
    norms = rnn.gradient_norms(x)
    assert np.isclose(norms[-1], 1.0)
    assert norms[0] < 1e-3 * norms[-1]
