---
name: property-tests
description: Finds properties that must hold for every input of a numerical or tokenization function in this repository (round-trips, invariants, equivariance, causality, equivalence with a reference) and tests them with Hypothesis over generated inputs. Use when hardening a tokenizer, a normalization, a masking routine or any function whose example-based tests only cover the cases the author thought of, or when the user asks for property or fuzz tests.
metadata:
  adapted-from: https://github.com/GeenccMustafa/claude-skills/tree/main/plugins/verification/skills/property-tests
---

# Property Tests with Hypothesis

## Overview

Example tests check the inputs their author imagined. Property tests state something that must be true for *every* input and let [Hypothesis](https://hypothesis.readthedocs.io/en/latest/) search for a counterexample, shrinking it to a minimal one when it finds it. ML code is full of such properties, because it is built from mathematics.

Hypothesis is a dev dependency of this repository. Its NumPy strategies live in [`hypothesis.extra.numpy`](https://hypothesis.readthedocs.io/en/latest/numpy.html).

## When to Use

- Tokenizers and anything with an encode/decode pair
- Normalizations (softmax, LayerNorm, RMSNorm) and masking
- Any function with a known symmetry (permutation, shift, scale)
- An optimized implementation that must match a simple one (vectorized vs loop, tiled vs naive attention, KV cache vs full recompute)

**When NOT to use:** training loops and anything slow per call; use a small example test instead.

## 1. Hunt for Properties

| Shape | Statement | Examples here |
|---|---|---|
| **Round-trip** | `decode(encode(x)) == x` | BPE on text over the training alphabet |
| **Invariant** | Always true of the output | Softmax is non-negative and sums to 1; LayerNorm output has mean ≈ 0 and variance ≈ 1 |
| **Invariance** | Output unchanged by a transformation of the input | Softmax is unchanged by adding a constant to every logit |
| **Equivariance** | Transforming the input transforms the output the same way | Self-attention without positions commutes with permutations |
| **Causality / locality** | Changing input `j` cannot affect output `i < j` | Decoder logits, causal attention |
| **Bounds** | Output within a known range | Attention weights in [0, 1]; cosine similarity in [-1, 1]; BPE never produces more tokens than characters + 1 per word |
| **Oracle equivalence** | Fast code matches slow, obviously-correct code | Vectorized multi-head attention vs the per-head loop |
| **Monotonicity** | More of X never gives less of Y | More BPE merges never lengthen a tokenization of training words |

Write each candidate as one sentence before writing code.

## 2. Write Good Generators

- Generate the real domain, including edge cases: length-1 sequences, all-equal values, very large and very negative logits, empty strings, repeated characters.
- Constrain only what the contract requires. For floats, exclude NaN and infinity unless the function promises to handle them, and bound the magnitude so that tolerances stay meaningful:

```python
import hypothesis.extra.numpy as hnp
from hypothesis import strategies as st

finite = st.floats(-50, 50, allow_nan=False, allow_infinity=False, width=64)
matrices = hnp.arrays(
    np.float64, hnp.array_shapes(min_dims=2, max_dims=2, max_side=8), elements=finite
)
```

- Keep shapes small (`max_side=8`). Properties that hold for 8×8 hold for 512×512; the bugs are in the logic, not the size.

## 3. Write the Tests

```python
from hypothesis import given, settings, strategies as st


@given(matrices)
def test_softmax_rows_are_probability_distributions(x):
    p = softmax(x)
    assert np.all(p >= 0)
    np.testing.assert_allclose(p.sum(axis=-1), 1.0, rtol=1e-12)


@given(matrices, finite)
def test_softmax_is_shift_invariant(x, c):
    np.testing.assert_allclose(softmax(x + c), softmax(x), atol=1e-12)
```

- One property per test, named after the statement.
- Use `np.testing.assert_allclose` with tolerances suited to float64.
- Put property tests in the lecture's `tests/` folder next to the example tests, in a file named `test_<module>_properties.py`.
- Keep the default run fast: the `conftest.py` at the repository root registers a `default` profile (few examples) and a `ci` profile (more examples), selected with `HYPOTHESIS_PROFILE`.
- When Hypothesis finds a counterexample that turns out to be a real bug, keep it as a permanent `@example(...)` on the property test.

## 4. When a Property Fails

1. Read the shrunk counterexample.
2. Decide whether the **code** or the **property** is wrong. Floating-point is the usual suspect: if the property is mathematically true but fails numerically, tighten the generator bounds or justify a tolerance, and write the reason in a comment. Never loosen a tolerance only to go green.
3. Fix the code, then add the counterexample with `@example`.

## Verification

- [ ] Each tested property is stated in one sentence in the test name or docstring
- [ ] Generators cover edge cases and are constrained only by the function's contract
- [ ] `uv run pytest` passes with the default profile, and `HYPOTHESIS_PROFILE=ci uv run pytest` passes too
- [ ] Every real counterexample found is pinned with `@example`
- [ ] `.hypothesis/` is git-ignored
