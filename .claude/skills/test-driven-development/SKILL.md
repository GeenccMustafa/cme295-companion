---
name: test-driven-development
description: Drives implementation of numerical and ML code in this repository with the red-green-refactor loop, using shape checks, mathematical invariants, known values and gradient checks as tests. Use when implementing any new module in a lecture package (l01, l02, ...), fixing a bug in one, or changing the behaviour of existing code, even when the change looks obvious from reading it.
metadata:
  adapted-from: https://github.com/addyosmani/agent-skills/tree/main/skills/test-driven-development
---

# Test-Driven Development for Numerical Code

## Overview

Write a failing test before the code that makes it pass. For bug fixes, reproduce the bug with a test first. In numerical code, "it runs and the shapes look right" is not evidence: a transposed matrix or a missing mask produces output of the correct shape and plausible-looking numbers. Tests must pin down the *mathematics*, not just the plumbing.

## When to Use

- Implementing a new function or class in `lectures/NN-*/lNN/`
- Fixing a bug (write the reproduction test first)
- Changing the behaviour of existing code (new argument, new convention)
- Porting a NumPy implementation to PyTorch (the NumPy version becomes the oracle)

**When NOT to use:** README or EXERCISES edits, docstring-only changes (those are covered by doctests and ruff), CI configuration.

## The Repository's Commands

```bash
uv run pytest                                   # full suite, including doctests
uv run pytest lectures/01-transformers          # one lecture
uv run pytest lectures/01-transformers/tests/test_attention.py -k causal   # one test
uv run ruff check . && uv run ruff format --check .
```

`--doctest-modules` is on by default, so every `Examples` section in a docstring is a test too. Run the focused test during the loop and the full suite before finishing.

## The Cycle

```
  RED                     GREEN                    REFACTOR
  test fails for the  ──► minimal code that   ──►  clean up, add docstring,
  right reason            passes                   tests still pass
```

1. **RED.** Write the test in `lectures/NN-*/tests/test_<module>.py`. Run it and confirm it fails *because the behaviour is missing*, not because of an import error or typo.
2. **GREEN.** Write the simplest implementation that passes. Prefer explicit loops over clever broadcasting at first if it is clearer.
3. **REFACTOR.** Vectorize, rename, add the NumPy-style docstring (see the `numpy-docstrings` skill). Run the tests after every step.

## What to Test in Numerical Code

Use several of these for every non-trivial function. One kind of test alone is rarely enough.

| Kind | What it catches | Example from Lecture 01 |
|---|---|---|
| **Shape** | Wrong axis, missing transpose, broadcasting accidents | `MultiHeadAttention` returns `(n_q, d_model)` and weights `(h, n_q, n_k)` |
| **Invariant** | Violated mathematical properties | Softmax rows sum to 1; LayerNorm output has mean 0 |
| **Structural property** | Wrong masking or wiring | Changing a later target token leaves earlier decoder logits unchanged (causality); permuting inputs permutes self-attention outputs |
| **Known value** | Wrong formula or constant | `PE[0] == [0, 1, 0, 1]`; uniform logits give loss `log V`; BPE first merge on the lecture's toy corpus |
| **Limit / edge case** | Numerical instability | `softmax([1000, 1000])` is finite; a fully masked row gives zeros, not NaN |
| **Statistical** | Wrong scaling | Raw `q · k` has variance ≈ `d_k`; after dividing by `√d_k` it is ≈ 1 |
| **Gradient check** | Wrong hand-derived gradients | Compare analytic gradients with central finite differences |
| **Learning sanity** | Broken training loop | Loss decreases on a tiny dataset; the model can overfit a handful of examples |
| **Reference parity** | Divergence from a trusted implementation | NumPy attention matches `torch.nn.functional.scaled_dot_product_attention` (only when torch is installed; mark with `pytest.importorskip("torch")`) |

### Gradient checking

Any hand-written backward pass needs a finite-difference test:

```python
def numerical_grad(f, x, eps=1e-6):
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        old = x[idx]
        x[idx] = old + eps
        f_plus = f()
        x[idx] = old - eps
        f_minus = f()
        x[idx] = old
        grad[idx] = (f_plus - f_minus) / (2 * eps)
    return grad


np.testing.assert_allclose(analytic, numerical_grad(loss_fn, W), rtol=1e-4, atol=1e-6)
```

Use float64 for gradient checks; float32 is too noisy for `eps=1e-6`.

## Rules for Reliable Numerical Tests

- **Seed everything.** Use `np.random.default_rng(seed)`, never the global `np.random` state. Tests must give the same result on every run and in any order.
- **Compare with tolerances.** Use `np.allclose` / `np.testing.assert_allclose`, never `==` on floats. Pick tolerances on purpose and keep them tight; a loose `atol=1e-1` hides bugs.
- **Statistical tests need margins and enough samples.** Assert `0.8 < var < 1.2` with thousands of samples, not `var == 1`.
- **Keep tests small and fast.** Tiny dimensions (`d_model=16`, a handful of tokens) exercise the same code paths as large ones. The whole suite should run in seconds.
- **Name tests after the behaviour.** `test_decoder_is_causal`, not `test_decoder_2`.
- **One concept per test.** A shape test and a causality test are two tests.
- **Test through the public API.** Do not assert on private helpers such as `_split_heads`; test what the module promises.
- **Doctests are documentation first.** Keep them short and deterministic. Print shapes, `.tolist()` of small arrays, or `bool(...)` of a comparison, so the output does not depend on NumPy's repr of scalars.

## Common Rationalizations

| Rationalization | Reality |
|---|---|
| "The output shape is right, so it works" | A transposed weight or a missing `√d_k` gives the right shape. Test invariants and known values. |
| "I derived the gradient carefully" | Hand-derived gradients are the most common source of silent bugs. Gradient-check them. |
| "The loss went down, so training is correct" | A loss can fall with a broken mask (the model peeks at the answer). Test causality directly. |
| "Random weights make outputs untestable" | Properties still hold with random weights: causality, equivariance, normalization, shapes. |
| "I'll add tests after the lecture is done" | Then they will test what the code does, not what the lecture says it should do. |

## Red Flags

- A new function in `lNN/` with no test in `tests/`
- A test that passed on its very first run (check that it can fail: break the code briefly)
- `np.random.seed` or unseeded randomness in a test
- Float equality with `==`
- Tolerances loosened until a failing test passes
- Hand-written gradients without a finite-difference check
- Skipped or `xfail` tests added to make the suite green

## Verification

- [ ] Every new behaviour has a test, and each test was seen failing before the fix or implementation
- [ ] Shape, invariant and at least one known-value or structural test exist for each new module
- [ ] Hand-written gradients are checked against finite differences
- [ ] `uv run pytest` passes (unit tests and doctests)
- [ ] `uv run ruff check .` and `uv run ruff format --check .` pass
- [ ] No test is skipped, disabled or made non-deterministic
