---
name: differential-testing
description: Compares two implementations that must compute the same result by running both on the same inputs and failing on any difference beyond a stated tolerance, with the trusted implementation as the oracle. Use when porting code to another framework, optimizing or refactoring an implementation, or checking a from-scratch implementation against a library reference.
metadata:
  adapted-from: https://github.com/GeenccMustafa/claude-skills/tree/main/plugins/verification/skills/differential-test
---

# Differential Testing

## Overview

When two implementations must compute the same function, the simpler, already-tested one is the best specification available. Feed both the same inputs and compare. Every difference is a bug, a precision issue with a known cause, or a documented convention difference; nothing else is acceptable.

Typical pairs in this repository:

| Oracle | Under test |
|---|---|
| NumPy implementation in `lNN/` | PyTorch port of the same lecture |
| Naive version (full recompute, per-head loop, materialized attention matrix) | Optimized version (KV cache, batched heads, tiled online-softmax attention) |
| Library reference (`torch.nn.functional.scaled_dot_product_attention`, `torch.nn.LayerNorm`) | From-scratch implementation |

## 1. Pin Both Sides

- Share the **weights**, not just the architecture: copy the NumPy parameters into the PyTorch module (or the reverse) before comparing. Two randomly initialized models never match.
- Fix seeds and run in **float64** on CPU. Compare float32 or GPU runs separately, with looser, justified tolerances.
- Put the model in evaluation mode (`model.eval()`, dropout off) and disable anything stochastic.

## 2. Choose Inputs

1. Generated inputs, reusing the strategies written for `property-based-testing` (varied shapes, lengths, masks)
2. The lecture's own worked example (for instance the end-to-end translation example in Lecture 01)
3. Edge cases: sequence length 1, fully padded rows, very large logits

## 3. Normalize Only Known Conventions

Write the conversions explicitly, in one helper, and keep the list short. Each rule is a place a real bug could hide.

- Weight layout: PyTorch `nn.Linear` stores `(out_features, in_features)` and computes `x @ W.T`; this repository stores `(in, out)` and computes `x @ W`.
- Mask polarity: this repository uses `True = may attend`. PyTorch `scaled_dot_product_attention` boolean masks use the same polarity, but `nn.MultiheadAttention`'s `key_padding_mask` uses `True = ignore`.
- Batch dimension: add or remove it at the boundary.
- Fused vs separate Q/K/V projections: split or concatenate the weight matrices.

Check each convention against the library documentation for the pinned version (see `source-driven-development`) and cite the page in a comment.

## 4. Compare

```python
torch = pytest.importorskip("torch")


def test_attention_matches_torch_reference():
    rng = np.random.default_rng(0)
    q, k, v = (rng.normal(size=(5, 8)) for _ in range(3))
    mask = causal_mask(5)
    ours, _ = scaled_dot_product_attention(q, k, v, mask)
    ref = torch.nn.functional.scaled_dot_product_attention(
        *(torch.from_numpy(a)[None] for a in (q, k, v)), attn_mask=torch.from_numpy(mask)[None]
    )[0].numpy()
    np.testing.assert_allclose(ours, ref, rtol=1e-10, atol=1e-12)
```

- Use `pytest.importorskip("torch")` so the default NumPy-only install still passes.
- Compare intermediate tensors too (scores, weights, each layer's output), not only the final logits. When the final output differs, the first differing intermediate tells you where.
- State the tolerance and its reason. float64 on CPU should agree to roughly 1e-10; anything looser needs an explanation.

## 5. Report

For any mismatch: the smallest input that reproduces it, both outputs, the maximum absolute and relative error, and the first intermediate tensor that diverges. Fix the side under test; change the oracle only if it is shown to be wrong, with a test that proves it.

## Verification

- [ ] Both sides share weights, seeds, dtype and evaluation mode
- [ ] Convention conversions live in one helper and each is cited
- [ ] Intermediates are compared, not just final outputs
- [ ] Tolerances are stated and justified
- [ ] The test skips cleanly when the optional dependency is absent
