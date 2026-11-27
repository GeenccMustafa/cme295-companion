---
name: code-review-and-quality
description: Reviews changes to this repository along five axes adapted to teaching ML code (mathematical correctness, numerical robustness, clarity for learners, tests, and documentation), and classifies every finding by severity with file and line evidence. Use before merging any change, when asked to review a diff or pull request, or when checking work written by yourself, another agent or a contributor.
metadata:
  adapted-from: https://github.com/addyosmani/agent-skills/tree/main/skills/code-review-and-quality
---

# Code Review for a Teaching Codebase

## Overview

A change is ready when it is correct, robust, easy for a learner to read, tested, and documented. The repository-specific failure modes and output format are in [REVIEW.md](../../../REVIEW.md); this skill is the process.

## When to Use

- Before merging any change to `main`
- When asked to review a diff or pull request
- After finishing a lecture or a module, as a self-review

## The Five Axes

### 1. Mathematical correctness

- Does the code compute what the docstring's `Notes` and the cited paper say? Check equations term by term: scaling factors, signs, which axis a softmax or norm is taken over, where masks are applied, pre-norm vs post-norm.
- Are shapes right in every branch, including self- vs cross-attention and sequence length 1?
- Are hand-derived gradients checked by a finite-difference test?

### 2. Numerical robustness

- Softmax and log-softmax subtract the maximum; logs never see 0 without a guard.
- Masked rows that are fully blocked do not produce NaN.
- Randomness goes through an explicit `np.random.Generator` with a seed.
- Dtypes are deliberate; no silent integer division or float32/float64 mixing.

### 3. Clarity for learners

- Could a student who just watched the lecture follow the code next to the slides? Names match the lecture's notation (Q, K, V, d_k, h).
- Clarity beats cleverness. A loop over heads is acceptable if it makes the idea visible; an opaque `einsum` needs a comment with the shapes.
- No dead code, no commented-out experiments, no speculative generality (options nobody uses yet).
- Functions do one thing. Files stay small enough to read in one sitting.

### 4. Tests

- Review the tests first: do they pin the mathematics (invariants, known values, structural properties) or only the shapes? See `test-driven-development` and `property-tests`.
- Would the tests fail if the bug you are worried about existed? If unsure, try it.
- Are tests deterministic and fast?

### 5. Documentation

- NumPy-style docstrings with shapes, Notes, References and Examples (see `numpy-docstrings`).
- Lecture README, `SYLLABUS.md` and the top-level README agree with the code.
- Claims and numbers are sourced (see `source-driven-development`).

## Process

1. **Understand the intent.** Read the PR description or the request, and the lecture section the change implements.
2. **Read the tests.**
3. **Read the implementation** against the five axes.
4. **Run the checks** from `pre-pr-check`. A review that did not run the tests says so.
5. **Classify findings**:
   - **Important**: wrong results, NaN or crash on valid input, a test that cannot fail, documentation that states something false.
   - **Nit**: naming, wording, small clarity improvements. At most three per review.
   - **Pre-existing**: a real problem that this change did not introduce.
6. **Write the review** starting with the count, for example "1 Important, 2 Nits, 0 Pre-existing", then each finding with `file:line`, the failing scenario and a suggested fix. If nothing needs fixing, say so in one line.

## Change Size

Prefer one lecture module per change. A whole new lecture can land as one PR if it is reviewed module by module. Split pure refactors from behaviour changes.

## Honesty

Do not approve what you have not read or run. Do not soften an Important finding into a Nit. If a finding is uncertain, say what would confirm it.

## Verification

- [ ] All five axes were considered
- [ ] Tests were read before the implementation
- [ ] The checks were run, or the review says they were not
- [ ] Every finding has `file:line`, a scenario and a severity
- [ ] The summary starts with the finding count
