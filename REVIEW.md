# REVIEW.md

What a bad change looks like in this repository. The process is in the [`code-review-and-quality`](.claude/skills/code-review-and-quality/SKILL.md) skill.

## What failure looks like here

- `lectures/*/lNN/`: **it computes the wrong thing while looking right.** A missing √d_k, a softmax over the wrong axis, a mask applied after the softmax, a transposed weight. Shapes match, numbers look plausible, the lesson is wrong.
- `lectures/*/lNN/`: **it breaks on valid input.** NaN from a fully masked row, overflow in `exp`, failure at sequence length 1.
- `lectures/*/tests/`: **the test cannot fail.** It checks only shapes, uses loose tolerances, or would still pass with the bug it is meant to catch.
- Docstrings, `README.md`, `SYLLABUS.md`: **they lie.** A formula, shape, default, citation or checkbox that does not match the code or the source.
- Anywhere: **course material is copied in.** Slides, transcripts and exam text are linked, never committed.

## Focus on

- Equations in code against the `Notes` section and the cited paper, term by term
- Masking: polarity (`True` means "may attend"), where it is applied, fully masked rows
- Numerical stability of softmax, log-softmax, normalizations and losses
- Hand-written gradients have a finite-difference test
- Every random draw goes through a seeded `np.random.Generator`
- New public functions have NumPy-style docstrings with shapes and a doctest

## Don't flag

- Anything `ruff format` and `ruff check` already enforce
- Explicit loops chosen for readability over vectorization, unless the README claims performance
- `uv.lock`

## Evidence

- Every finding cites `file:line` and describes the input that produces the wrong result.
- Do not infer behaviour from a name; read the code it calls.

## Nits

At most three per review, and none on a review that has Important findings.

## Output

- Start with a count: "1 Important, 2 Nits, 0 Pre-existing."
- Tag each finding Important, Nit or Pre-existing (a real bug this change did not introduce).
- If there is nothing to fix, say so in one line.
