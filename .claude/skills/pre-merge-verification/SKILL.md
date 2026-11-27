---
name: pre-merge-verification
description: Runs a project's local checks from cheapest to most expensive (formatting, linting, tests, runnable examples), fixes failures for at most three rounds before escalating, and records in the commit what was verified. Use when asked to check a change before committing, pushing or opening a pull request ("is this ready", "make sure it's green"), not on every small edit.
metadata:
  adapted-from: https://github.com/GeenccMustafa/claude-skills/tree/main/plugins/verification/skills/pre-pr-check
---

# Pre-Merge Verification

## Overview

CI runs the same checks as below. Running them locally first keeps red builds out of `main` and leaves a record of how each change was verified.

## 1. Run the Checks, Cheapest First

These match [.github/workflows/ci.yml](../../../.github/workflows/ci.yml):

| # | Check | Command |
|---|---|---|
| 1 | Format | `uv run ruff format --check .` |
| 2 | Lint (includes numpy-style docstring rules) | `uv run ruff check .` |
| 3 | Unit tests, doctests and property tests | `uv run pytest` |
| 4 | Thorough property tests | `HYPOTHESIS_PROFILE=ci uv run pytest` |
| 5 | Every lecture demo runs | `for d in lectures/*/demo.py; do uv run python "$d" > /dev/null || exit 1; done` |

Optional, when PyTorch code changed: `uv sync --extra torch && uv run pytest` so that the differential tests are not skipped.

## 2. Fix, at Most Three Rounds

- Read the actual failure and fix its cause. `uv run ruff format .` and `uv run ruff check --fix .` are fine for mechanical issues; read what they changed.
- Never weaken a check: no loosened tolerances, no `# noqa` without a reason, no skipped tests, no lint rules turned off to go green.
- If the same check still fails after three rounds, stop and report what fails, what was tried and the best hypothesis.
- No edits after the last green run: the commit that is pushed must be the commit that was checked.

## 3. Review, When It Is Worth It

For a new lecture, a new module, or a change to shared conventions, run `/code-review` on the diff and check it against [REVIEW.md](../../../REVIEW.md). Fix Important findings. Skip this for typo and wording fixes.

## 4. Record the Verification

End the commit message with a trailer listing only what actually ran and passed:

```
Verified-by: format, lint, pytest, hypothesis-ci, demos
```

For changes with nothing to run, say so with a reason:

```
Verified-by: none (README wording only)
```

Find unverified commits later with `git log --grep "Verified-by: none"`.

## 5. Report

A short table: check, PASS / FAIL / SKIPPED (with reason), and any notable output. Commit or push only if the user asked for it.
