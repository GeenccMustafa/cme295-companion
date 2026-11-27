---
name: git-workflow-and-versioning
description: Structures commits, branches, pull requests and releases: small atomic commits, messages that explain why, short-lived branches and tagged releases. Use when committing, writing a commit message, splitting work into commits, opening a pull request, or tagging a release.
metadata:
  adapted-from: https://github.com/addyosmani/agent-skills/tree/main/skills/git-workflow-and-versioning
---

# Git Workflow and Versioning

## Overview

History in this repository should read like the course: one idea at a time, each commit leaving `main` green.

## When to Use

- Any commit, branch, pull request or release
- Splitting a large working tree into commits

## Commits

- **Atomic.** One logical change per commit: a module with its tests and docstrings; a README section; a CI change. Do not mix a refactor with a behaviour change.
- **Green.** Every commit passes `uv run ruff check .`, `uv run ruff format --check .` and `uv run pytest`.
- **Message format.** English, imperative subject of at most 72 characters, no trailing period, then a blank line and a body that explains *why* and lists notable details:

  ```
  Add multi-head attention to Lecture 01

  - Split d_model into h heads of size d_k and concatenate the outputs
  - Test shapes, parameter count and permutation equivariance
  - Document the per-head projections as column blocks of W_Q, W_K, W_V

  Verified-by: format, lint, pytest, demos
  ```

- Prefix the subject with the lecture when it helps scanning: `L02: Add RoPE`.
- End with the `Verified-by:` trailer from the `pre-merge-verification` skill.
- Commits are authored by the repository owner. Do not add AI co-author trailers or "generated with" lines.
- Never commit secrets, `.env` files, `.claude/settings.local.json`, virtual environments, course PDFs or video transcripts (course materials are linked, not copied).

## Branches and Pull Requests

- `main` is always green and is what readers clone.
- Work on short-lived branches named `lecture-NN/<topic>` (for example `lecture-02/rope`) or `fix/<topic>`, `docs/<topic>`, `ci/<topic>`.
- A pull request description has: **Summary** (what and why), **Testing** (the checks that ran and their results), and **Notes for reviewers** (anything non-obvious, deviations from the paper).
- Merge only with CI green.

## Releases

- Tag a release when a lecture is complete (guide, code, tests, exercises, SYLLABUS boxes ticked): `lecture-01`, `lecture-02`, ...
- Use semantic versions for the package (`0.N.0` after lecture N) and record what each lecture added in the GitHub release notes.

## Before Committing

```bash
git status
git diff --staged
```

Check that only intended files are staged, nothing generated or secret is included, and the message describes the staged diff rather than the whole session.

## Verification

- [ ] Each commit is one logical change and passes the checks
- [ ] Messages are in English, imperative, with a body explaining why
- [ ] No secrets, local settings or course materials are committed
- [ ] The `Verified-by:` trailer lists only checks that actually ran
