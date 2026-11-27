# AGENTS.md

Guidance for AI coding agents (Claude Code, Codex, Cursor and others) working in this repository.

## What this is

A lecture-by-lecture study companion for Stanford CME295, "Transformers & Large Language Models". Each lecture has a guide, from-scratch NumPy implementations, tests and exercises. The code is written to be read: clarity for a learner comes before performance.

## Goals and non-goals

- **Goals:** correct implementations that match the papers, explicit tensor shapes, documentation that teaches, tests that pin the mathematics.
- **Non-goals:** a production library, GPU performance, training large models, copying course materials (slides, transcripts and exams are linked, never committed).

## Commands

```bash
uv sync                                   # install (NumPy, pytest, ruff, hypothesis)
uv sync --extra torch                     # also install PyTorch for differential tests
uv run pytest                             # unit tests, property tests and doctests
uv run ruff format . && uv run ruff check .
uv run python lectures/01-transformers/demo.py
```

## Layout

- `SYLLABUS.md`: the plan for every lecture, with checkboxes for what is implemented
- `lectures/NN-name/`: `README.md` (guide), `EXERCISES.md`, `demo.py`, `lNN/` (package), `tests/`
- `REVIEW.md`: what a bad change looks like here
- `.claude/skills/`: workflows for agents (see below)

## Conventions

- NumPy-style docstrings on every public object: shapes for every array, math in `Notes`, sources in `References`, runnable `Examples`.
- One concept per module; one module per commit, with its tests.
- Seed all randomness with `np.random.default_rng(seed)`.
- Everything in the repository (code, docs, commits, issues) is written in English.
- Commit messages: imperative subject, body explaining why, `Verified-by:` trailer. No AI co-author trailers.

## Skills

| Skill | Use it to |
|---|---|
| [lecture-development](.claude/skills/lecture-development/SKILL.md) | Plan and build a lecture: guide, code, tests, demo and exercises |
| [test-driven-development](.claude/skills/test-driven-development/SKILL.md) | Write a failing test first, then the code that passes it |
| [property-based-testing](.claude/skills/property-based-testing/SKILL.md) | Test properties that must hold for every input |
| [differential-testing](.claude/skills/differential-testing/SKILL.md) | Check that two implementations of the same computation agree |
| [source-driven-development](.claude/skills/source-driven-development/SKILL.md) | Ground formulas, numbers and API usage in primary sources, and cite them |
| [documentation](.claude/skills/documentation/SKILL.md) | Write docstrings and guides, and catch documentation drift |
| [code-review-and-quality](.claude/skills/code-review-and-quality/SKILL.md) | Review a change across five axes, using REVIEW.md |
| [pre-merge-verification](.claude/skills/pre-merge-verification/SKILL.md) | Run the local checks before a commit, push or pull request |
| [git-workflow-and-versioning](.claude/skills/git-workflow-and-versioning/SKILL.md) | Commit, branch, open pull requests and tag releases |
