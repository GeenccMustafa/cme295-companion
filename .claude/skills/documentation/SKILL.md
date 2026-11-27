---
name: documentation
description: Writes and maintains code and learning documentation: docstrings that state shapes, math, sources and runnable examples, guide pages with a consistent structure, and checks for documentation that has drifted from the code. Use when writing or editing a module, class or function, when writing a guide, README or exercise sheet, or when a code change may have made existing documentation wrong.
metadata:
  adapted-from: https://github.com/addyosmani/agent-skills/tree/main/skills/documentation-and-adrs
---

# Documentation

## Overview

The code in this repository is read more than it is run. Docstrings are where a reader learns what a function computes, in which shapes, under which mathematical definition, and where that definition comes from. They follow the [numpydoc convention](https://numpydoc.readthedocs.io/en/latest/format.html), enforced by ruff's pydocstyle rules (`convention = "numpy"`), and their `Examples` run as doctests in `pytest`.

## When to Use

- Writing or changing any public module, class, method or function in `lectures/*/lNN/`
- Writing a lecture `README.md` or `EXERCISES.md`
- After any behaviour change, to check that docs still describe the code

## Module Docstrings

Every module starts with a docstring that teaches the idea before the code:

1. **Summary line**: one sentence, imperative or noun phrase, ending with a period.
2. **Motivation**: one or two paragraphs. What problem does this solve, and what did earlier approaches get wrong?
3. **Notes**: the mathematics, in `.. math::` blocks with LaTeX. Define every symbol.
4. **References**: numbered, with section numbers and full URLs (see `source-driven-development`).
5. **Examples**: a short doctest showing the main entry point.

Use a raw string (`r"""`) whenever the docstring contains a backslash.

## Function and Class Docstrings

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """Compute scaled dot-product attention.

    Parameters
    ----------
    Q : numpy.ndarray of shape (n_q, d_k)
        Queries, one per position that is attending.
    mask : numpy.ndarray of bool, optional
        Broadcastable to ``(n_q, n_k)``. False entries are excluded.

    Returns
    -------
    output : numpy.ndarray of shape (n_q, d_v)
        For each query, the attention-weighted average of the values.
    weights : numpy.ndarray of shape (n_q, n_k)
        Attention weights. Each row sums to 1.

    Examples
    --------
    >>> out, w = scaled_dot_product_attention(x, x, x, mask=causal_mask(4))
    >>> w[0].tolist()
    [1.0, 0.0, 0.0, 0.0]
    """
```

Rules:

- **Every array parameter and return value states its shape**, using the names the module defines (`n`, `d_model`, `h`, `d_k`, `V`). Shapes are the most useful single fact about ML code.
- Sections, in this order when present: Parameters, Returns, Raises, Attributes (classes), Notes, References, Examples.
- Document defaults as `name : type, default value`.
- `Raises` lists every exception the function raises on purpose, and the condition.
- Classes document constructor parameters in the class docstring and public attributes under `Attributes`.
- Mention deliberate deviations from the paper in `Notes`.
- Private helpers (`_name`) get a one-line docstring.
- Doctests must be deterministic and robust to NumPy's repr: show shapes, `.tolist()` of small arrays, or `bool(...)`. Seed every random generator.

## Comments

Docstrings say *what* and *why*; comments are for local *why* inside a function. Shape annotations on tensor lines (`# (h, n_q, d_k)`) are encouraged. Do not restate the code in English.

## Lecture README Structure

Each `lectures/NN-name/README.md` has, in this order:

1. Title, date, length, video link and slides link
2. A two-sentence summary and, where it helps, a small ASCII diagram of how the ideas connect
3. **Learning objectives**: numbered, each starting with a verb (explain, derive, implement, compare)
4. **Topics**: the lecture's chapters with video timestamps, each linked to the file that implements it
5. **Key formulas**
6. **Shapes** (when the lecture has a worked example)
7. **Code in this folder**: one row per file
8. **Exercises** link and **Further reading** (primary sources only)

`EXERCISES.md` has a **Concepts** part (pen and paper, checkable against the code) and a **Coding** part (each task extends a specific file).

The top-level `README.md` lecture table and `SYLLABUS.md` checkboxes must be updated in the same change as the lecture itself.

## Documentation Drift

Before finishing a change, check:

- Do the parameter names, defaults and shapes in docstrings match the signatures?
- Do the README's "Code in this folder" table and topic links point to files that exist?
- Do `SYLLABUS.md` checkboxes reflect what is actually implemented?
- Do commands in READMEs still work? Run them.

Doctests catch drift in examples automatically; nothing catches drift in prose except reading it.

## Verification

- [ ] `uv run ruff check .` passes (pydocstyle rules)
- [ ] `uv run pytest` passes (doctests run)
- [ ] Every array parameter and return value has a shape
- [ ] New modules have Notes, References and Examples sections
- [ ] README, SYLLABUS and top-level README are consistent with the code
