---
name: lecture-development
description: Plans and builds a complete lecture unit (guide, implementation, tests, demo and exercises) from its syllabus entry, one module at a time, keeping the test suite green after every step. Use when starting a new lecture, scaffolding a lecture folder, or resuming an unfinished one.
metadata:
  adapted-from: https://github.com/addyosmani/agent-skills/tree/main/skills/planning-and-task-breakdown and https://github.com/addyosmani/agent-skills/tree/main/skills/incremental-implementation
---

# Lecture Development

## Overview

A lecture is done when its guide, code, tests and exercises are all in place and consistent with `SYLLABUS.md`. Build it in thin slices: each module lands with its tests and docstrings before the next one starts, and the suite stays green throughout.

[`lectures/01-transformers/`](../../../lectures/01-transformers/) is the reference layout.

## When to Use

- Starting the next lecture
- Resuming a lecture that is partly done (start from step 1 and skip what exists)

## 1. Gather the Sources

- The lecture's entry in [SYLLABUS.md](../../../SYLLABUS.md): topics, objectives and planned code.
- The lecture slides (PDF link in the course site) and the video chapter list with timestamps.
- The primary papers for every method the lecture covers (see `source-driven-development`).

Course slides, transcripts and videos are **linked, never copied** into the repository.

## 2. Plan the Modules

Turn the SYLLABUS "Code (planned)" line into an ordered list of modules. For each, write one line: file name, what it implements, the properties its tests will check, the primary source. Order by dependency: building blocks first (normalizations, positional encodings), then the components that use them (attention variants, layers), then the assembled model and the demo.

Keep each module small: one concept, roughly 50–200 lines including docstrings. Prefer NumPy. Use PyTorch only where training or a reference comparison needs it, as an optional extra.

Show the plan to the user before writing code when the lecture's scope is ambiguous.

## 3. Scaffold

```
lectures/NN-name/
├── README.md          # lecture guide (structure in the documentation skill)
├── EXERCISES.md       # Concepts + Coding
├── demo.py            # documented functions + main() guard
├── lNN/
│   └── __init__.py    # package docstring listing the modules
└── tests/             # test_<module>.py and test_<module>_properties.py
```

- The folder name is the two-digit number and a short slug (`02-large-language-models`).
- The package is `lNN` so that every lecture can be imported in one pytest run.
- The root `conftest.py` puts every lecture folder on `sys.path`, so tests and doctests need no per-lecture setup.
- Within a lecture, import sibling modules relatively (`from .attention import ...`). Importing from an earlier lecture's package is encouraged when building on it (`from l01.attention import softmax`), because it shows how the ideas stack up. `demo.py` must then add that lecture's folder to `sys.path` too, and the README says which earlier modules are reused.

## 4. Build One Module at a Time

For each module, in plan order:

1. Write the failing tests (`test-driven-development`). Add property tests where a property exists (`property-based-testing`) and a comparison with a library reference when one is available (`differential-testing`).
2. Implement until they pass.
3. Write the docstrings with shapes, Notes, References and Examples (`documentation`).
4. Run `uv run ruff format . && uv run ruff check . && uv run pytest`.
5. Commit the module on its own (`git-workflow-and-versioning`).

Do not start the next module with the suite red. Do not add features the SYLLABUS does not list; note ideas as exercises instead.

## 5. Write the Guide and Demo

- `demo.py`: one documented function per stage of the lecture, printing intermediate results, called from `main()`.
- `README.md`: follow the lecture guide structure in the `documentation` skill; link every topic row to its module.
- `EXERCISES.md`: concept questions whose answers can be checked with the code, and coding tasks that extend specific files.

## 6. Update the Repository Indexes

In the same change:

- `SYLLABUS.md`: tick the implemented items, set the lecture status, keep unimplemented items unticked.
- Top-level `README.md`: link the lecture in the table and update its status.

## 7. Finish

Run the `pre-merge-verification` skill, then `code-review-and-quality` on the lecture's diff. When everything passes and the user agrees, tag `lecture-NN`.

## Verification

- [ ] Every SYLLABUS item marked done has code, tests and docstrings
- [ ] Each module was committed with its tests, and the suite was green after each
- [ ] The README links resolve and the demo runs
- [ ] `SYLLABUS.md` and the top-level README reflect the lecture's real state
- [ ] No course slides, transcripts or videos were added to the repository
