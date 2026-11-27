# CME295 Companion: Transformers & LLMs, Lecture by Lecture

A self-study companion for Stanford **CME295 "Transformers & Large Language Models"** (Autumn 2026, taught by Afshine and Shervine Amidi).

Each lecture gets its own folder with:

- a **lecture guide** (`README.md`): learning objectives, topics with video timestamps, key formulas, reading list
- **from-scratch code** (NumPy first, PyTorch where it helps) that implements the ideas in the lecture
- **tests** that pin the behaviour of that code: unit tests, Hypothesis property tests and doctests
- **exercises** to check your understanding

The code is written for reading. Each file is small, every tensor shape is annotated, and nothing is hidden behind a framework call until it has been built by hand once.

> This is an unofficial study resource. It is not affiliated with Stanford University or the course staff. Slides, videos and exams belong to their authors and are linked, not copied.

## Lectures

| # | Lecture | Date | Status |
|---|---|---|---|
| 01 | [Transformers](lectures/01-transformers/) | 2026-09-25 | 🟢 in progress |
| 02 | Large Language Models | 2026-10-02 | ⚪ planned |
| 03 | LLM Training | 2026-10-09 | ⚪ planned |
| 04 | Reinforcement Learning with LLMs | 2026-10-16 | ⚪ planned |
| 05 | LLM Systems | 2026-10-30 | ⚪ planned |
| 06 | AI Agents | 2026-11-06 | ⚪ planned |
| 07 | LLM Evaluation | 2026-11-13 | ⚪ planned |
| 08 | Diffusion LLMs | 2026-11-20 | ⚪ planned |
| 09 | Trending Topics | 2026-12-04 | ⚪ planned |

The full plan (topics, objectives and code deliverables for every lecture) is in [SYLLABUS.md](SYLLABUS.md).

## Getting started

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/GeenccMustafa/cme295-companion.git
cd cme295-companion
uv sync
uv run pytest
```

Run a lecture's demo:

```bash
uv run python lectures/01-transformers/demo.py
```

## Repository layout

```
cme295-companion/
├── SYLLABUS.md                  # course-wide plan
├── AGENTS.md                    # guidance for AI coding agents (CLAUDE.md imports it)
├── REVIEW.md                    # what a bad change looks like here
├── .claude/skills/              # agent workflows: new lecture, TDD, property tests, review, ...
├── lectures/
│   └── 01-transformers/
│       ├── README.md            # lecture guide
│       ├── EXERCISES.md         # practice problems
│       ├── demo.py              # runnable walk-through
│       ├── l01/                 # importable package with the implementations
│       └── tests/               # pytest suite for this lecture
└── pyproject.toml
```

Each lecture's package has a unique name (`l01`, `l02`, ...) so that all lectures can be tested in one run.

## Working with AI coding agents

[AGENTS.md](AGENTS.md) describes the conventions, and [.claude/skills/](.claude/skills/) holds the workflows agents follow here, such as building a new lecture, test-driven development for numerical code, property tests and pre-PR checks. Some skills are adapted from [addyosmani/agent-skills](https://github.com/addyosmani/agent-skills) and [GeenccMustafa/claude-skills](https://github.com/GeenccMustafa/claude-skills); see [.claude/skills/README.md](.claude/skills/README.md).

## Official course resources

- Course site: <https://cme295.stanford.edu/>
- Syllabus: <https://cme295.stanford.edu/syllabus/>
- Lecture videos: YouTube playlist linked from the course site

## License

Code is released under the [MIT License](LICENSE). Course materials referenced here remain the property of their authors.
