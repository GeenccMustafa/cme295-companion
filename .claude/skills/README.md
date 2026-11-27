# Agent skills

Workflows that AI coding agents load when working in this repository. Claude Code discovers them automatically from this folder; other agents can read them through [AGENTS.md](../../AGENTS.md).

| Skill | Origin |
|---|---|
| [new-lecture](new-lecture/SKILL.md) | Written for this repository, combining ideas from `planning-and-task-breakdown` and `incremental-implementation` in [addyosmani/agent-skills](https://github.com/addyosmani/agent-skills) |
| [numpy-docstrings](numpy-docstrings/SKILL.md) | Adapted from `documentation-and-adrs` in [addyosmani/agent-skills](https://github.com/addyosmani/agent-skills/tree/main/skills/documentation-and-adrs) |
| [test-driven-development](test-driven-development/SKILL.md) | Adapted from [addyosmani/agent-skills](https://github.com/addyosmani/agent-skills/tree/main/skills/test-driven-development) |
| [source-driven-development](source-driven-development/SKILL.md) | Adapted from [addyosmani/agent-skills](https://github.com/addyosmani/agent-skills/tree/main/skills/source-driven-development) |
| [code-review-and-quality](code-review-and-quality/SKILL.md) | Adapted from [addyosmani/agent-skills](https://github.com/addyosmani/agent-skills/tree/main/skills/code-review-and-quality) |
| [git-workflow-and-versioning](git-workflow-and-versioning/SKILL.md) | Adapted from [addyosmani/agent-skills](https://github.com/addyosmani/agent-skills/tree/main/skills/git-workflow-and-versioning) |
| [property-tests](property-tests/SKILL.md) | Adapted from [GeenccMustafa/claude-skills](https://github.com/GeenccMustafa/claude-skills/tree/main/plugins/verification/skills/property-tests) |
| [differential-test](differential-test/SKILL.md) | Adapted from [GeenccMustafa/claude-skills](https://github.com/GeenccMustafa/claude-skills/tree/main/plugins/verification/skills/differential-test) |
| [pre-pr-check](pre-pr-check/SKILL.md) | Adapted from [GeenccMustafa/claude-skills](https://github.com/GeenccMustafa/claude-skills/tree/main/plugins/verification/skills/pre-pr-check) |

The adaptations replace generic web and JavaScript guidance with Python, NumPy, pytest, Hypothesis and uv, and add checks specific to numerical and machine learning code.

## Licenses

Skills adapted from [addyosmani/agent-skills](https://github.com/addyosmani/agent-skills) are used under the MIT License, Copyright (c) 2025 Addy Osmani; see [LICENSE-agent-skills](LICENSE-agent-skills). Skills from [GeenccMustafa/claude-skills](https://github.com/GeenccMustafa/claude-skills) are by the same author as this repository, also under the MIT License.
