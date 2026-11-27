# Agent skills

Workflows that AI coding agents load when working in this repository. Claude Code discovers them automatically from this folder; other agents can read them through [AGENTS.md](../../AGENTS.md).

| Skill | Origin |
|---|---|
| [lecture-development](lecture-development/SKILL.md) | Written for this repository, combining ideas from `planning-and-task-breakdown` and `incremental-implementation` in [addyosmani/agent-skills](https://github.com/addyosmani/agent-skills) |
| [documentation](documentation/SKILL.md) | Adapted from `documentation-and-adrs` in [addyosmani/agent-skills](https://github.com/addyosmani/agent-skills/tree/main/skills/documentation-and-adrs) |
| [test-driven-development](test-driven-development/SKILL.md) | Adapted from [addyosmani/agent-skills](https://github.com/addyosmani/agent-skills/tree/main/skills/test-driven-development) |
| [source-driven-development](source-driven-development/SKILL.md) | Adapted from [addyosmani/agent-skills](https://github.com/addyosmani/agent-skills/tree/main/skills/source-driven-development) |
| [code-review-and-quality](code-review-and-quality/SKILL.md) | Adapted from [addyosmani/agent-skills](https://github.com/addyosmani/agent-skills/tree/main/skills/code-review-and-quality) |
| [git-workflow-and-versioning](git-workflow-and-versioning/SKILL.md) | Adapted from [addyosmani/agent-skills](https://github.com/addyosmani/agent-skills/tree/main/skills/git-workflow-and-versioning) |
| [property-based-testing](property-based-testing/SKILL.md) | Adapted from `property-tests` in [GeenccMustafa/claude-skills](https://github.com/GeenccMustafa/claude-skills/tree/main/plugins/verification/skills/property-tests) |
| [differential-testing](differential-testing/SKILL.md) | Adapted from `differential-test` in [GeenccMustafa/claude-skills](https://github.com/GeenccMustafa/claude-skills/tree/main/plugins/verification/skills/differential-test) |
| [pre-merge-verification](pre-merge-verification/SKILL.md) | Adapted from `pre-pr-check` in [GeenccMustafa/claude-skills](https://github.com/GeenccMustafa/claude-skills/tree/main/plugins/verification/skills/pre-pr-check) |

The adaptations replace generic web and JavaScript guidance with Python, NumPy, pytest, Hypothesis and uv, and add checks specific to numerical and machine learning code.

## Naming

- A skill's name describes the activity, not a tool or a single step: `documentation`, not `numpy-docstrings`; `pre-merge-verification`, not `pre-pr-check`.
- Names are lowercase and hyphenated, and match the folder name.
- The description starts with what the skill does in general terms, followed by one or more "Use when" triggers. Tools and repository specifics belong in the body.

## Licenses

Skills adapted from [addyosmani/agent-skills](https://github.com/addyosmani/agent-skills) are used under the MIT License, Copyright (c) 2025 Addy Osmani; see [LICENSE-agent-skills](LICENSE-agent-skills). Skills from [GeenccMustafa/claude-skills](https://github.com/GeenccMustafa/claude-skills) are by the same author as this repository, also under the MIT License.
