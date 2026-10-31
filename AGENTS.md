# Transformer Labs repository ownership

This collection is the independent `transformer-labs` Git repository, checked out at `ai_projects/projects/transformer-labs/` as a submodule of `ai_projects`. Its project folders are ordinary directories tracked by this collection. Shared tutorials and learning checkpoints remain in the umbrella Docs submodule. Local bare origin: workspace `.ai/repositories/transformer-labs.git`.

Commit project changes in this collection first, then commit the updated collection pointer in `ai_projects`. Local commits are not automatically copied to the bare origin; run `git push origin main` in this collection before another checkout needs the new commit. No external hosting is configured.

Read the project README and project-local AGENTS.md before editing. Preserve existing source, local changes, private study notes, data, models, environments and run evidence. Do not create nested Git repositories or submodules for individual projects.

When checked out inside the workspace, follow `../../AGENTS.md` and shared `../../docs/guide/{roadmap,progress,resume,decisions}.md`. Run relevant project checks from each project folder. Shared CLI/TUI code belongs to workspace `cli/`. Keep ignored credentials, environments, datasets, weights, generated output and private study records out of commits. Software checks do not advance learner or hosted gates.

This ownership change on 2026-10-30 supersedes older instructions describing all collection source as directly tracked by ai_projects. Project learning checkpoints are unchanged.


## Standalone execution contract

Every runnable project must own its dependency manifest/lock, configuration and documented native entry points. From a standalone collection checkout, setup, run, test and build must work from the project folder without the workspace CLI, `AI_WORKSPACE`, an injected `PYTHONPATH` or sibling repositories. Shared workspace documentation links are optional context outside the workspace, not runtime requirements.

The workspace CLI/TUI is an optional adapter to those same entry points. Keep model logic in the project; do not import `ai_cli` or duplicate the launcher inside collections. Document both native and workspace commands, and verify a clean project copy outside the workspace when changing execution or packaging. Use the project's actual test framework; Python projects can declare `[tool.ai-workspace] test-args` for the optional adapter. Documentation-only projects remain planned until their boilerplate is explicitly prepared; require both launch paths before calling a new scaffold ready.
