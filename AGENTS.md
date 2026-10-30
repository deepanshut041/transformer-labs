# Transformer Labs repository ownership

This collection is the independent `transformer-labs` Git repository, checked out at `ai_projects/projects/transformer-labs/` as a submodule of `ai_projects`. Its project folders are ordinary directories tracked by this collection. Shared tutorials and learning checkpoints remain in the umbrella Docs submodule. Local bare origin: workspace `.ai/repositories/transformer-labs.git`.

Commit project changes in this collection first, then commit the updated collection pointer in `ai_projects`. Local commits are not automatically copied to the bare origin; run `git push origin main` in this collection before another checkout needs the new commit. No external hosting is configured.

Read the project README and project-local AGENTS.md before editing. Preserve existing source, local changes, private study notes, data, models, environments and run evidence. Do not create nested Git repositories or submodules for individual projects.

When checked out inside the workspace, follow `../../AGENTS.md` and shared `../../docs/guide/{roadmap,progress,resume,decisions}.md`. Run relevant project checks from each project folder. Shared CLI/TUI code belongs to workspace `cli/`. Keep ignored credentials, environments, datasets, weights, generated output and private study records out of commits. Software checks do not advance learner or hosted gates.

This ownership change on 2026-10-30 supersedes older instructions describing all collection source as directly tracked by ai_projects. Project learning checkpoints are unchanged.
