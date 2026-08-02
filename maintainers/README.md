# Maintainer notes

These are engineering records —
phase plans, API locks, release gates, design rationale — not the primary path
for learning BuildML. If you are trying to use the library, start with the
[README](../README.md), [Sphinx docs](../docs/index.rst), and
[guides](../guides/README.md) instead.

## How documentation is split

| Location | Audience | Published |
| --- | --- | --- |
| [`docs/`](../docs/) | Users — installation, concepts, workflow, API reference | [Read the Docs](https://buildml.readthedocs.io/) |
| [`guides/`](../guides/) | Users — long tutorials and glossary (Markdown, GitHub-friendly) | Included in Sphinx via MyST |
| `maintainers/` (here) | Maintainers and curious contributors | In-repo only; not on RTD |

Sphinx is the published site. Markdown guides are canonical for quickstarts so
GitHub and RTD stay aligned. This folder is versioned for transparency: you
can see why an API looks the way it does without opening a private wiki.

## How to read these files

**Locks** (`*-m0-lock.md`) record decisions we do not want to re-litigate every
PR — public method names, bundle schemas, CI scope. **Phase plans** describe
what shipped in each alpha and what was explicitly deferred. **Alpha gates** and
**release checklists** are sign-off criteria for a version line, not user
tutorials.

Write new notes the way you would for a teammate joining next month: decision,
rationale, current state. Skip conversational residue, duplicate status
stamps, and process theater.

## File index

| File | Purpose |
| --- | --- |
| `editorial-standards.md` | Copy and terminology rules for reports and user-facing docs |
| `quality-bar.md` | Engineering quality expectations for releases |
| `development-environment.md` | Dev setup for contributors |
| `architecture-review.md` | Current 2.3 Session architecture (supersedes 1.x audit) |
| `classical-ml-capability-map.md` | Classical feature inventory |
| `ingest-engine-checkpoint-design.md` | Ingest, engine, and checkpoint design |
| `phase-1-build-plan.md` | Classical 2.x reconstruction plan |
| `reconstruction-roadmap.md` | Rebuild sequencing; domains through AI `2.3.0a1` shipped |
| `deep-learning-phase-plan.md` | Torch rollout plan |
| `dl-m0-lock.md` | Locked Torch public API |
| `dl-alpha-gate.md` | Torch alpha sign-off criteria |
| `rag-phase-plan.md` | RAG rollout plan |
| `rag-m0-lock.md` | Locked RAG public API |
| `rag-alpha-gate.md` | RAG alpha sign-off criteria |
| `llm-operator-phase-plan.md` | AI operator rollout plan |
| `llm-m0-lock.md` | Locked AI operator public API |
| `ai-alpha-gate.md` | AI operator alpha sign-off criteria |
| `classical-alpha-gate.md` | Classical alpha sign-off criteria |
| `release-checklist-*.md` | Per-version release checklists |

`scripts/lint_user_copy.py` skips `maintainers/` (and quotes in
`editorial-standards.md`) because these files are maintainer records, not
user guidance.

— Leonard
