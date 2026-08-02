# Maintainer documentation

Engineering notes for BuildML maintainers: phase plans, release gates, design
locks, editorial standards, and reconstruction records. Versioned in git for
project history; **not** part of the primary user documentation path.

## Where documentation lives

| Location | Purpose | Published site |
| --- | --- | --- |
| [`docs/`](../docs/) | Sphinx source only — `.rst`, `conf.py`, build helpers | [Read the Docs](https://buildml.readthedocs.io/) |
| [`guides/`](../guides/) | User quickstarts and glossary (Markdown, GitHub-friendly) | Included in Sphinx via MyST |
| `maintainers/` (here) | Internal plans, gates, checklists | Not published |

For usage, start with the public [README](../README.md),
[classical quickstart](../guides/quickstart-classical.md), and
[Sphinx docs](../docs/index.rst).

## What lives here

| File | Purpose |
| --- | --- |
| `editorial-standards.md` | Copy and terminology rules for reports and docs |
| `quality-bar.md` | Engineering quality expectations for releases |
| `phase-1-build-plan.md` | Classical 2.x reconstruction plan |
| `reconstruction-roadmap.md` | Longer-term rebuild sequencing |
| `architecture-review.md` | Architecture review notes |
| `classical-ml-capability-map.md` | Classical feature inventory |
| `development-environment.md` | Maintainer dev setup |
| `ingest-engine-checkpoint-design.md` | Ingest/engine/checkpoint design notes |
| `deep-learning-phase-plan.md` | Torch/DL rollout plan |
| `rag-phase-plan.md` | RAG rollout plan |
| `llm-operator-phase-plan.md` | AI operator rollout plan |
| `dl-m0-lock.md` | Locked DL public API (M0) |
| `rag-m0-lock.md` | Locked RAG public API (M0) |
| `llm-m0-lock.md` | Locked AI operator public API (M0) |
| `classical-alpha-gate.md` | Classical alpha sign-off criteria |
| `dl-alpha-gate.md` | Torch alpha sign-off criteria |
| `rag-alpha-gate.md` | RAG alpha sign-off criteria |
| `ai-alpha-gate.md` | AI operator alpha sign-off criteria |
| `release-checklist-a1.md` | Classical 2.0.0a1 release checklist |
| `release-checklist-dl-a1.md` | Torch 2.1.0a1 release checklist |
| `release-checklist-rag-a1.md` | RAG 2.2.0a1 release checklist |
| `release-checklist-ai-a1.md` | AI 2.3.0a1 release checklist |

`scripts/lint_user_copy.py` skips everything under `maintainers/` (including
`editorial-standards.md`, which quotes banned phrases as examples).
