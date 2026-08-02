# BuildML architecture review (current)

**Status:** Supersedes the 1 Aug 2026 audit of legacy 1.0.9 / `SupervisedLearning`.  
**Package line:** `2.3.0a1` (AI operator alpha on classical `2.0`, Torch `2.1`, RAG `2.2` bases).  
**Updated:** Phase A correctness pass (Aug 2026).

> Historical 1.x god-object findings remain useful only as the reason the rewrite happened.
> Do not treat the tables below as a description of HEAD.

## Snapshot (2.3 Session reality)

| Metric | Value |
| --- | --- |
| Public root class | `buildml.Session` |
| Architecture | Session orchestrates; domain packages own implementations |
| Classical spine | Ingest → roles → split → preprocess / fold recipes → fit → evaluate → CV/search |
| Optional domains | `buildml.dl` (Torch), `buildml.rag`, `buildml.ai` (operator) |
| Packaging | `pyproject.toml` + extras (`engines`, `torch`, `rag`, `ai`, `dashboard`, …) |
| Tests / CI | pytest matrix + lint; alpha gates under `maintainers/*-alpha-gate.md` |

**Product in one sentence:** BuildML is a hybrid build session for tabular (and attached domain) ML workflows: method-simple Session API, leakage-safe fit scope, honest scale modes, and progressive depth via extras.

**Data flow:** Source → ingest (`Dataset` + engine/mode) → `Session` methods → domain helpers → typed results / checkpoints / pipeline bundles / domain bundles.

## Layering

| Layer | Responsibility |
| --- | --- |
| `buildml.session` | Orchestration, history, walkthrough, public method surface |
| `buildml.data` / `ingest` | Dataset handle, splits, engines, materialization gates |
| `buildml.preprocess` | Session plans + fold-local `PreprocessRecipe` |
| `buildml.model` | Classical fit/evaluate/CV/search/diagnostics |
| `buildml.dl` / `rag` / `ai` | Optional domain packages behind extras |
| `buildml.explain` / `dashboard` | Catalog, Teaching Studio, local dashboard |

## Locked honesty rules (current)

1. **Split before fit-capable work** — `LeakageError` on full-data fit paths.
2. **Session-global preprocess + CV** — refuse by default without fold-local `PreprocessRecipe`; opt-in only via `allow_session_global_preprocess=True`.
3. **`ColumnRole.WEIGHT`** — wired as sklearn `sample_weight` on classical fit/evaluate/CV/search; unsupported estimators raise.
4. **`DataMode`** — `memory` | `lazy` only. Legacy `out_of_core` coerces to `lazy`. There is no out-of-core sklearn fit mode.
5. **Domain bundles ≠ Session checkpoints** — Torch/RAG/AI artifacts stay in their own schemas.

## Related maintainer docs

- [reconstruction-roadmap.md](./reconstruction-roadmap.md) — sequencing (domains shipped through AI alpha)
- [classical-ml-capability-map.md](./classical-ml-capability-map.md)
- [quality-bar.md](./quality-bar.md)
- Domain plans / M0 locks / alpha gates in this folder

## What this file is not

- Not a capability wishlist for Phase B/C/D.
- Not a re-litigation of 1.x `SupervisedLearning` APIs (removed/rewritten).
