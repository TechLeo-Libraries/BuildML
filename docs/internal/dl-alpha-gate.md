# DL alpha gate

Concrete exit criteria for declaring BuildML **2.1.0a1** deep-learning alpha.
Sibling to [classical-alpha-gate.md](./classical-alpha-gate.md). This is a
release checklist, not a capability wishlist.

Related docs: [quickstart-dl-alpha.md](../quickstart-dl-alpha.md) ·
[dl-m0-lock.md](./dl-m0-lock.md) · [deep-learning-phase-plan.md](./deep-learning-phase-plan.md) ·
[glossary.md](../glossary.md) · [editorial-standards.md](./editorial-standards.md)

---

## Verdict rubric

| Status | Meaning |
| --- | --- |
| **Pass** | Every **must** criterion below is green in CI or explicitly verified |
| **Fail** | Any **must** criterion is red, missing, or contradicted by docs |
| **Conditional** | Musts pass, but a listed known limit blocks a claimed workflow |

Assess readiness after CI: **Pass** when all must IDs are green; otherwise
**Fail** or **Conditional** per the known-limits section.

---

## Must criteria

### Leakage and fit scope

| ID | Criterion | Evidence |
| --- | --- | --- |
| TL1 | Train DataLoader may shuffle train only; validation/test loaders do not shuffle into train batches | `tests/unit/test_dl_torch_slice.py` leakage tests |
| TL2 | Optional loader normalize fits mean/std on train and freezes stats on validation/test | Loader unit tests + catalog leakage notes |
| TL3 | Guided training refuses fit without a split (same `assert_can_fit` philosophy as classical) | Session Torch delegates + unit tests |
| TL4 | Early stopping monitors validation (default `val_loss`); docs forbid test-tuned early stop | `TrainConfig` / catalog / concept `early-stopping-partition` |
| TL5 | Group and time split membership is honored by loaders (no cross-partition mix) | `tests/unit/test_dl_m2_depth.py` |

### End-to-end smoke

| ID | Criterion | Evidence |
| --- | --- | --- |
| TS1 | Path: ingest → roles → split → `make_torch_loaders` → `fit_torch` → `evaluate_torch` → save/load bundle → optional resume + training curve | `tests/integration/test_dl_alpha_smoke.py` |
| TS2 | Smoke runs with `buildml[torch]` (not required on core CI) | CI `torch` job |
| TS3 | Trainer bundle and Session checkpoint remain distinct artifacts | Bundle schema tests + TS1; `CHECKPOINT_BOUNDARY` |

### Docs and catalog

| ID | Criterion | Evidence |
| --- | --- | --- |
| TD1 | Public Session Torch methods have catalog entries | `buildml.explain.catalog` + DL unit tests |
| TD2 | Quickstart covers loaders → fit → evaluate → bundle → resume/curve and known limits | `docs/quickstart-dl-alpha.md` |
| TD3 | Glossary covers Torch/DL terms used in the alpha path | `docs/glossary.md` |
| TD4 | Editorial / user-copy lint clean | `scripts/lint_user_copy.py` in CI |
| TD5 | README documents `torch` / `dl` extras and Session Torch APIs without claiming RAG/LLM | `README.md` |

### CI and packaging

| ID | Criterion | Evidence |
| --- | --- | --- |
| TC1 | `import buildml` succeeds without Torch | Core CI import smoke |
| TC2 | Dedicated `torch` CI job on Python 3.11–3.12 with DL unit + integration tests | `.github/workflows/ci.yml` |
| TC3 | Missing Torch raises `MissingExtraError("torch", ...)` with install hint | Missing-extra unit tests |
| TC4 | Version is `2.1.0a1` in `pyproject.toml` and `buildml/_version.py` | Packaging files |

---

## Should criteria (alpha-tolerant)

| ID | Criterion | Notes |
| --- | --- | --- |
| TW1 | Early stopping, grad clip, and LR schedulers (`none`/`step`/`plateau`/`cosine`) | Covered by M2 depth tests |
| TW2 | Structured `TrainingCurveReport` + walkthrough `torch_training_status` | Core DL results; Studio is optional |
| TW3 | Resume training via `load_torch_bundle` → `fit_torch(..., resume=True)` | Contract + optimizer/scheduler restore when compatible |
| TW4 | Confusion / residual-style diagnostics on `evaluate_torch` | Structured fields on evaluate result |

---

## Known limits (do not claim as done)

1. **CPU slice is the merge gate.** No GPU CI on every PR; CUDA/MPS fall back with
   an explicit warning when unavailable.
2. **Tabular numeric features first.** No image / sequence / multimodal product path.
3. **No built-in model zoo.** Caller supplies `nn.Module`; docs may show tiny examples only.
4. **Materialized tensors.** Partition frames become batches via Pandas/NumPy; no
   Polars/DuckDB zero-copy into DataLoaders.
5. **Classical preprocess is not auto-applied** before loaders.
6. **No fold-local Torch CV** and no DistributedDataParallel in this alpha.
7. **No mixed precision, TorchScript/ONNX productization, or AutoML architecture search.**
8. **RAG / LLM operator** are out of DL alpha scope (later domains).
9. **Session checkpoints never embed Torch weights.** Use `buildml.torch_bundle.v1`.

---

## Smoke path (canonical)

```text
Session.ingest → set_roles → split
  → make_torch_loaders
  → fit_torch (small MLP, few epochs, CPU; optional early stop)
  → evaluate_torch(partition="test")
  → save_torch_bundle / load_torch_bundle
  → torch_training_curve (optional)
  → fit_torch(..., resume=True) (optional)
  → explain("fit_torch")
```

CI entry: `pytest tests/integration/test_dl_alpha_smoke.py -q`

---

## Sign-off checklist

Copy into a release note when cutting a DL alpha tag (see also
[release-checklist-dl-a1.md](./release-checklist-dl-a1.md)):

- [ ] TL1–TL5 green
- [ ] TS1–TS3 green on CI `torch` job
- [ ] TD1–TD5 green
- [ ] TC1–TC4 green
- [ ] Known limits reviewed; README/quickstart/`CHANGELOG.md` do not contradict them
- [ ] Version is `2.1.0a1` in `pyproject.toml` and `buildml/_version.py`
- [ ] Changelog / history notes name this gate document
- [ ] Docs do not claim RAG/LLM as shipped features

Tag only after remote CI is green on the release candidate push. Do not tag from
this checklist alone.
