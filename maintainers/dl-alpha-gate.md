# DL alpha gate

Concrete exit criteria for declaring BuildML **2.1.0a1** deep-learning alpha.
Sibling to [classical-alpha-gate.md](./classical-alpha-gate.md). This is a
release checklist, not a capability wishlist.

> **Historical gate for the `2.1.0a1` line.** The package version at HEAD is
> **`2.3.0a1`** (AI operator alpha). Do not treat TC4 / sign-off version pins
> below as the current package version — they record what the DL alpha cut
> required. Current packaging: `pyproject.toml` / `buildml/_version.py` →
> `2.3.0a1`.

Related docs: [quickstart-torch.md](../guides/quickstart-torch.md) ·
[dl-m0-lock.md](./dl-m0-lock.md) · [deep-learning-phase-plan.md](./deep-learning-phase-plan.md) ·
[glossary.md](../guides/glossary.md) · [editorial-standards.md](./editorial-standards.md)

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
| TD2 | Quickstart covers loaders → fit → evaluate → bundle → resume/curve and known limits | `guides/quickstart-torch.md` |
| TD3 | Glossary covers Torch/DL terms used in the alpha path | `guides/glossary.md` |
| TD4 | Editorial / user-copy lint clean | `scripts/lint_user_copy.py` in CI |
| TD5 | README documents `torch` / `dl` extras and Session Torch APIs without claiming RAG/LLM | `README.md` |

### CI and packaging

| ID | Criterion | Evidence |
| --- | --- | --- |
| TC1 | `import buildml` succeeds without Torch | Core CI import smoke |
| TC2 | Dedicated `torch` CI job on Python 3.11–3.12 with DL unit + integration tests | `.github/workflows/ci.yml` |
| TC3 | Missing Torch raises `MissingExtraError("torch", ...)` with install hint | Missing-extra unit tests |
| TC4 | Version was `2.1.0a1` in `pyproject.toml` and `buildml/_version.py` at the DL alpha cut (HEAD is now `2.3.0a1`) | Historical packaging pin |

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
2. **Tabular + text/sequence + tabular/text/image/audio multimodal fusion in scope**
   (Pass G/J/L). Audio multimodal uses a small 1D-CNN fusion branch. A separate
   **speech ASR + finetune-lite** path ships in Pass O (`buildml[speech]`) —
   integration/finetune, not Whisper-scale FM training from scratch.
3. **Built-in models are a happy path** (tabular MLP, text classifier, multimodal
   fusion when `fit_torch` omits `module`). No broad model zoo; custom `nn.Module`
   remains first-class.
4. **Materialized tensors.** Partition frames become batches via Pandas/NumPy; no
   Polars/DuckDB zero-copy into DataLoaders.
5. **Classical preprocess is not auto-applied** before loaders (disclosure /
   `apply_plans=` bridge only).
6. **Fold-local Torch CV** (`cross_validate_torch`) and **nested Torch HPO**
   (`nested_cv_torch` / `search_torch`) are shipped. Single-node DDP and
   torchrun **multi-node** join (`fit_torch_ddp(..., multi_node=True)`) ship in
   Pass O — not Kubernetes multi-cluster orchestration.
7. **AMP + TorchScript/ONNX export** (`mixed_precision`, `export_torch`) and
   **managed local serving** (`buildml[serve]` / `serve_bundle`) are alpha
   product paths. Serving is localhost-oriented with no auth product claim.
   ONNX opset/dynamic-axes limits apply. No AutoML architecture search beyond
   the documented MLP knob space.
8. **RAG / LLM operator** are separate domains (`2.2` / `2.3` lines).
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
- [ ] Version was `2.1.0a1` at DL alpha cut (current HEAD package is `2.3.0a1`)
- [ ] Changelog / history notes name this gate document
- [ ] Docs do not claim RAG/LLM as shipped features

Tag only after remote CI is green on the release candidate push. Do not tag from
this checklist alone.
