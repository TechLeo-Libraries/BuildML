# DL M0 design lock

Approved lock for the deep-learning thin slice.  
Parent plan: [deep-learning-phase-plan.md](./deep-learning-phase-plan.md).

**Status:** Locked and shipped in 2.1.0a1.  
**Approved:** 2026-08-01

---

## Public API (Session delegates)

| Method | Role |
| --- | --- |
| `Session.make_torch_loaders(...)` | Build train / validation / test `DataLoader`s from current roles + split |
| `Session.fit_torch(module, ...)` | Train a caller-supplied `nn.Module` on the train loader (early stop / schedulers / `resume=True`) |
| `Session.evaluate_torch(partition=...)` | Metrics on a named partition using the last Torch trainer |
| `Session.torch_training_curve()` | Structured curve series + interpretation / limitations / disclosures |
| `Session.save_torch_bundle(path)` | Persist trainer bundle (weights, opt, scheduler, config, history, contract) |
| `Session.load_torch_bundle(path)` | Restore a trainer bundle into the Session |

Result slot: `session.dl_train_result` (typed `TrainResult`). Classical `fit_result` is unchanged.

Prefix rule: `*_torch` keeps classical `fit` / `evaluate` unambiguous.

Resume recipe (M2): `load_torch_bundle` → `make_torch_loaders` → `fit_torch(..., resume=True)` (additional epochs; optimizer/scheduler state restored when compatible).

---

## Trainer bundle vs Session checkpoint

| Artifact | Schema id | Contains | Does not contain |
| --- | --- | --- | --- |
| Session checkpoint | existing checkpoint formats | data, roles, splits, history, optional classical plans | Torch weights / optimizer |
| Torch trainer bundle | `buildml.torch_bundle.v1` | module state, optimizer (+ scheduler) state, `TrainConfig`, epoch history, early-stop bookkeeping, feature/label contract, device used | dataset rows, split indices, Session history |

Data resume: `checkpoint_load` (or keep Session) → `load_torch_bundle` → `evaluate_torch` and/or `fit_torch(..., resume=True)`.

Layout:

```text
<path>/
  meta.json      # format, buildml_version, contract, config summary
  trainer.pt     # torch.save dict (weights, optimizer, history, …)
```

---

## Packaging and CI

| Decision | Lock |
| --- | --- |
| Canonical extra | `torch` → `pip install 'buildml[torch]'` |
| Alias extra | `dl` → `buildml[torch]` |
| Torch pin | `torch>=2.2` (wheels cover Python 3.10–3.13 on current stable lines) |
| DL CI Python | 3.11 and 3.12 |
| Core import | Must succeed without Torch |
| Version line for DL alpha (M3) | `2.1.0a1` (classical remains `2.0.0a1`) |

---

## Other M0 locks

| Topic | Decision |
| --- | --- |
| Classical preprocess before loaders | No auto-apply; loaders read the current Session frame. Call classical prep explicitly first if needed. |
| Serializer | `torch.save` payload + JSON `meta.json` sidecar |
| Device | Prefer requested device; fall back to CPU with an explicit warning when CUDA/MPS unavailable |
| Normalize | Optional train-fit mean/std in loader path; frozen on val/test |
