# Artifacts: checkpoints vs bundles

```bash
pip install buildml
```

A checkpoint resumes the data workflow. A pipeline bundle scores new rows.
A domain bundle holds that domain's fitted plan. They do not embed each
other. Load a checkpoint expecting weights, or a pipeline expecting the
table, and you will get a silent gap.

Pickle / joblib / torch loaders default to `trusted=False`. Pass
`trusted=True` only for a file you made. RAG bundles are JSONL / NumPy and
do not use that gate.

---

## The three you pick first

| Artifact | Typical API | Holds | Does not hold |
| --- | --- | --- | --- |
| Session checkpoint | `checkpoint_save` / `checkpoint_load` | data, roles, splits, history, optional preprocess plan objects, integrity manifest | Fitted estimator weights, Torch trainer, RAG index, AI keys |
| Estimator model bundle | `save_model` / `load_model` | estimator + feature contract | Preprocess plans, dataset, splits |
| Pipeline bundle | `save_pipeline` / `load_pipeline` | plans + estimator + model card + schema contract | Dataset rows, full history, Torch/RAG |
| Score helper | `predict_from_pipeline` | one-shot inference | Does not mutate Session |

A domain mixin that fits also has `save_bundle` / `load_bundle`. That
file holds the domain's `*Plan` (estimator or factors, plus the
disclosures that domain needs). It does not hold the dataset, split
indices, or a different domain's weights. Torch, RAG, and AI transcripts
are separate on purpose:

- `session.dl.save_bundle` / `load_bundle`: weights, optimizer, config,
  history. Load does not rebuild DataLoaders. Rebuild with
  `session.dl.make_multimodal_loaders(..., use_saved_preprocess=True)` or
  `preprocess=`.
- `session.rag.save_bundle` / `load_bundle`: embeddings, index, chunk
  config.
- `session.ai.save_transcript` / `load_transcript`: conversation and
  tool calls. Keys redacted. Raw rows stay out unless you opted into
  `FULL_SAMPLE`.

Serving helpers (`session.dl.pack_torchserve`,
`session.dl.prepare_tensorrt`, `session.dl.emit_k8s_ddp`) write recipes
and templates. They do not start a managed cloud.

Schemas follow `buildml.<domain>_bundle.v1` (and
`buildml.torch_bundle.v1`, `buildml.rag_bundle.v1`,
`buildml.ai.transcript.v1`). Inspect the loaded plan rather than
assuming a checkpoint is a deployable model.

---

## Use case: checkpoint mid-loop, pipeline at the end

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression

from buildml import Session

frame = pd.DataFrame(
    {
        "age": [21, None, 35, 40, 29, 33, 52, 47],
        "income": [40, 55, 60, 80, 50, 70, 90, 65],
        "approved": [0, 1, 0, 1, 0, 1, 1, 0],
    }
)

session = (
    Session.ingest(frame)
    .set_roles({"age": "feature", "income": "feature", "approved": "target"})
    .split(test_size=0.25, stratify=True, random_state=0)
    .impute(strategy="median")
    .scale(method="standard")
)

session.checkpoint_save(
    "artifacts/checkpoint",
    sidecar_layout="auto",
    sidecar_partition_rows=25_000,
    sidecar_compression="zstd",
)

restored = Session.checkpoint_load("artifacts/checkpoint")
print(restored.reattach_result.status)

restored.fit(LogisticRegression(max_iter=500), task="classification")
restored.save_pipeline("artifacts/pipeline", evaluate_partition="test")
print(restored.model_card.lineage.get("plans_present"))

# Estimator-only (no plans): prefer pipeline when prep must travel:
restored.save_model("artifacts/model_only")
```

`data_only=True` on load discards prior workflow semantics. Use it when
you want the frame without replaying history.

---

## Use case: predict_from_pipeline on new rows

```python
from buildml.pipeline import predict_from_pipeline

holdout = restored.partition("test")
scored = predict_from_pipeline(
    "artifacts/pipeline",
    holdout,
    return_proba=True,
)
print(scored)
```

Schema mismatches raise clearly. Resample plans do not synthesize
inference rows.

---

## Use case: Torch / RAG / AI stay separate

```python
# Torch (buildml[torch])
# session.dl.save_bundle("artifacts/torch_bundle")
# restored.dl.load_bundle(path, module, map_location="cpu")
# restored.dl.make_loaders(...)  # required again: load does not rebuild loaders

# RAG (buildml[rag])
# session.rag.save_bundle("artifacts/rag_bundle")
# Session().rag.load_bundle("artifacts/rag_bundle")

# AI (buildml[ai])
# session.ai.save_transcript("artifacts/transcript.json")  # secrets redacted
```

Serving a pipeline or TorchScript artifact:
[serve-deploy](serve-deploy.md).

---

## Reattach statuses

Inspect `reattach_result` after checkpoint load. Typical outcomes are
resume-ready, blocked (schema or integrity mismatch), or fresh-ingest
guidance. Do not assume a checkpoint is a deployable model.

---

## Failure modes

| Mistake | Consequence |
| --- | --- |
| Expecting weights in a checkpoint | No estimator: call `save_pipeline` / `session.dl.save_bundle` |
| Expecting dataset in a pipeline | Scoring artifact only |
| Loading Torch bundle and evaluating without loaders | `ValidationError`: rebuild the correct loader kind |
| Committing AI transcripts with FULL_SAMPLE | Privacy risk: prefer STATS_ONLY + redact |
| Treating TorchServe/TRT/K8s helpers as managed cloud | Recipes/templates only |

---

## Related

- [Classical end-to-end](classical-end-to-end.md)
- [Torch deep](torch-deep.md)
- [RAG deep](rag-deep.md)
- [Serve & deploy](serve-deploy.md)
- [AI safety](ai-operator-safety.md)
