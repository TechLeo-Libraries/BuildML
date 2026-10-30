# Artifacts: checkpoints vs bundles

```bash
pip install buildml
```

A checkpoint resumes the data workflow. A pipeline bundle scores new rows.
A domain bundle holds that domain's fitted plan. They store different information. Choose the artifact according to whether
you need to resume data preparation or use a fitted model for predictions.

Pickle / joblib / torch loaders default to `trusted=False`. Pass
`trusted=True` only for a file you created or whose source and contents you trust. RAG bundles are JSONL / NumPy and
do not use that gate.

---

## Artifact types and scoring

| Artifact | Typical API | Holds | Does not hold |
| --- | --- | --- | --- |
| Session checkpoint | `checkpoint_save` / `checkpoint_load` | data, roles, splits, history, optional preprocess plan objects, integrity manifest | Fitted estimator weights, Torch trainer, RAG index, AI keys |
| Estimator model bundle | `save_model` / `load_model` | estimator + feature contract | Preprocess plans, dataset, splits |
| Pipeline bundle | `save_pipeline` / `load_pipeline` | plans + estimator + model card + schema contract | Dataset rows, full history, Torch/RAG |
| Score helper | `predict_from_pipeline` | one-shot inference | Does not mutate Session |

Categorical pipeline contracts distinguish raw labels from prepared numeric
features. Ordinal encoding can keep the same column name in both stages:
raw labels `"01"` and `"1"` remain distinct, while `apply_plans=False` expects
their already encoded numeric values. Newly fitted encode plans retain their
raw column types. Raw source columns remain required even when another encoded
output reuses their name. Missing raw inputs and malformed numeric features
raise `ValidationError`.

For native encode plans, raw categorical inputs use the encoder's string
conversion policy before replay. Nullable-string coercion must not merge the
null labels learned at fitting. Null-category behavior was checked separately
under Pandas 2.3.3 and 3.0.5, each using bundles fitted in that runtime. Ordinary
nullable-string coercion still applies when no native encode plan consumes the
column, and numeric contract checks retain their normal validation.

Older bundles may already contain an input contract inferred from post-encoding
types. Loading does not reconstruct missing raw type information. For an
affected ordinal bundle, refit the encoder and model from the original raw data
and save a new bundle, or rebuild an explicit schema contract from the known raw
schema using `build_schema_contract` and pass it to `save_pipeline_bundle`.
Include the encoded output names in `encoded_numeric_columns` so feature-stage
validation remains numeric. Passing transformed values as raw labels can change
their meaning; validate the replacement bundle against known predictions.

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
and templates. Deploying those templates requires a separately configured service.

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

restored = Session.checkpoint_load("artifacts/checkpoint", trusted=True)
print(restored.reattach_result.status)

restored.fit(LogisticRegression(max_iter=500), task="classification")
restored.save_pipeline("artifacts/pipeline", evaluate_partition="test")
print(restored.model_card.lineage.get("plans_present"))

# Estimator-only (no plans): prefer pipeline when prep must travel:
restored.save_model("artifacts/model_only")
```

`data_only=True` on load discards prior workflow semantics. Use it when
you want the frame without replaying history.

Saving successfully into a reused checkpoint directory replaces the recorded
snapshot, including its preprocessing plans. Saving a session with no plans
removes the previous plans file. Restore also honors an explicit
`meta.has_plans=False` in checkpoints written by older versions that left an
inactive file behind. Metadata without that flag retains the legacy behavior:
loading a plans file requires `trusted=True`.

When a partitioned native sidecar records a `parts` list, restore requires exactly
those Parquet files. An added or missing partition raises `ValidationError`
before native reattachment, preserving the current Session on a failed
`reattach`. Recover the original files to resume the saved split, or use
`data_only=True` to read `data/frame.parquet` as fresh ingest, then assign roles
and create a new split. Split membership is positional; store a meaningful
pandas index as an explicit column.

`session.reattach(...)` replaces the current in-memory workflow state while
keeping the Session object's identity. Models, domain plans, retrieval indexes,
AI configuration and cached reports from the previous dataset are discarded.
It restores the checkpoint's rows, split, history and preprocessing plans; load
the appropriate trusted model/domain bundle explicitly before scoring.

Historical fit/evaluate records remain part of the audit trail. They do not make
those operations available when a required fitted model is absent. Workflow,
explain and dry-run views report the current missing prerequisites. Dry-run
accepts flat names and facade paths such as `classical.evaluate` or
`session.graph.evaluate`, returning their canonical names in the report.
Unknown names remain blocked steps with `status="unknown"`. Previews and
printed audit summaries do not fit, transform or append history; risk rankings
remain heuristic review cues.

---

## Use case: predict_from_pipeline on new rows

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

restored = Session.checkpoint_load("artifacts/checkpoint", trusted=True)
print(restored.reattach_result.status)

restored.fit(LogisticRegression(max_iter=500), task="classification")
restored.save_pipeline("artifacts/pipeline", evaluate_partition="test")
print(restored.model_card.lineage.get("plans_present"))

# Estimator-only (no plans): prefer pipeline when prep must travel:
restored.save_model("artifacts/model_only")

from buildml.pipeline import predict_from_pipeline

holdout = restored.partition("test")
scored = predict_from_pipeline(
    "artifacts/pipeline",
    holdout,
    return_proba=True,
    trusted=True,
)
print(scored)
```

Schema mismatches raise clearly. Resample plans do not synthesize
inference rows.

---

## Torch, RAG, and AI artifacts

Use the complete examples in the corresponding guides:

| Artifact | Complete workflow |
| --- | --- |
| Torch model bundle and loader reconstruction | [Torch guide](torch-deep.md) |
| RAG embeddings, index, and configuration | [RAG guide](rag-deep.md) |
| AI conversation and tool-call transcript | [AI operator guide](ai-operator-safety.md) |

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
