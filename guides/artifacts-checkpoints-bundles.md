# Artifacts: checkpoints vs bundles vs Torch/RAG/AI

> **Install (GitHub 2.x):**
> `pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"`
> See [installation](../docs/installation.rst).

BuildML separates **workflow resume** from **deployable scoring** from
**domain-specific trainer/index/transcript** artifacts. Mixing them causes
silent gaps (no weights in a checkpoint, no dataset in a pipeline).

---

## Conceptual matrix

| Artifact | Typical API | Contains | Does **not** contain |
| --- | --- | --- | --- |
| Session checkpoint | `checkpoint_save` / `checkpoint_load` | data, roles, splits, history, optional preprocess **plan objects**, integrity manifest | Fitted estimator weights, Torch trainer, RAG index, AI keys/transcript |
| Estimator model bundle | `save_model` / `load_model` | estimator + feature contract | Preprocess plans, dataset, splits |
| Pipeline bundle | `save_pipeline` / `load_pipeline` | plans + estimator + model card + schema contract | Dataset rows, full history, Torch/RAG |
| Score helper | `predict_from_pipeline` | one-shot inference | Does not mutate Session |
| Torch trainer bundle | `save_torch_bundle` / `load_torch_bundle` | weights, optimizer (+ scheduler), config, history, feature contract, optional multimodal_preprocess meta | Dataset, split indices; **load does not rebuild DataLoaders** (rebuild with `make_multimodal_torch_loaders(..., use_saved_preprocess=True)` or `preprocess=`) |
| RAG bundle | `save_rag_bundle` / `load_rag_bundle` | embeddings, index, chunk config | Tabular Session data, Torch weights |
| AI transcript | `save_ai_transcript` / `load_ai_transcript` | conversation, tool calls, egress manifests | API keys; raw rows unless FULL_SAMPLE opt-in |
| TorchServe pack | `pack_torchserve` | directory recipe for operator-owned TorchServe | Running server |
| TensorRT plan | `prepare_tensorrt_export` | `trtexec` plan files | Built `.engine` (operator builds) |
| K8s DDP YAML | `emit_k8s_ddp_job` | Job template | Live multi-cluster orchestration |

Schemas to remember: `buildml.torch_bundle.v1`, `buildml.rag_bundle.v1`,
`buildml.ai_transcript.v1`.

---

## Use case — checkpoint mid-loop, pipeline at the end

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

# Estimator-only (no plans) — prefer pipeline when prep must travel:
restored.save_model("artifacts/model_only")
```

`data_only=True` on load deliberately discards prior workflow semantics — use
when you want the frame without replaying history.

---

## Use case — predict_from_pipeline on new rows

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

Schema mismatches raise clearly. Resample plans do not synthesize inference rows.

---

## Use case — Torch / RAG / AI stay separate

```python
# Torch (buildml[torch])
# session.save_torch_bundle("artifacts/torch_bundle")
# restored.load_torch_bundle(path, module, map_location="cpu")
# restored.make_torch_loaders(...)  # required again — load does not rebuild loaders

# RAG (buildml[rag])
# session.save_rag_bundle("artifacts/rag_bundle")
# Session().load_rag_bundle("artifacts/rag_bundle")

# AI (buildml[ai])
# session.save_ai_transcript("artifacts/transcript.json")  # secrets redacted
```

Serving a pipeline or TorchScript artifact:
[serve-deploy](serve-deploy.md).

---

## Reattach statuses

Inspect `reattach_result` after checkpoint load. Typical outcomes include
resume-ready vs blocked (schema/integrity mismatch) vs fresh-ingest guidance.
Do not assume a checkpoint is a deployable model.

---

## Failure modes

| Mistake | Consequence |
| --- | --- |
| Expecting weights in a checkpoint | No estimator — call `save_pipeline` / `save_torch_bundle` |
| Expecting dataset in a pipeline | Scoring artifact only |
| Loading Torch bundle and evaluating without loaders | `ValidationError` — rebuild correct loader kind |
| Committing AI transcripts with FULL_SAMPLE | Privacy risk — prefer STATS_ONLY + redact |
| Treating TorchServe/TRT/K8s helpers as managed cloud | Recipes/templates only |

---

## Related

- [Classical end-to-end](classical-end-to-end.md)
- [Torch deep](torch-deep.md)
- [RAG deep](rag-deep.md)
- [Serve & deploy](serve-deploy.md)
- [AI safety](ai-operator-safety.md)
