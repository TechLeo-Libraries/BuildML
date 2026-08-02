# AI tools and operator patterns

> **Install (GitHub 2.x + AI):**
> ```bash
> pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"
> pip install "buildml[ai]"
> ```
> See [installation](../docs/installation.rst).

This guide is the **operator playbook**: which tools exist, how to chain them
safely across classical / RAG / Torch, and patterns that avoid silent leakage.

Safety primitives: [ai-operator-safety](ai-operator-safety.md).
Quickstart: [quickstart-ai](quickstart-ai.md).

---

## Why a typed allowlist

The model never gets a Python REPL. Every side effect goes through
`ToolSpec` entries in `build_default_registry()`. That is the trust boundary:
prompt injection can *ask* for `os.system`, but the executor cannot run it.

```python
from buildml.ai import registered_tool_names

print(registered_tool_names())
```

---

## Tool inventory (default registry)

### Read-only

| Tool | Purpose |
| --- | --- |
| `describe_dataset` | Schema / shape summary |
| `explain_operation` | Catalog overlay for an operation |
| `workflow_status` | Resolver statuses |
| `eda_summary` | Compact EDA findings |
| `dry_run_plan` | Preview ops without mutation |
| `head` | Preview rows (respects egress) |
| `evaluate` | Metrics for a partition (read of fitted model) |
| `walkthrough` | Audit report |
| `ai_status` | Provider / budget / transcript status |
| `rag_retrieve` | Retrieve chunks (index must exist) |

### Classical writes (confirm)

| Tool | Purpose |
| --- | --- |
| `set_roles` | Column roles |
| `split` | Random/stratified split |
| `impute` / `encode` / `scale` | Train-fitted prep |
| `fit` | Sklearn fit on train |
| `checkpoint_save` | Workflow checkpoint |

### Destructive (always confirm)

| Tool | Purpose |
| --- | --- |
| `drop_columns` | Drop columns |

### RAG writes

| Tool | Purpose |
| --- | --- |
| `rag_ingest_corpus` | Load corpus |
| `rag_embed_and_index` | Build index (refuses `eval_only` contamination) |
| `rag_generate` | Grounded generate (needs provider) |

### Torch / speech / packs

| Tool | Purpose |
| --- | --- |
| `make_torch_loaders` / `make_text_torch_loaders` | Tabular / text loaders |
| `make_multimodal_torch_loaders` | Tabular+text (+ optional image/audio) |
| `make_image_multimodal_torch_loaders` / `make_audio_multimodal_torch_loaders` | Image/audio-centric |
| `fit_torch` / `evaluate_torch` / `cross_validate_torch` | Train / eval / CV |
| `search_torch` / `nested_cv_torch` | HPO / nested |
| `export_torch` | TorchScript / ONNX |
| `make_speech_torch_loaders` / `fit_speech_torch` / `domain_adapt_speech_torch` | Speech classify |
| `transcribe_speech` / `evaluate_asr` | ASR path |
| `load_pretrained_backbone` / `attach_backbone_head` | Curated backbones |
| `pack_torchserve` / `prepare_tensorrt_export` / `emit_k8s_ddp_job` / `emit_k8s_serve_deployment` | Operator recipes / YAML templates |

**Note:** `serve_bundle` is Session/CLI-primary (not an AI tool) — localhost
serving stays out of the LLM allowlist by design. See
[serve-deploy](serve-deploy.md).

Exact names evolve with the library; CI keeps the teaching catalog synced.
When unsure, print `registered_tool_names()`.

---

## Pattern 1 — Classical propose → confirm chain

```python
from buildml import Session
import pandas as pd

session = Session.ingest(
    pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1], "y": [0, 1, 0, 1]})
)
session.ai_configure(provider="mock")

for tool, params in [
    ("set_roles", {"mapping": {"a": "feature", "b": "feature", "y": "target"}}),
    ("split", {"test_size": 0.25, "stratify": True, "random_state": 0}),
    ("impute", {"strategy": "median"}),
    ("scale", {"method": "standard"}),
]:
    session.ai_execute(tool, params, confirm=True)

session.ai_execute(
    "fit",
    {"estimator": "LogisticRegression", "task": "classification"},
    confirm=True,
)
print(session.ai_execute("evaluate", {"partition": "test"}, confirm=False))
```

If a tool’s parameter schema rejects an estimator shorthand, fall back to
direct Session APIs for that step — never invent kwargs.

---

## Pattern 2 — RAG retrieve then grounded generate

```python
session = Session()
session.ai_configure(provider="mock")
session.ai_execute(
    "rag_ingest_corpus",
    {"documents": [{"doc_id": "a", "text": "Hold out a test partition."}]},
    confirm=True,
)
session.ai_execute("rag_embed_and_index", {}, confirm=True)
hits = session.ai_execute(
    "rag_retrieve",
    {"query": "test partition", "k": 3},
    confirm=False,
)
print(hits)
```

Keep `eval_only` documents out of index tools
([rag-deep](rag-deep.md)).

---

## Pattern 3 — Torch loaders → fit → evaluate

```python
# After roles + split on a numeric frame:
session.ai_configure(provider="mock")
session.ai_execute("make_torch_loaders", {"batch_size": 4, "normalize": True}, confirm=True)
session.ai_execute("fit_torch", {"epochs": 3, "device": "cpu"}, confirm=True)
session.ai_execute("evaluate_torch", {"partition": "validation"}, confirm=False)
```

Nested search tools exist (`search_torch`, `nested_cv_torch`) — still do not
tune on Session test.

---

## Pattern 4 — Autonomy with a tight allowlist

```python
session.ai_run_autonomous(
    "report workflow status after describing the dataset",
    confirm_autonomy=True,
    max_steps=4,
    tool_allowlist=["describe_dataset", "workflow_status", "ai_status"],
    allow_destructive=False,
)
```

Shrink the allowlist to the minimum for the job. Broad allowlists + autonomy
increase residual risk even with caps.

---

## Pattern 5 — Teaching-first before writes

```python
session.ai_execute(
    "explain_operation",
    {"operation": "impute", "moment": "before"},
    confirm=False,
)
session.ai_execute("dry_run_plan", {"operations": ["impute", "scale", "fit"]}, confirm=False)
session.ai_execute("workflow_status", {}, confirm=False)
```

---

## Failure modes

| Issue | Guidance |
| --- | --- |
| Hallucinated tool name | Rejected — print registry |
| Hallucinated params | Schema validation error — fix or use Session API |
| Auto-confirm destructive | Not allowed |
| Autonomy + FULL_SAMPLE egress | Blocked by safety caps |
| Serving via AI tool | Use CLI/`serve_bundle` instead |

---

## Related

- [AI operator safety](ai-operator-safety.md)
- [Torch deep](torch-deep.md)
- [RAG deep](rag-deep.md)
- [Speech](speech-asr-finetune.md)
- [Pretrained](pretrained-backbones.md)
