# Serve and deploy recipes

> **Install (GitHub 2.x + serve):**
> ```bash
> pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"
> pip install "buildml[serve]"
> # Torch export packs also need buildml[torch]; ONNX checker: buildml[onnx]
> ```
> See [installation](../docs/installation.rst).

BuildML provides **local managed serving** for classical pipeline bundles and
TorchScript artifacts, plus **operator-owned recipes** for TorchServe,
TensorRT (`trtexec`), and Kubernetes torchrun Jobs. This is not a managed
cloud IAM / multi-cluster product.

Related: [artifacts](artifacts-checkpoints-bundles.md), [torch-deep](torch-deep.md),
[features](../docs/features.rst).

---

## Why localhost-first

Exposing ML scores without auth is a common incident. Defaults:

- Bind `127.0.0.1`
- Optional API-key / Bearer middleware
- Non-loopback binds require `api_keys` unless
  `allow_insecure_public_bind=True`
- Prefer TLS at a reverse proxy for any non-local exposure

---

## Use case A — Serve a classical pipeline bundle

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
    .fit(LogisticRegression(max_iter=500), task="classification")
)
session.save_pipeline("artifacts/pipeline", evaluate_partition="test")

# pip install "buildml[serve]"
handle = session.serve_bundle(
    "artifacts/pipeline",
    kind="pipeline",
    host="127.0.0.1",
    port=8080,
    api_keys=["dev-key"],
    blocking=False,
)
print(handle)
# handle.stop() when finished
```

CLI equivalents:

```bash
buildml-serve --bundle artifacts/pipeline --kind pipeline --api-key dev-key
# or: python -m buildml.serving --bundle artifacts/pipeline --kind pipeline
```

---

## Use case B — Serve TorchScript

```python
# After fit_torch + export_torch("artifacts/model.ts.pt", format="torchscript"):
# session.serve_bundle(
#     "artifacts/model.ts.pt",
#     kind="torchscript",
#     api_keys=["dev-key"],
# )
```

Scoring contracts differ by `kind` — do not assume pipeline JSON equals
TorchScript tensor payloads. Inspect the serving OpenAPI / docs route locally.

---

## Use case C — Auth and public bind honesty

```python
# Refused without keys on non-loopback (unless allow_insecure_public_bind):
# session.serve_bundle(
#     "artifacts/pipeline",
#     host="0.0.0.0",
#     api_keys=["rotate-me"],
# )

# Emergency lab-only override (do not use in production):
# session.serve_bundle(
#     "artifacts/pipeline",
#     host="0.0.0.0",
#     allow_insecure_public_bind=True,
# )
```

API keys are **not** cloud IAM. Rotate keys; terminate TLS at a proxy.

---

## Use case D — TorchServe directory pack (recipe)

```python
# session.export_torch("artifacts/model.ts.pt", format="torchscript")
# result = session.pack_torchserve(
#     "artifacts/torchserve_dir",
#     torchscript_path="artifacts/model.ts.pt",
#     model_name="buildml_model",
# )
# Operator runs TorchServe against the directory — BuildML does not start it.
```

---

## Use case E — TensorRT trtexec plan (recipe)

```python
# session.export_torch("artifacts/model.onnx", format="onnx")
# plan = session.prepare_tensorrt_export(
#     "artifacts/trt_plan",
#     onnx_path="artifacts/model.onnx",
#     engine_name="model.engine",
#     fp16=True,
# )
# Operator runs trtexec — BuildML does not build .engine files.
```

---

## Use case F — Kubernetes torchrun Job YAML (template)

```python
session = Session()
session.emit_k8s_ddp_job(
    "artifacts/ddp-job.yaml",
    job_name="buildml-torchrun-ddp",
    namespace="default",
    image="pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime",
    nnodes=2,
    nproc_per_node=2,
    script_path="/workspace/train.py",
)
# Apply with kubectl yourself — not live multi-cluster orchestration.
```

Also see example templates under `deploy/k8s` when present in the repo.

---

## AI operator note

Pack/export helpers may appear on the AI tool allowlist
(`pack_torchserve`, `prepare_tensorrt_export`, `emit_k8s_ddp_job`).
**`serve_bundle` is not an AI tool** — keep process binding under human/CLI
control ([ai-tools](ai-tools-operator-patterns.md)).

---

## Failure modes / limits

| Limit | Honesty |
| --- | --- |
| Managed cloud | Not provided |
| TLS termination | Bring your own proxy |
| TorchServe / TRT / K8s | Recipes/templates only |
| Missing serve extra | `MissingExtraError` |
| Public bind without keys | `ValidationError` |

---

## Related

- [Artifacts](artifacts-checkpoints-bundles.md)
- [Torch deep](torch-deep.md)
- [Classical end-to-end](classical-end-to-end.md)
