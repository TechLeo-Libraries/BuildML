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
TensorRT (`trtexec`), and Kubernetes torchrun Jobs / serve Deployments. This is
not a managed cloud IAM / multi-cluster product.

Related: [artifacts](artifacts-checkpoints-bundles.md), [torch-deep](torch-deep.md),
[features](../docs/features.rst).

---

## Why localhost-first

Exposing ML scores without auth is a common incident. Defaults:

- Bind `127.0.0.1`
- Optional API-key / Bearer middleware
- Non-loopback binds require `api_keys` unless
  `allow_insecure_public_bind=True`
- Optional local HTTPS via `ssl_certfile` / `ssl_keyfile` (both required together)
- Prefer TLS at a reverse proxy for any non-local exposure

---

## Use case A: Serve a classical pipeline bundle

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

# Optional local HTTPS (both flags required):
# buildml-serve --bundle artifacts/pipeline --ssl-certfile cert.pem --ssl-keyfile key.pem
```

### HTTP surface

Managed serve exposes (OpenAPI at `/docs` / `/openapi.json`):

| Route | Role |
| --- | --- |
| `GET /health` | Liveness |
| `GET /metadata` | Bundle kind / contract summary |
| `POST /predict` | Single-row (or tensor) score |
| `POST /predict/batch` | Batch rows / inputs |

Example against a running pipeline server:

```bash
curl -s http://127.0.0.1:8080/metadata
curl -s -X POST http://127.0.0.1:8080/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"rows":[{"age":30,"income":55},{"age":40,"income":80}]}'
```

---

## Use case B: Serve TorchScript

```python
# After fit_torch + export_torch("artifacts/model.ts.pt", format="torchscript"):
# session.serve_bundle(
#     "artifacts/model.ts.pt",
#     kind="torchscript",
#     api_keys=["dev-key"],
# )
```

Scoring contracts differ by `kind`: do not assume pipeline JSON equals
TorchScript tensor payloads. Inspect `/metadata` and OpenAPI locally.

---

## Use case C: Auth, public bind, optional local HTTPS

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

# Optional local HTTPS: both cert and key required (ValidationError otherwise):
# session.serve_bundle(
#     "artifacts/pipeline",
#     ssl_certfile="cert.pem",
#     ssl_keyfile="key.pem",
#     api_keys=["dev-key"],
# )
```

API keys are **not** cloud IAM. Local SSL is **not** a managed cert product.
Rotate keys; prefer a reverse proxy for production TLS.

---

## Use case D: TorchServe directory pack + compose example

```python
# session.export_torch("artifacts/model.ts.pt", format="torchscript")
# result = session.pack_torchserve(
#     "artifacts/torchserve_dir",
#     torchscript_path="artifacts/model.ts.pt",
#     model_name="buildml_model",
# )
# Operator runs TorchServe against the directory: BuildML does not start it.
```

Repo recipe for a local compose loop (operator-run; not a managed cloud):

- `deploy/torchserve/docker-compose.example.yml`

```bash
# After packing a .mar into a model-store (see pack_torchserve ARCHIVE.txt):
# docker compose -f deploy/torchserve/docker-compose.example.yml up
```

---

## Use case E: TensorRT trtexec plan (recipe)

```python
# session.export_torch("artifacts/model.onnx", format="onnx")
# plan = session.prepare_tensorrt_export(
#     "artifacts/trt_plan",
#     onnx_path="artifacts/model.onnx",
#     engine_name="model.engine",
#     fp16=True,
# )
# Operator runs trtexec: BuildML does not build .engine files.
```

---

## Use case F: Kubernetes torchrun Job (ConfigMap + GPU)

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
    include_configmap=True,  # default: emits ConfigMap + Job + Service
    gpu_limit=1,             # nvidia.com/gpu requests/limits in the template
)
# Apply with kubectl yourself: not live multi-cluster orchestration.
```

Static multi-node example: `deploy/k8s/torchrun-ddp-multinode.example.yaml`.

---

## Use case G: Kubernetes serve Deployment (template)

```python
session = Session()
session.emit_k8s_serve_deployment(
    "artifacts/serve-deploy.yaml",
    name="buildml-serve",
    namespace="default",
    image="python:3.12-slim",
    replicas=1,
    port=8080,
    # gpu_limit=1,  # optional GPU request when your cluster schedules GPUs
)
# Template only: wire volumes, TLS, API keys, and RBAC yourself.
```

Static example: `deploy/k8s/serve-deployment.example.yaml`.

---

## AI operator note

Pack/export helpers may appear on the AI tool allowlist
(`pack_torchserve`, `prepare_tensorrt_export`, `emit_k8s_ddp_job`,
`emit_k8s_serve_deployment`).
**`serve_bundle` is not an AI tool**: keep process binding under human/CLI
control ([ai-tools](ai-tools-operator-patterns.md)).

---

## Failure modes / limits

| Limit | Honesty |
| --- | --- |
| Managed cloud | Not provided |
| TLS termination | Local SSL pair optional; prefer your proxy for production |
| TorchServe / TRT / K8s | Recipes/templates only |
| Missing serve extra | `MissingExtraError` |
| Public bind without keys | `ValidationError` |
| Partial SSL (`ssl_certfile` xor `ssl_keyfile`) | `ValidationError` |

---

## Related

- [Artifacts](artifacts-checkpoints-bundles.md)
- [Torch deep](torch-deep.md)
- [Classical end-to-end](classical-end-to-end.md)
