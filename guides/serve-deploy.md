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
- Optional API-key / Bearer **and/or** HTTP Basic middleware (either may authorize)
- When auth is enabled, OpenAPI `/docs` defaults to **closed** unless
  `docs_enabled=True` / `--docs`
- Non-loopback binds require `api_keys` or `basic_auth` unless
  `allow_insecure_public_bind=True`
- Optional local HTTPS via `ssl_certfile` / `ssl_keyfile` (both required together)
- Prefer TLS at a reverse proxy for any non-local exposure
- Declarative `ServeConfig` layers: defaults ← YAML ← env ← CLI

---

## ServeConfig (YAML / env / CLI)

```yaml
# serve.yaml
host: 127.0.0.1
port: 8080
bundle: artifacts/pipeline
kind: pipeline
trusted: true
api_keys:
  - rotate-me
# basic_auth: "ops:change-me"
# docs_enabled: false   # default when auth is on
```

```bash
buildml-serve --config serve.yaml
# Env overrides (examples): BUILDML_BUNDLE, BUILDML_API_KEY,
# BUILDML_SERVE_BASIC_AUTH, BUILDML_SERVE_DOCS_ENABLED, BUILDML_SERVE_HOST
buildml-serve --bundle artifacts/pipeline --api-key "$BUILDML_API_KEY" --trusted
buildml-serve --bundle artifacts/pipeline --basic-auth "ops:change-me" --trusted
# Opt in to OpenAPI when auth is on:
buildml-serve --bundle artifacts/pipeline --api-key "$KEY" --docs --trusted
```

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
handle = session.dl.serve(
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
# After session.dl.fit + session.dl.export("artifacts/model.ts.pt", format="torchscript"):
# session.dl.serve(
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
# Refused without auth on non-loopback (unless allow_insecure_public_bind):
# session.dl.serve(
#     "artifacts/pipeline",
#     host="0.0.0.0",
#     api_keys=["rotate-me"],
#     # or basic_auth=("ops", "change-me"),
# )

# Docs stay closed when auth is on; opt in explicitly:
# session.dl.serve(
#     "artifacts/pipeline",
#     api_keys=["rotate-me"],
#     docs_enabled=True,
# )

# Emergency lab-only override (do not use in production):
# session.dl.serve(
#     "artifacts/pipeline",
#     host="0.0.0.0",
#     allow_insecure_public_bind=True,
# )

# Optional local HTTPS: both cert and key required (ValidationError otherwise):
# session.dl.serve(
#     "artifacts/pipeline",
#     ssl_certfile="cert.pem",
#     ssl_keyfile="key.pem",
#     api_keys=["dev-key"],
# )
```

API keys / Basic auth are **not** cloud IAM. Local SSL is **not** a managed cert
product. Rotate secrets; prefer a reverse proxy for production TLS.

---

## Use case C2: First-party Docker image

```bash
# From repo root:
docker build -f deploy/serve/Dockerfile -t buildml-serve:local .
docker run --rm -p 8080:8080 \
  -e BUILDML_API_KEY=rotate-me \
  -e BUILDML_BUNDLE=/models/bundle \
  -v "$PWD/artifacts/pipeline:/models/bundle:ro" \
  buildml-serve:local
# Compose example: deploy/serve/docker-compose.example.yml
```

Image runs as non-root, ships a `/health` HEALTHCHECK, and does **not** enable
the insecure public-bind override (API key / Basic auth required for `0.0.0.0`).

---

## Use case D: TorchServe directory pack + compose example

```python
# session.dl.export("artifacts/model.ts.pt", format="torchscript")
# result = session.dl.pack_torchserve(
#     "artifacts/torchserve_dir",
#     torchscript_path="artifacts/model.ts.pt",
#     model_name="buildml_model",
# )
# Operator runs TorchServe against the directory: BuildML does not start it.
```

Repo recipe for a local compose loop (operator-run; not a managed cloud):

- `deploy/torchserve/docker-compose.example.yml`

```bash
# After packing a .mar into a model-store (see session.dl.pack_torchserve / ARCHIVE.txt):
# docker compose -f deploy/torchserve/docker-compose.example.yml up
```

---

## Use case E: TensorRT trtexec plan (recipe)

```python
# session.dl.export("artifacts/model.onnx", format="onnx")
# plan = session.dl.prepare_tensorrt(
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
session.dl.emit_k8s_ddp(
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
session.dl.emit_k8s_serve(
    "artifacts/serve-deploy.yaml",
    name="buildml-serve",
    namespace="default",
    image="buildml-serve:local",  # build via deploy/serve/Dockerfile
    replicas=1,
    port=8080,
    # gpu_limit=1,  # optional GPU request when your cluster schedules GPUs
)
# Template only: replace Secret api-key, wire volumes/TLS/Ingress/RBAC yourself.
# Renderer includes readiness/liveness probes and secretKeyRef; no insecure bind flag.
```

Static example: `deploy/k8s/serve-deployment.example.yaml`.

---

## AI operator note

Pack/export helpers may appear on the AI tool allowlist
(`session.dl.pack_torchserve`, `session.dl.prepare_tensorrt`, `session.dl.emit_k8s_ddp`,
`session.dl.emit_k8s_serve`).
**`session.dl.serve` is not an AI tool**: keep process binding under human/CLI
control ([ai-tools](ai-tools-operator-patterns.md)).

---

## Failure modes / limits

| Limit | Honesty |
| --- | --- |
| Managed cloud | Not provided |
| TLS termination | Local SSL pair optional; prefer your proxy for production |
| TorchServe / TRT / K8s / Docker | Recipes/templates only (`deploy/serve`, `deploy/k8s`) |
| Missing serve extra | `MissingExtraError` |
| Public bind without auth | `ValidationError` |
| Partial SSL (`ssl_certfile` xor `ssl_keyfile`) | `ValidationError` |
| Auth on → docs closed | Override with `docs_enabled=True` / `--docs` |

---

## Related

- [Artifacts](artifacts-checkpoints-bundles.md)
- [Torch deep](torch-deep.md)
- [Classical end-to-end](classical-end-to-end.md)
