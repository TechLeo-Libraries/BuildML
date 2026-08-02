"""FastAPI application factory for BuildML managed model serving."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd

from buildml.core.errors import MissingExtraError, ValidationError

BundleKind = Literal["pipeline", "torchscript"]

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
except ImportError:  # pragma: no cover - exercised when serve extra missing
    FastAPI = None  # type: ignore[assignment]
    HTTPException = None  # type: ignore[assignment]
    JSONResponse = None  # type: ignore[assignment]


class ServingState:
    """Process-local loaded bundle for the serving app."""

    def __init__(
        self,
        *,
        kind: BundleKind,
        path: Path,
        pipeline_bundle: Any | None = None,
        torchscript_module: Any | None = None,
        title: str = "BuildML Serve",
    ) -> None:
        self.kind = kind
        self.path = path
        self.pipeline_bundle = pipeline_bundle
        self.torchscript_module = torchscript_module
        self.title = title


_STATE: ServingState | None = None


def get_serving_state() -> ServingState:
    if _STATE is None:
        raise ValidationError("Serving app has no loaded bundle state")
    return _STATE


def set_serving_state(state: ServingState) -> None:
    global _STATE
    _STATE = state


def clear_serving_state() -> None:
    global _STATE
    _STATE = None


def _require_fastapi() -> None:
    if FastAPI is None:
        raise MissingExtraError("serve", "Managed model serving")


def load_serving_bundle(
    path: str | Path,
    *,
    kind: BundleKind = "pipeline",
    map_location: str = "cpu",
) -> ServingState:
    """Load a classical pipeline or TorchScript artifact for serving."""
    root = Path(path)
    if kind == "pipeline":
        from buildml.pipeline.bundle import load_pipeline_bundle

        if not root.exists():
            raise ValidationError(f"Pipeline bundle path does not exist: {root}")
        bundle = load_pipeline_bundle(root)
        return ServingState(kind="pipeline", path=root, pipeline_bundle=bundle)
    if kind == "torchscript":
        from buildml.dl.extras import require_torch

        torch = require_torch(feature="TorchScript serving")
        ts_path = root
        if root.is_dir():
            candidates = [
                root / "model.ts.pt",
                root / "model.pt",
                root / "model.ts",
            ]
            ts_path = next((c for c in candidates if c.exists()), root)
        if not ts_path.exists() or not ts_path.is_file():
            raise ValidationError(
                f"TorchScript file not found at {ts_path}. Pass a .pt/.ts file "
                "or a directory containing model.ts.pt."
            )
        module = torch.jit.load(str(ts_path), map_location=map_location)
        module.eval()
        return ServingState(
            kind="torchscript",
            path=ts_path,
            torchscript_module=module,
        )
    raise ValidationError("kind must be 'pipeline' or 'torchscript'")


def create_serving_app(
    path: str | Path,
    *,
    kind: BundleKind = "pipeline",
    title: str = "BuildML Serve",
    map_location: str = "cpu",
    api_keys: str | list[str] | tuple[str, ...] | None = None,
) -> Any:
    """Create a FastAPI app that serves health + predict for a bundle.

    Security honesty
    ----------------
    * Auth is **optional** library middleware (API key / Bearer), not a managed
      IAM / cloud identity product.
    * Intended for localhost or reverse-proxy fronted deployments (prefer TLS
      + auth at the proxy for internet exposure).
    * Do not expose bare to the public internet without a reverse proxy.
    """
    _require_fastapi()
    state = load_serving_bundle(path, kind=kind, map_location=map_location)
    state.title = title
    set_serving_state(state)

    from buildml.serving.auth import APIKeyAuthMiddleware, normalize_api_keys

    keys = normalize_api_keys(api_keys) if api_keys is not None else frozenset()
    auth_enabled = bool(keys)

    app = FastAPI(
        title=title,
        docs_url="/docs",
        redoc_url=None,
        description=(
            "BuildML managed serving (alpha). Localhost-oriented. "
            "Optional API-key/Bearer middleware when configured; still not a "
            "managed cloud. Put a reverse proxy (TLS) in front for non-local exposure."
        ),
    )
    if auth_enabled:
        app.add_middleware(APIKeyAuthMiddleware, api_keys=keys)

    @app.get("/health")
    def health() -> dict[str, Any]:
        st = get_serving_state()
        return {
            "ok": True,
            "product": "buildml-serve",
            "title": st.title,
            "kind": st.kind,
            "path": str(st.path),
            "auth": auth_enabled,
            "auth_mode": "api_key_bearer" if auth_enabled else None,
            "bind_recommendation": "127.0.0.1",
            "tls_note": (
                "Terminate TLS at a reverse proxy; this process does not manage certificates."
            ),
        }

    @app.post("/predict")
    def predict(payload: dict[str, Any]) -> Any:
        st = get_serving_state()
        try:
            if st.kind == "pipeline":
                return _predict_pipeline(st, payload)
            return _predict_torchscript(st, payload)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def _predict_pipeline(state: ServingState, payload: dict[str, Any]) -> dict[str, Any]:
    from buildml.pipeline.score import predict_from_pipeline

    rows = payload.get("rows")
    if rows is None and "instances" in payload:
        rows = payload["instances"]
    if not isinstance(rows, list) or not rows:
        raise ValidationError(
            "POST /predict for pipeline bundles expects JSON "
            '{"rows": [ {"col": value, ...}, ... ]}'
        )
    frame = pd.DataFrame(rows)
    return_proba = bool(payload.get("return_proba", False))
    result = predict_from_pipeline(
        state.pipeline_bundle,
        frame,
        return_proba=return_proba,
        apply_plans=bool(payload.get("apply_plans", True)),
    )
    body: dict[str, Any] = {
        "ok": True,
        "kind": "pipeline",
        "n_rows": result.n_rows,
        "task": result.task,
        "predictions": result.predictions.tolist(),
        "warnings": list(result.warnings),
    }
    if result.probabilities is not None:
        body["probabilities"] = result.probabilities.to_dict(orient="list")
    return body


def _predict_torchscript(state: ServingState, payload: dict[str, Any]) -> dict[str, Any]:
    from buildml.dl.extras import require_torch

    torch = require_torch(feature="TorchScript serving predict")
    raw = payload.get("inputs")
    if raw is None:
        raise ValidationError(
            "POST /predict for torchscript expects JSON "
            '{"inputs": [[...], ...]}  (batched float features)'
        )
    tensor = torch.as_tensor(raw, dtype=torch.float32)
    module = state.torchscript_module
    assert module is not None
    with torch.no_grad():
        out = module(tensor)
    if hasattr(out, "detach"):
        values = out.detach().cpu().tolist()
    else:
        values = out
    return {
        "ok": True,
        "kind": "torchscript",
        "n_rows": int(tensor.shape[0]) if hasattr(tensor, "shape") else None,
        "outputs": values,
    }
