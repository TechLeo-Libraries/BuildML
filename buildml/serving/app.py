"""Put a saved bundle behind an HTTP endpoint, honestly scoped.

The gap between "the model works in my notebook" and "the model answers
requests" is mostly plumbing: load the artifact once, accept JSON, rebuild a
frame with the columns the model expects, apply the same preprocessing that was
fitted at training time, and return predictions. This module is that plumbing.

Be clear about what it is not. This is a library that builds a FastAPI app; it
is not a managed serving product. There is no identity provider, no certificate
management, no autoscaling, no request queue, and no model registry. The
optional API-key middleware is a shared-secret check, useful for keeping a
colleague on the same network from hitting the wrong endpoint, and no substitute
for authentication at a reverse proxy. Bind to localhost, or put a proxy that
terminates TLS in front of it. The ``/health`` endpoint says so in its own
response, so nobody discovers it later.

Two bundle kinds are supported: a classical pipeline bundle, which carries the
fitted preprocessing plans and applies them per request, and a TorchScript
module, which takes pre-built numeric tensors.

See Also
--------
buildml.serving.launch : Running the app in a background thread.
buildml.pipeline.bundle : The artifact being served.
"""

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
    """The one loaded bundle this process serves from.

    Loading a model is expensive and must not happen per request, so the
    artifact is read once at startup and held here. Exactly one of
    ``pipeline_bundle`` or ``torchscript_module`` is populated, according to
    ``kind``.

    Attributes
    ----------
    kind:
        ``'pipeline'`` or ``'torchscript'``, which decides how a request body is
        interpreted and which predict path runs.
    path:
        Where the artifact was loaded from, reported by ``/health`` so a running
        server can be traced back to a file on disk.
    pipeline_bundle:
        The loaded pipeline bundle, or ``None`` for TorchScript.
    torchscript_module:
        The loaded TorchScript module in eval mode, or ``None`` for a pipeline.
    title:
        The name shown in the API docs and health response.

    Notes
    -----
    **The state is module-global and therefore per process.** This is what makes
    the app importable by an ASGI server, and it also means a worker pool loads
    the bundle once per worker. Size the memory accordingly.

    See Also
    --------
    load_serving_bundle : Constructing this from a path.
    """

    def __init__(
        self,
        *,
        kind: BundleKind,
        path: Path,
        pipeline_bundle: Any | None = None,
        torchscript_module: Any | None = None,
        title: str = "BuildML Serve",
    ) -> None:
        """Hold an already-loaded artifact and the metadata describing it.

        Does no loading or validation of its own: the caller has already read
        the artifact and knows which kind it is.

        Parameters
        ----------
        kind:
            Which of the two artifact kinds this holds.
        path:
            The resolved artifact path, for reporting.
        pipeline_bundle:
            The loaded bundle when ``kind`` is ``'pipeline'``.
        torchscript_module:
            The loaded module when ``kind`` is ``'torchscript'``. Should already
            be in eval mode.
        title:
            The display name for docs and health.
        """
        self.kind = kind
        self.path = path
        self.pipeline_bundle = pipeline_bundle
        self.torchscript_module = torchscript_module
        self.title = title


_STATE: ServingState | None = None


def get_serving_state() -> ServingState:
    """Return the loaded bundle, refusing if the app was never initialised.

    Every request handler goes through this rather than touching the module
    global, so a request arriving before startup finished fails with an
    explanation instead of an ``AttributeError`` on ``None``.

    Returns
    -------
    ServingState
        The bundle this process is serving.

    Raises
    ------
    ValidationError
        If no bundle has been loaded. In normal use this cannot happen, since
        :func:`create_serving_app` loads before returning the app; it indicates
        the state was cleared, or a handler was called outside an app.
    """
    if _STATE is None:
        raise ValidationError("Serving app has no loaded bundle state")
    return _STATE


def set_serving_state(state: ServingState) -> None:
    """Install ``state`` as the bundle this process serves, replacing any prior.

    Called by :func:`create_serving_app` after loading. Exposed for tests and
    for hot-swapping a model in a long-running process.

    Parameters
    ----------
    state:
        The loaded bundle to serve.

    Notes
    -----
    **Replacement is not synchronised with in-flight requests.** A request that
    has already read the old state finishes against it, which is fine for a swap
    between versions of the same model and not for a swap to a different schema.
    """
    global _STATE
    _STATE = state


def clear_serving_state() -> None:
    """Drop the loaded bundle, so the next request fails rather than guessing.

    Primarily for tests, which need each case to start from a known-empty
    process. Also releases the reference to a large model.

    Notes
    -----
    **The app object remains valid after clearing**, but every request will
    raise until a new state is installed.
    """
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
    trusted: bool = False,
) -> ServingState:
    """Read an artifact from disk once, ready to answer requests from memory.

    Loading is the expensive step and belongs at startup, not in a handler. For
    a pipeline bundle this reads the estimator, the fitted preprocessing plans,
    and the model card. For TorchScript it loads the module and puts it in eval
    mode, which matters: dropout and batch-norm behave differently in training
    mode, and a module left in training mode returns subtly wrong answers rather
    than failing.

    Parameters
    ----------
    path:
        The artifact location. For a pipeline, the bundle directory. For
        TorchScript, either the file itself or a directory containing
        ``model.ts.pt``, ``model.pt``, or ``model.ts``, tried in that order.
    kind:
        Which artifact kind to expect. Defaults to ``'pipeline'``.
    map_location:
        Where to place TorchScript tensors: ``'cpu'`` (the default),
        ``'cuda'``, or a specific device. Ignored for pipelines. Loading a
        GPU-saved module onto a CPU-only host needs this left at ``'cpu'``.
    trusted:
        Must be ``True`` to deserialize pickle/joblib/TorchScript payloads.
        Pass only for artifacts you created or fully trust.

    Returns
    -------
    ServingState
        The loaded artifact and its metadata.

    Raises
    ------
    ValidationError
        If the pipeline directory does not exist, if no TorchScript file is
        found at or under the path, or if ``kind`` is neither supported value.
    MissingExtraError
        If TorchScript serving is requested without PyTorch installed.

    Notes
    -----
    **This does not start a server.** It only loads, which makes it usable for
    validating an artifact in CI without binding a port.

    See Also
    --------
    create_serving_app : Loading and building the app in one step.
    """
    root = Path(path)
    if kind == "pipeline":
        from buildml.pipeline.bundle import load_pipeline_bundle

        if not root.exists():
            raise ValidationError(f"Pipeline bundle path does not exist: {root}")
        bundle = load_pipeline_bundle(root, trusted=trusted)
        return ServingState(kind="pipeline", path=root, pipeline_bundle=bundle)
    if kind == "torchscript":
        from buildml.core.serialization import require_trusted_deserialize
        from buildml.dl.extras import require_torch

        require_trusted_deserialize(
            trusted=trusted, artifact="TorchScript module", path=root
        )
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
    trusted: bool = False,
) -> Any:
    """Load a bundle and wrap it in a FastAPI app with five endpoints.

    The app exposes ``/health`` for liveness and self-description, ``/metadata``
    for the model card and schema contract, ``/predict`` and ``/predict/batch``
    for inference, and ``/docs`` for the generated OpenAPI page. The bundle is
    loaded before the app is returned, so a bad artifact fails at construction
    rather than on the first request.

    Request bodies differ by kind. A pipeline takes ``{"rows": [{col: value,
    ...}, ...]}``: records, by column name, with the fitted preprocessing
    applied server-side. TorchScript takes ``{"inputs": [[...], ...]}``: a
    batch of numeric vectors, already in the order and scale the module expects,
    because a TorchScript module carries no preprocessing.

    Parameters
    ----------
    path:
        The artifact to serve.
    kind:
        ``'pipeline'`` or ``'torchscript'``.
    title:
        The name shown in the docs page and health response. Worth setting when
        several models run side by side.
    map_location:
        Device placement for TorchScript.
    api_keys:
        One key or several. When given, every request must present a matching
        key; when ``None``, the app is unauthenticated and must not be reachable
        from an untrusted network.
    trusted:
        Must be ``True`` to deserialize the served artifact. Operators own this
        opt-in; the default refuses pickle/joblib/TorchScript loads.

    Returns
    -------
    Any
        A ``FastAPI`` application, ready for ``uvicorn`` or a test client.
        Typed loosely because FastAPI is an optional dependency and cannot be
        named in the signature.

    Raises
    ------
    MissingExtraError
        If FastAPI is not installed. Install with ``pip install
        'buildml[serve]'``.
    ValidationError
        If the artifact cannot be loaded, or if ``api_keys`` contains a key that
        is empty or too short.

    Notes
    -----
    **The API-key middleware is a shared secret, not authentication.** It has no
    identities, no rotation, no expiry, and no audit trail. It is a guard
    against accidental access, not against an attacker. For anything reachable
    beyond localhost, terminate TLS and authenticate at a reverse proxy.

    **Prediction errors become 400 or 500 by intent.** A
    :class:`~buildml.core.errors.ValidationError` means the request was wrong
    and returns 400; anything else returns 500. Both include the message, which
    is convenient locally and worth suppressing at a proxy if the endpoint is
    exposed.

    **The bundle loads once per process.** Running several uvicorn workers loads
    it once per worker.

    Examples
    --------
    Serve locally with a key, and check it end to end::

        app = create_serving_app(
            "artifacts/churn-pipeline",
            title="Churn v3",
            api_keys="local-dev-key",
            trusted=True,
        )

        from fastapi.testclient import TestClient
        client = TestClient(app)
        headers = {"Authorization": "Bearer local-dev-key"}
        client.get("/health", headers=headers).json()["ok"]
        client.post(
            "/predict",
            headers=headers,
            json={"rows": [{"tenure": 12, "monthly_charges": 79.9}]},
        ).json()["predictions"]

    See Also
    --------
    buildml.serving.launch.serve_bundle : Doing this and starting a server.
    """
    _require_fastapi()
    state = load_serving_bundle(
        path, kind=kind, map_location=map_location, trusted=trusted
    )
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

    def _run_predict(payload: dict[str, Any]) -> dict[str, Any]:
        st = get_serving_state()
        if st.kind == "pipeline":
            return _predict_pipeline(st, payload)
        return _predict_torchscript(st, payload)

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
                "Prefer TLS at a reverse proxy. Optional local HTTPS via "
                "ssl_certfile/ssl_keyfile is library-owned: still not managed certs."
            ),
            "endpoints": ["/health", "/metadata", "/predict", "/predict/batch", "/docs"],
            "predict_contract": (
                "pipeline: {rows|instances:[...]} | torchscript: {inputs:[[...],...]}"
            ),
        }

    @app.get("/metadata")
    def metadata() -> dict[str, Any]:
        st = get_serving_state()
        body: dict[str, Any] = {
            "ok": True,
            "kind": st.kind,
            "path": str(st.path),
            "title": st.title,
            "auth": auth_enabled,
            "disclosures": (
                "Metadata endpoint for local managed serve completeness.",
                "Still not a managed model registry or cloud IAM product.",
            ),
        }
        if st.kind == "pipeline" and st.pipeline_bundle is not None:
            card = getattr(st.pipeline_bundle, "model_card", None)
            if card is not None and hasattr(card, "to_dict"):
                body["model_card"] = card.to_dict()
            card_json = Path(st.path) / "model_card.json"
            if "model_card" not in body and card_json.is_file():
                import json

                try:
                    body["model_card"] = json.loads(card_json.read_text(encoding="utf-8"))
                except Exception as exc:  # noqa: BLE001
                    body["model_card_warning"] = str(exc)
            contract = getattr(st.pipeline_bundle, "contract", None)
            if contract is not None and hasattr(contract, "to_dict"):
                body["schema_contract"] = contract.to_dict()
        return body

    @app.post("/predict")
    def predict(payload: dict[str, Any]) -> Any:
        try:
            return _run_predict(payload)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/predict/batch")
    def predict_batch(payload: dict[str, Any]) -> Any:
        try:
            result = _run_predict(payload)
            result["batch"] = True
            return result
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
