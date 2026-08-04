"""HIGH-depth production serving: ServeConfig, Basic auth, docs gate, Docker/K8s."""

from __future__ import annotations

import base64
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.serving.config import (
    ServeConfig,
    serve_compose_example_path,
    serve_dockerfile_path,
)

_FASTAPI_SPEC = importlib.util.find_spec("fastapi") is not None
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _tiny_bundle(tmp_path: Path) -> Path:
    from sklearn.linear_model import LogisticRegression

    frame = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 0.0, 1.0, 0.2, 0.8],
            "x2": [1.0, 0.0, 1.0, 0.0, 0.7, 0.3],
            "y": [0, 1, 0, 1, 0, 1],
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, random_state=0)
        .fit(LogisticRegression(max_iter=200))
    )
    bundle_dir = tmp_path / "pipe"
    session.save_pipeline(bundle_dir)
    return bundle_dir


def test_serve_config_from_yaml_env_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    yaml_path = tmp_path / "serve.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "host: 127.0.0.1",
                "port: 9090",
                "bundle: /models/from-yaml",
                "kind: pipeline",
                "trusted: true",
                "api_keys:",
                "  - yaml-key",
                "docs_enabled: true",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BUILDML_SERVE_PORT", "9191")
    monkeypatch.setenv("BUILDML_API_KEY", "env-key")
    monkeypatch.delenv("BUILDML_SERVE_DOCS_ENABLED", raising=False)

    cfg = ServeConfig.load(
        config_path=yaml_path,
        cli={"host": "0.0.0.0", "basic_auth": "ops:s3cret", "docs_enabled": False},
    )
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 9191  # env over YAML
    assert cfg.bundle == "/models/from-yaml"
    assert cfg.trusted is True
    assert "env-key" in cfg.api_keys
    assert cfg.basic_auth == ("ops", "s3cret")
    assert cfg.docs_enabled is False
    assert cfg.auth_enabled is True
    assert cfg.resolved_docs_enabled() is False

    # Auto docs closed when auth on and docs_enabled unset.
    auto = ServeConfig(bundle="/m", api_keys=("k",))
    assert auto.resolved_docs_enabled() is False
    open_docs = ServeConfig(bundle="/m", api_keys=("k",), docs_enabled=True)
    assert open_docs.resolved_docs_enabled() is True
    no_auth = ServeConfig(bundle="/m")
    assert no_auth.resolved_docs_enabled() is True


def test_serve_config_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("bundle: /m\nnot_a_field: 1\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="Unknown ServeConfig"):
        ServeConfig.from_yaml(path)


def test_serve_config_to_dict_redacts_secrets() -> None:
    cfg = ServeConfig(
        bundle="/m",
        api_keys=("super-secret",),
        basic_username="u",
        basic_password="p",
    )
    body = cfg.to_dict()
    assert body["api_keys"] == ["***REDACTED***"]
    assert body["basic_password"] == "***REDACTED***"
    assert body["auth_enabled"] is True


@pytest.mark.skipif(not _FASTAPI_SPEC, reason="fastapi not installed")
def test_basic_auth_401_and_200(tmp_path: Path) -> None:
    from starlette.testclient import TestClient

    from buildml.serving.app import clear_serving_state, create_serving_app

    bundle = _tiny_bundle(tmp_path)
    try:
        app = create_serving_app(
            bundle,
            kind="pipeline",
            basic_auth=("alice", "wonderland"),
            trusted=True,
        )
        client = TestClient(app)
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["auth"] is True
        assert "basic" in (health.json().get("auth_mode") or "")
        assert health.json()["docs_enabled"] is False

        denied = client.post("/predict", json={"rows": [{"x1": 0.1, "x2": 0.9}]})
        assert denied.status_code == 401
        assert "WWW-Authenticate" in denied.headers

        token = base64.b64encode(b"alice:wonderland").decode("ascii")
        ok = client.post(
            "/predict",
            json={"rows": [{"x1": 0.1, "x2": 0.9}]},
            headers={"Authorization": f"Basic {token}"},
        )
        assert ok.status_code == 200
        assert ok.json()["ok"] is True

        wrong = base64.b64encode(b"alice:wrong").decode("ascii")
        bad = client.post(
            "/predict",
            json={"rows": [{"x1": 0.1, "x2": 0.9}]},
            headers={"Authorization": f"Basic {wrong}"},
        )
        assert bad.status_code == 401
    finally:
        clear_serving_state()


@pytest.mark.skipif(not _FASTAPI_SPEC, reason="fastapi not installed")
def test_either_api_key_or_basic_authorizes(tmp_path: Path) -> None:
    from starlette.testclient import TestClient

    from buildml.serving.app import clear_serving_state, create_serving_app

    bundle = _tiny_bundle(tmp_path)
    try:
        app = create_serving_app(
            bundle,
            api_keys=["key-a"],
            basic_auth="bob:builder",
            trusted=True,
        )
        client = TestClient(app)
        via_key = client.post(
            "/predict",
            json={"rows": [{"x1": 0.1, "x2": 0.9}]},
            headers={"X-API-Key": "key-a"},
        )
        assert via_key.status_code == 200
        token = base64.b64encode(b"bob:builder").decode("ascii")
        via_basic = client.post(
            "/predict",
            json={"rows": [{"x1": 0.1, "x2": 0.9}]},
            headers={"Authorization": f"Basic {token}"},
        )
        assert via_basic.status_code == 200
    finally:
        clear_serving_state()


@pytest.mark.skipif(not _FASTAPI_SPEC, reason="fastapi not installed")
def test_docs_closed_when_auth_enabled(tmp_path: Path) -> None:
    from starlette.testclient import TestClient

    from buildml.serving.app import clear_serving_state, create_serving_app

    bundle = _tiny_bundle(tmp_path)
    try:
        app = create_serving_app(bundle, api_keys=["secret"], trusted=True)
        client = TestClient(app)
        health = client.get("/health")
        assert health.json()["docs_enabled"] is False
        assert "/docs" not in health.json()["endpoints"]
        # Docs routes are unregistered and not on the open-path allowlist, so
        # unauthenticated probes see 401 from auth middleware (or 404 if reached).
        docs = client.get("/docs")
        assert docs.status_code in {401, 404}
        openapi = client.get("/openapi.json")
        assert openapi.status_code in {401, 404}

        open_app = create_serving_app(
            bundle, api_keys=["secret"], docs_enabled=True, trusted=True
        )
        open_client = TestClient(open_app)
        assert open_client.get("/health").json()["docs_enabled"] is True
        # Docs path itself is open when opted in; schema is reachable.
        assert open_client.get("/openapi.json").status_code == 200
    finally:
        clear_serving_state()


def test_dockerfile_and_compose_example_exist() -> None:
    dockerfile = serve_dockerfile_path()
    assert dockerfile.is_file(), f"missing {dockerfile}"
    text = dockerfile.read_text(encoding="utf-8")
    assert "buildml-serve" in text
    assert "USER buildml" in text or "USER 10001" in text or "useradd" in text
    assert "HEALTHCHECK" in text
    assert "ENTRYPOINT" in text
    # Non-comment instructions must not enable the insecure bind override.
    active = "\n".join(
        ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")
    )
    assert "--allow-insecure-public-bind" not in active

    compose = serve_compose_example_path()
    assert compose.is_file()
    compose_text = compose.read_text(encoding="utf-8")
    assert "BUILDML_API_KEY" in compose_text
    assert "--allow-insecure-public-bind" not in compose_text


def test_k8s_render_without_insecure_flags_by_default() -> None:
    from buildml.dl.k8s import render_serve_deployment

    yaml_text = render_serve_deployment(name="buildml-serve")
    assert "kind: Deployment" in yaml_text
    assert "kind: Service" in yaml_text
    assert "kind: Secret" in yaml_text
    assert "secretKeyRef" in yaml_text
    assert "readinessProbe" in yaml_text
    assert "livenessProbe" in yaml_text
    assert "buildml-serve:local" in yaml_text
    assert "deploy/serve/Dockerfile" in yaml_text
    # Flag must not appear as a CLI argument to buildml-serve.
    assert "--allow-insecure-public-bind" not in yaml_text
    assert "--trusted" in yaml_text
    assert "runAsNonRoot: true" in yaml_text

    example = _REPO_ROOT / "deploy" / "k8s" / "serve-deployment.example.yaml"
    assert example.is_file()
    example_text = example.read_text(encoding="utf-8")
    assert "--allow-insecure-public-bind" not in example_text
    assert "secretKeyRef" in example_text
    assert "readinessProbe" in example_text
    assert "buildml-serve:local" in example_text


def test_serve_cli_config_and_basic_auth_flags() -> None:
    from buildml.serving.cli import build_parser

    parser = build_parser()
    help_text = parser.format_help()
    assert "--config" in help_text
    assert "--basic-auth" in help_text
    assert "--docs" in help_text
    assert "--no-docs" in help_text
    args = parser.parse_args(
        [
            "--bundle",
            "bundle/",
            "--basic-auth",
            "u:p",
            "--docs",
            "--api-key",
            "k1",
        ]
    )
    assert args.basic_auth == "u:p"
    assert args.docs_enabled is True
    assert args.api_keys == ["k1"]


def test_public_bind_accepts_basic_auth(tmp_path: Path) -> None:
    from buildml.serving.launch import _ensure_bind_security

    _ensure_bind_security(
        "0.0.0.0",
        api_keys=None,
        basic_auth=("u", "p"),
        allow_insecure_public_bind=False,
    )
    with pytest.raises(ValidationError, match="api_keys or basic_auth"):
        _ensure_bind_security(
            "0.0.0.0",
            api_keys=None,
            basic_auth=None,
            allow_insecure_public_bind=False,
        )


def test_dl_capability_matrix_serving_notes() -> None:
    from buildml.dl.catalog import dl_capability_matrix

    matrix = dl_capability_matrix()
    assert "serving" in matrix
    assert matrix["serving"]["dockerfile"] == "deploy/serve/Dockerfile"
    assert "http_basic" in matrix["serving"]["auth"]
