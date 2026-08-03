"""Pass R: pretrained zoo hooks, serve auth, TorchServe/TRT packs, K8s templates."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import MissingExtraError, ValidationError
from buildml.dl.k8s import render_torchrun_ddp_job, write_torchrun_ddp_job
from buildml.dl.packaging import pack_torchserve_model, prepare_tensorrt_export_plan
from buildml.dl.speech import refuse_foundation_model_pretrain
from buildml.explain.catalog import OPERATION_CATALOG

_TORCH_SPEC = importlib.util.find_spec("torch") is not None
_TORCHVISION_SPEC = importlib.util.find_spec("torchvision") is not None
_TRANSFORMERS_SPEC = importlib.util.find_spec("transformers") is not None
_FASTAPI_SPEC = importlib.util.find_spec("fastapi") is not None


def _tiny_frame() -> pd.DataFrame:
    return pd.DataFrame({"a": [1.0, 2.0], "y": [0, 1]})


def test_catalog_covers_pass_r_ops() -> None:
    for name in (
        "load_pretrained_backbone",
        "pack_torchserve",
        "prepare_tensorrt_export",
        "emit_k8s_ddp_job",
        "domain_adapt_speech_torch",
        "refuse_speech_foundation_pretrain",
        "serve_bundle",
    ):
        assert name in OPERATION_CATALOG


def test_refuse_foundation_model_pretrain_message() -> None:
    with pytest.raises(ValidationError, match="does not train Whisper-scale"):
        refuse_foundation_model_pretrain()


def test_session_refuse_speech_foundation_pretrain() -> None:
    session = Session.ingest(_tiny_frame()).set_roles({"a": "feature", "y": "target"})
    with pytest.raises(ValidationError, match="foundation"):
        session.refuse_speech_foundation_pretrain()


def test_pack_torchserve_and_tensorrt_plan(tmp_path: Path) -> None:
    ts = tmp_path / "model.ts.pt"
    ts.write_bytes(b"fake-torchscript-bytes")
    pack = pack_torchserve_model(ts, tmp_path / "tserve")
    assert pack.kind == "torchserve"
    assert (tmp_path / "tserve" / "model.pt").is_file()
    assert (tmp_path / "tserve" / "handler.py").is_file()
    assert (tmp_path / "tserve" / "manifest.json").is_file()
    assert any("does not" in lim.lower() or "not" in lim.lower() for lim in pack.limitations)

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"not-real-onnx")
    plan = prepare_tensorrt_export_plan(onnx_path, tmp_path / "trt")
    assert plan.kind == "tensorrt_plan"
    assert (tmp_path / "trt" / "tensorrt_plan.json").is_file()
    assert (tmp_path / "trt" / "TENSORRT.md").is_file()
    assert any("not" in lim.lower() for lim in plan.limitations)


def test_session_pack_helpers(tmp_path: Path) -> None:
    session = Session.ingest(_tiny_frame()).set_roles({"a": "feature", "y": "target"})
    ts = tmp_path / "m.pt"
    ts.write_bytes(b"abc")
    pack = session.pack_torchserve(tmp_path / "out_ts", torchscript_path=ts)
    assert pack.path.is_dir()
    onnx_path = tmp_path / "m.onnx"
    onnx_path.write_bytes(b"xyz")
    plan = session.prepare_tensorrt_export(tmp_path / "out_trt", onnx_path=onnx_path)
    assert plan.path.is_dir()


def test_k8s_torchrun_job_render(tmp_path: Path) -> None:
    yaml_text = render_torchrun_ddp_job(nnodes=2, nproc_per_node=2)
    assert "kind: Job" in yaml_text
    assert "torchrun" in yaml_text
    assert "Indexed" in yaml_text
    assert not yaml_text.startswith(" ")
    out = write_torchrun_ddp_job(tmp_path / "job.yaml", nnodes=3, nproc_per_node=1)
    assert out.path is not None and out.path.is_file()
    assert out.nnodes == 3
    # serviceAccountName must not break YAML indentation via dedent poisoning.
    with_sa = render_torchrun_ddp_job(nnodes=2, service_account="buildml-trainer")
    assert "serviceAccountName: buildml-trainer" in with_sa
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        yaml = None
    if yaml is not None:
        docs = [d for d in yaml.safe_load_all(with_sa) if d is not None]
        assert [d["kind"] for d in docs] == ["Job", "Service", "ConfigMap"]
        assert docs[0]["spec"]["template"]["spec"]["serviceAccountName"] == "buildml-trainer"
    session = Session.ingest(_tiny_frame()).set_roles({"a": "feature", "y": "target"})
    result = session.emit_k8s_ddp_job(tmp_path / "session-job.yaml", nnodes=2)
    assert result.path.is_file()
    assert any(
        "multi-cluster" in lim.lower() or "not live" in lim.lower() for lim in result.limitations
    )


def test_backbone_dispatch_rejects_unknown_architecture() -> None:
    from buildml.dl.zoo import load_pretrained_backbone

    with pytest.raises(ValidationError, match="Unknown audio architecture"):
        load_pretrained_backbone("audio", "not-a-real-arch", weights="none")
    with pytest.raises(ValidationError, match="Unknown speech architecture"):
        load_pretrained_backbone("speech", "whisper-large", weights="none")
    with pytest.raises(ValidationError, match="Unsupported weights mode"):
        load_pretrained_backbone("vision", "resnet18", weights="imagenet")  # type: ignore[arg-type]


@pytest.mark.skipif(not _FASTAPI_SPEC, reason="fastapi not installed")
def test_serving_api_key_auth(tmp_path: Path) -> None:
    from sklearn.linear_model import LogisticRegression
    from starlette.testclient import TestClient

    from buildml.serving.app import create_serving_app

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
    app = create_serving_app(bundle_dir, kind="pipeline", api_keys=["secret-key"], trusted=True)
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["auth"] is True
    denied = client.post("/predict", json={"rows": [{"x1": 0.1, "x2": 0.9}]})
    assert denied.status_code == 401
    ok = client.post(
        "/predict",
        json={"rows": [{"x1": 0.1, "x2": 0.9}]},
        headers={"Authorization": "Bearer secret-key"},
    )
    assert ok.status_code == 200
    ok2 = client.post(
        "/predict",
        json={"rows": [{"x1": 0.1, "x2": 0.9}]},
        headers={"X-API-Key": "secret-key"},
    )
    assert ok2.status_code == 200
    wrong = client.post(
        "/predict",
        json={"rows": [{"x1": 0.1, "x2": 0.9}]},
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert wrong.status_code == 401

    open_app = create_serving_app(bundle_dir, kind="pipeline", trusted=True)
    open_client = TestClient(open_app)
    open_health = open_client.get("/health")
    assert open_health.status_code == 200
    assert open_health.json()["auth"] is False
    open_ok = open_client.post(
        "/predict",
        json={"rows": [{"x1": 0.1, "x2": 0.9}]},
    )
    assert open_ok.status_code == 200


def _require_torch_or_skip() -> None:
    if not _TORCH_SPEC:
        pytest.skip("torch not installed")
    try:
        from buildml.dl.extras import require_torch

        require_torch(feature="pass R zoo")
    except (MissingExtraError, ImportError, OSError) as exc:
        pytest.skip(f"torch not importable: {exc}")


@pytest.mark.skipif(not (_TORCH_SPEC and _TORCHVISION_SPEC), reason="torchvision missing")
def test_load_vision_backbone_mock() -> None:
    _require_torch_or_skip()
    from buildml.dl.zoo import load_vision_backbone

    backbone = load_vision_backbone("resnet18", weights="mock", freeze=True, seed=0)
    assert backbone.modality == "vision"
    assert backbone.frozen is True
    assert backbone.feature_dim > 0
    assert all(not p.requires_grad for p in backbone.module.parameters())
    session = Session.ingest(_tiny_frame()).set_roles({"a": "feature", "y": "target"})
    bb = session.load_pretrained_backbone("vision", "resnet18", weights="mock")
    assert bb.weight_mode == "mock"


@pytest.mark.skipif(not (_TORCH_SPEC and _TRANSFORMERS_SPEC), reason="transformers missing")
def test_load_speech_backbone_mock() -> None:
    _require_torch_or_skip()
    from buildml.dl.zoo import load_speech_backbone

    backbone = load_speech_backbone(weights="mock", freeze=True, seed=0)
    assert backbone.modality == "speech"
    assert backbone.feature_dim > 0
    assert any("not" in lim.lower() for lim in backbone.limitations)


def test_pass_r_ai_tools_registered_and_serve_bundle_excluded() -> None:
    from buildml.ai.tools import registered_tool_names
    from buildml.explain.sync import EXPLICITLY_NON_AI_SESSION_METHODS

    names = set(registered_tool_names())
    for required in (
        "load_pretrained_backbone",
        "pack_torchserve",
        "prepare_tensorrt_export",
        "emit_k8s_ddp_job",
        "domain_adapt_speech_torch",
    ):
        assert required in names
    assert "serve_bundle" not in names
    assert "serve_bundle" in EXPLICITLY_NON_AI_SESSION_METHODS


def test_serve_refuses_public_bind_without_api_keys(tmp_path: Path) -> None:
    from buildml.serving.launch import serve_bundle

    fake = tmp_path / "model.ts.pt"
    fake.write_bytes(b"not-a-real-bundle")
    with pytest.raises(ValidationError, match="allow_insecure_public_bind|api_keys"):
        serve_bundle(fake, kind="torchscript", host="0.0.0.0", port=18080, trusted=True)


def test_serve_cli_documents_insecure_public_bind_flag() -> None:
    from buildml.serving.cli import build_parser

    parser = build_parser()
    help_text = parser.format_help()
    assert "--allow-insecure-public-bind" in help_text
    args = parser.parse_args(
        ["--bundle", "bundle/", "--host", "0.0.0.0", "--allow-insecure-public-bind"]
    )
    assert args.allow_insecure_public_bind is True
