"""Pass V: capability depth — zoo heads, ASR eval, serve/K8s/RAG edges, AI wiring."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import MissingExtraError, ValidationError
from buildml.explain.catalog import OPERATION_CATALOG

_TORCH_SPEC = importlib.util.find_spec("torch") is not None
_TORCHVISION_SPEC = importlib.util.find_spec("torchvision") is not None
_TRANSFORMERS_SPEC = importlib.util.find_spec("transformers") is not None
_FASTAPI_SPEC = importlib.util.find_spec("fastapi") is not None
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _tiny_frame() -> pd.DataFrame:
    return pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "y": [0, 1, 0, 1]})


def _require_torch_or_skip() -> None:
    if not _TORCH_SPEC:
        pytest.skip("torch not installed")
    try:
        from buildml.dl.extras import require_torch

        require_torch(feature="pass V capability depth")
    except (MissingExtraError, ImportError, OSError) as exc:
        pytest.skip(f"torch not importable: {exc}")


def test_catalog_covers_pass_v_ops() -> None:
    for name in (
        "attach_backbone_head",
        "evaluate_asr",
        "emit_k8s_serve_deployment",
    ):
        assert name in OPERATION_CATALOG


def test_list_pretrained_backbones_expanded() -> None:
    from buildml.dl.zoo import list_pretrained_backbones

    catalog = list_pretrained_backbones()
    assert isinstance(catalog, (list, tuple))
    archs = {
        str(item.get("architecture") if isinstance(item, dict) else getattr(item, "architecture", item))
        for item in catalog
    }
    # Expanded curated set beyond Pass R defaults.
    assert "resnet18" in archs
    assert any("vit" in a for a in archs)
    assert any("wav2vec" in a or "hubert" in a for a in archs)
    assert any("whisper" in a for a in archs)
    assert len(archs) >= 4


def test_evaluate_asr_wer_cer_and_session() -> None:
    from buildml.dl.speech import evaluate_asr

    result = evaluate_asr(
        hypotheses=["hello world", "good night"],
        references=["hello world", "good morning"],
        lowercase=True,
    )
    wer = getattr(result, "wer", None)
    cer = getattr(result, "cer", None)
    if wer is None and hasattr(result, "to_dict"):
        payload = result.to_dict()
        wer = payload.get("wer")
        cer = payload.get("cer")
    assert wer is not None and float(wer) >= 0.0
    assert cer is not None and float(cer) >= 0.0
    assert float(wer) > 0.0  # second pair differs

    session = Session.ingest(_tiny_frame()).set_roles({"a": "feature", "y": "target"})

    class _Speech:
        texts = ["hello world", "good night"]

    session._dl_speech_result = _Speech()
    scored = session.evaluate_asr(references=["hello world", "good morning"])
    assert session.dl_asr_eval is scored
    session_wer = getattr(scored, "wer", None)
    if session_wer is None and hasattr(scored, "to_dict"):
        session_wer = scored.to_dict().get("wer")
    assert session_wer is not None


def test_speech_contract_roundtrip() -> None:
    from buildml.dl.speech import SpeechContract

    contract = SpeechContract(
        audio_column="audio",
        target_column="y",
        class_labels=(0, 1),
        sample_rate=16_000,
        max_samples=8_000,
        encoder_dim=32,
    )
    assert hasattr(SpeechContract, "from_dict")
    restored = SpeechContract.from_dict(contract.to_dict())
    assert restored.audio_column == "audio"
    assert restored.target_column == "y"
    assert tuple(restored.class_labels) == (0, 1)
    assert restored.sample_rate == 16_000
    assert restored.max_samples == 8_000
    assert restored.encoder_dim == 32


def test_k8s_configmap_and_serve_deployment(tmp_path: Path) -> None:
    from buildml.dl.k8s import render_torchrun_ddp_job, write_serve_deployment

    yaml_text = render_torchrun_ddp_job(nnodes=2, include_configmap=True)
    assert "kind: ConfigMap" in yaml_text
    assert "kind: Job" in yaml_text
    assert "kind: Service" in yaml_text

    out = write_serve_deployment(tmp_path / "serve.yaml", name="buildml-serve")
    assert out.path is not None and out.path.is_file()
    text = out.path.read_text(encoding="utf-8")
    assert "kind: Deployment" in text
    assert "kind: Service" in text
    assert "buildml-serve" in text

    session = Session.ingest(_tiny_frame()).set_roles({"a": "feature", "y": "target"})
    result = session.emit_k8s_serve_deployment(tmp_path / "session-serve.yaml")
    assert result.path.is_file()
    assert any("not" in lim.lower() for lim in result.limitations)


def test_torchserve_compose_file() -> None:
    from buildml.dl import packaging as packaging_mod

    compose_path = None
    if hasattr(packaging_mod, "TORCHSERVE_COMPOSE_EXAMPLE"):
        compose_path = Path(packaging_mod.TORCHSERVE_COMPOSE_EXAMPLE)
    elif hasattr(packaging_mod, "torchserve_compose_example_path"):
        compose_path = Path(packaging_mod.torchserve_compose_example_path())
    else:
        candidates = [
            _REPO_ROOT / "deploy" / "torchserve" / "docker-compose.example.yml",
            _REPO_ROOT / "deploy" / "torchserve" / "docker-compose.example.yaml",
            _REPO_ROOT / "deploy" / "docker-compose.torchserve.example.yml",
        ]
        compose_path = next((p for p in candidates if p.is_file()), None)
        if compose_path is None and hasattr(packaging_mod, "write_torchserve_compose"):
            # Writer exists; smoke that it can emit a compose file.
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                written = packaging_mod.write_torchserve_compose(tmp)
                path = Path(getattr(written, "path", written))
                assert path.is_file()
                body = path.read_text(encoding="utf-8")
                assert "torchserve" in body.lower() or "services:" in body
                return
    assert compose_path is not None and compose_path.is_file()
    body = compose_path.read_text(encoding="utf-8")
    assert "services:" in body
    assert "torchserve" in body.lower()


def test_rag_faithfulness_and_generate_field() -> None:
    import importlib

    from buildml.rag.generate import EchoGroundedProvider, generate_from_retrieve, hits_to_citations
    from buildml.rag.results import GenerateResult, Hit, RetrieveResult

    score_fn = None
    for mod_name in ("buildml.rag.generate", "buildml.rag.faithfulness", "buildml.rag.evaluate"):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        for attr in ("score_faithfulness", "estimate_faithfulness", "faithfulness_score"):
            if hasattr(mod, attr):
                score_fn = getattr(mod, attr)
                break
        if score_fn is not None:
            break
    assert score_fn is not None, "expected a faithfulness scoring helper in rag generate/evaluate"

    hits = (
        Hit(
            chunk_id="c1",
            doc_id="d1",
            text="Paris is the capital of France.",
            score=0.9,
            rank=1,
        ),
    )
    retrieve = RetrieveResult(
        query="What is the capital of France?",
        k=1,
        hits=hits,
        embedder_id="test",
        mode="dense",
    )
    provider = EchoGroundedProvider()
    generated = generate_from_retrieve(retrieve, provider)
    assert isinstance(generated, GenerateResult)
    payload = generated.to_dict()
    assert hasattr(generated, "faithfulness") or "faithfulness" in payload

    citations = hits_to_citations(hits)
    score = score_fn(generated.answer, citations)
    assert score is not None
    if isinstance(score, (int, float)):
        numeric = float(score)
    else:
        numeric = float(getattr(score, "score", getattr(score, "faithfulness", score)))
    assert 0.0 <= numeric <= 1.0


def test_ssl_pair_validation(tmp_path: Path) -> None:
    from buildml.serving.launch import serve_bundle

    fake = tmp_path / "model.ts.pt"
    fake.write_bytes(b"not-a-real-bundle")
    cert = tmp_path / "cert.pem"
    cert.write_text("CERT", encoding="utf-8")
    with pytest.raises(ValidationError, match="ssl_"):
        serve_bundle(
            fake,
            kind="torchscript",
            host="127.0.0.1",
            port=18081,
            ssl_certfile=cert,
            ssl_keyfile=None,
        )


@pytest.mark.skipif(not _FASTAPI_SPEC, reason="fastapi not installed")
def test_serve_metadata_batch_openapi_and_ssl_cli(tmp_path: Path) -> None:
    from sklearn.linear_model import LogisticRegression
    from starlette.testclient import TestClient

    from buildml.serving.app import create_serving_app
    from buildml.serving.cli import build_parser

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
    app = create_serving_app(bundle_dir, kind="pipeline")
    client = TestClient(app)

    meta = client.get("/metadata")
    assert meta.status_code == 200
    meta_body = meta.json()
    assert "kind" in meta_body

    batch = client.post(
        "/predict/batch",
        json={"rows": [{"x1": 0.1, "x2": 0.9}, {"x1": 0.8, "x2": 0.2}]},
    )
    assert batch.status_code == 200
    batch_body = batch.json()
    assert "predictions" in batch_body or "outputs" in batch_body or "results" in batch_body

    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    schema = openapi.json()
    paths = schema.get("paths", {})
    assert "/predict" in paths
    assert "/predict/batch" in paths or any("batch" in p for p in paths)
    assert "/metadata" in paths

    parser = build_parser()
    help_text = parser.format_help()
    assert "--ssl-certfile" in help_text
    assert "--ssl-keyfile" in help_text
    args = parser.parse_args(
        [
            "--bundle",
            "bundle/",
            "--ssl-certfile",
            "cert.pem",
            "--ssl-keyfile",
            "key.pem",
        ]
    )
    assert args.ssl_certfile == "cert.pem"
    assert args.ssl_keyfile == "key.pem"


@pytest.mark.skipif(not (_TORCH_SPEC and _TORCHVISION_SPEC), reason="torchvision missing")
def test_vision_mock_attach_head() -> None:
    _require_torch_or_skip()
    session = Session.ingest(_tiny_frame()).set_roles({"a": "feature", "y": "target"})
    try:
        backbone = session.load_pretrained_backbone("vision", "resnet18", weights="mock")
    except (MissingExtraError, OSError) as exc:
        pytest.skip(f"vision backbone unavailable: {exc}")
    head = session.attach_backbone_head(n_classes=2, freeze_backbone=True)
    assert session.dl_backbone_head is head
    module = getattr(head, "module", head)
    assert hasattr(module, "forward") or callable(module)
    assert getattr(backbone, "feature_dim", 0) > 0


@pytest.mark.skipif(not (_TORCH_SPEC and _TRANSFORMERS_SPEC), reason="transformers missing")
def test_hubert_mock_backbone() -> None:
    _require_torch_or_skip()
    from buildml.dl.zoo import load_pretrained_backbone

    try:
        backbone = load_pretrained_backbone("audio", "hubert_base", weights="mock", freeze=True)
    except ValidationError as exc:
        if "Unknown" in str(exc):
            # Fallback naming used by some catalogs.
            backbone = load_pretrained_backbone("audio", "hubert", weights="mock", freeze=True)
        else:
            raise
    except (MissingExtraError, OSError) as exc:
        pytest.skip(f"hubert backbone unavailable: {exc}")
    assert backbone.modality in {"audio", "speech"}
    assert backbone.feature_dim > 0
    assert backbone.weight_mode == "mock"


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_gated_fusion_and_frozen_preprocess() -> None:
    _require_torch_or_skip()
    import numpy as np

    from buildml.dl.multimodal import build_multimodal_fusion

    rng = np.random.default_rng(0)
    n = 24
    frame = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "text": [f"row {i} token" for i in range(n)],
            "y": (rng.random(n) > 0.5).astype(int),
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"num": "feature", "text": "feature", "y": "target"})
        .split(test_size=0.25, random_state=0)
    )
    bundle = session.make_multimodal_torch_loaders(
        text_column="text",
        numeric_columns=["num"],
        batch_size=4,
        seed=0,
    )
    contract = getattr(bundle, "multimodal_contract", None)
    assert contract is not None
    # Prefer gated fusion when the builder supports fusion= / fusion_type=.
    try:
        module = build_multimodal_fusion(contract, fusion="gated")
    except TypeError:
        try:
            module = build_multimodal_fusion(contract, fusion_type="gated")
        except TypeError:
            module = build_multimodal_fusion(contract)
    assert module is not None

    preprocess = contract.to_dict() if hasattr(contract, "to_dict") else contract
    restored = session.make_multimodal_torch_loaders(
        text_column="text",
        numeric_columns=["num"],
        batch_size=4,
        seed=0,
        preprocess=preprocess,
    )
    restored_contract = getattr(restored, "multimodal_contract", None)
    assert restored_contract is not None
    if getattr(contract, "normalize_mean", None) is not None:
        assert restored_contract.normalize_mean == contract.normalize_mean


@pytest.mark.skipif(not (_TORCH_SPEC and _FASTAPI_SPEC), reason="torch/fastapi missing")
def test_torchscript_serve_predict(tmp_path: Path) -> None:
    _require_torch_or_skip()
    import torch
    from starlette.testclient import TestClient

    from buildml.serving.app import create_serving_app

    class _Tiny(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.sum(dim=-1, keepdim=True)

    module = torch.jit.script(_Tiny())
    path = tmp_path / "tiny.ts.pt"
    module.save(str(path))
    app = create_serving_app(path, kind="torchscript")
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    pred = client.post("/predict", json={"inputs": [[1.0, 2.0, 3.0]]})
    assert pred.status_code == 200


def test_pass_v_ai_tools_registered() -> None:
    from buildml.ai.tools import registered_tool_names
    from buildml.explain.sync import REQUIRED_AI_TOOL_SESSION_METHODS

    names = set(registered_tool_names())
    for required in (
        "attach_backbone_head",
        "evaluate_asr",
        "emit_k8s_serve_deployment",
    ):
        assert required in names
        assert required in REQUIRED_AI_TOOL_SESSION_METHODS


def test_deploy_k8s_examples_exist() -> None:
    multi = _REPO_ROOT / "deploy" / "k8s" / "torchrun-ddp-multinode.example.yaml"
    serve = _REPO_ROOT / "deploy" / "k8s" / "serve-deployment.example.yaml"
    assert multi.is_file()
    assert serve.is_file()
    multi_text = multi.read_text(encoding="utf-8")
    serve_text = serve.read_text(encoding="utf-8")
    assert "kind: ConfigMap" in multi_text
    assert "kind: Job" in multi_text
    assert "kind: Deployment" in serve_text
