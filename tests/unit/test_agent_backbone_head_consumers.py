"""Offline actual audio encoders verify agent heads, gradients and count guards."""

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry
from buildml.core.errors import ValidationError
from buildml.dl.zoo import attach_backbone_head


@pytest.fixture
def runtime():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    return torch, transformers


def _call(session, name, args):
    registry = build_default_registry()
    return execute_tool(session, propose_tool_execution(name, args, registry), confirmed=True, registry=registry)


def _backbone(runtime, path, architecture="wav2vec2_base", freeze=True):
    torch, transformers = runtime
    config_cls = transformers.Wav2Vec2Config if architecture == "wav2vec2_base" else transformers.HubertConfig
    config = config_cls(hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=16, conv_dim=(4, 4), conv_stride=(2, 2), conv_kernel=(3, 3),
        num_conv_pos_embeddings=4, num_conv_pos_embedding_groups=2,
        mask_time_prob=0.0, hidden_dropout=0.0, attention_dropout=0.0,
        feat_proj_dropout=0.0, final_dropout=0.0, feat_extract_norm="layer")
    config.save_pretrained(path)
    session = Session.ingest(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
    torch.manual_seed(17)
    loaded = _call(session, "load_pretrained_backbone", dict(modality="audio", architecture=architecture,
        model_id=str(path), weights="none", freeze=freeze, seed=17))
    assert loaded.executed, loaded.error
    assert loaded.result.weight_mode == "none"
    assert loaded.result.meta["model_id"] == str(path)
    return session, loaded.result


@pytest.mark.parametrize("architecture", ["wav2vec2_base", "hubert_base"])
@pytest.mark.parametrize("freeze", [False, True])
def test_agent_real_encoder_head_matches_numpy_and_frozen_gradient_contract(runtime, tmp_path, architecture, freeze):
    torch, _ = runtime
    session, backbone = _backbone(runtime, tmp_path, architecture, freeze=not freeze)
    outcome = _call(session, "attach_backbone_head", dict(n_classes=3, freeze_backbone=freeze))
    assert outcome.executed, outcome.error
    head = outcome.result
    assert head.backbone.frozen is freeze
    assert all(parameter.requires_grad is not freeze for parameter in head.module.backbone.parameters())
    samples = torch.linspace(-1, 1, 128).reshape(2, 64)
    head.module.eval()
    with torch.no_grad():
        features = backbone.module(samples).last_hidden_state.numpy()
        weight = head.module.head.weight.numpy()
        bias = head.module.head.bias.numpy()
        expected = features.mean(axis=1) @ weight.T + bias
        np.testing.assert_allclose(head.module(samples).numpy(), expected, rtol=1e-5, atol=1e-6)
    before = [p.detach().clone() for p in head.module.backbone.parameters()]
    optimizer = torch.optim.SGD([p for p in head.module.parameters() if p.requires_grad], lr=0.1)
    loss = torch.nn.functional.cross_entropy(head.module(samples), torch.tensor([0, 2]))
    optimizer.zero_grad()
    loss.backward()
    assert head.module.head.weight.grad is not None
    optimizer.step()
    changed = any(not torch.equal(a, b) for a, b in zip(before, head.module.backbone.parameters(), strict=True))
    assert changed is (not freeze)
    assert all(p.grad is None for p in head.module.backbone.parameters()) if freeze else any(
        p.grad is not None and bool(torch.any(p.grad != 0)) for p in head.module.backbone.parameters())


@pytest.mark.parametrize("path", ["direct", "session"])
@pytest.mark.parametrize("n_classes", [2.5, "3", True, float("nan"), float("inf"), 0, 1])
def test_bad_class_counts_refused_before_mutating_backbone(runtime, tmp_path, path, n_classes):
    session, backbone = _backbone(runtime, tmp_path, freeze=True)
    with pytest.raises(ValidationError, match="n_classes"):
        if path == "direct":
            attach_backbone_head(backbone, n_classes=n_classes, freeze_backbone=False)
        else:
            session.attach_backbone_head(n_classes, freeze_backbone=False)
    assert all(not p.requires_grad for p in backbone.module.parameters())
    assert getattr(session, "_dl_backbone_head", None) is None


def test_invalid_freeze_flag_refused_before_parameter_mutation(runtime, tmp_path):
    session, backbone = _backbone(runtime, tmp_path, freeze=False)
    with pytest.raises(ValidationError, match="freeze_backbone"):
        session.attach_backbone_head(3, freeze_backbone="false")
    assert all(p.requires_grad for p in backbone.module.parameters())


def test_session_keeps_latest_freeze_state_when_replacing_head(runtime, tmp_path):
    session, _ = _backbone(runtime, tmp_path, freeze=True)
    first = session.attach_backbone_head(np.int64(3), freeze_backbone=False)
    assert first.backbone.frozen is False
    second = session.attach_backbone_head(4)
    assert second.backbone.frozen is False
    assert all(p.requires_grad for p in second.module.backbone.parameters())
