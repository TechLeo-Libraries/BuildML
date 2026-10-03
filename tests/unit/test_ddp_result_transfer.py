"""The parent can use trained tensors after spawned DDP workers exit."""

import dataclasses

import pytest

from scripts.check_documentation_examples import ROOT, execute, inventory


def test_spawned_ddp_returns_usable_model_and_history():
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("This Torch build has no Gloo backend")
    example = next(
        example for example in inventory([ROOT / "buildml/dl/ddp.py"])
        if "result = train_supervised_module_ddp(" in example.source
    )
    source = example.source + "\n" + "\n".join([
        "if __name__ == '__main__':",
        "    assert result.world_size == 2",
        "    assert result.train_result.n_epochs_ran == 1",
        "    model = result.train_result.module",
        "    output = model(torch.ones(3, 4))",
        "    assert output.shape == (3, 2)",
        "    assert torch.isfinite(output).all()",
    ])
    result = execute(dataclasses.replace(example, source=source), timeout=180)
    assert result["status"] == "passed", result


@pytest.mark.parametrize("settings", [
    {"early_stopping_patience": 1},
    {"scheduler": "plateau"},
])
def test_adaptive_ddp_rejected_before_launch(monkeypatch, settings):
    from buildml.core.errors import ValidationError
    from buildml.dl import ddp
    from buildml.dl.types import TrainConfig

    monkeypatch.setattr(ddp, "require_torch", lambda **kwargs: object())
    def unexpected_launch(*args, **kwargs):
        pytest.fail("Unsupported configuration launched workers")
    monkeypatch.setattr(ddp, "_train_single_node", unexpected_launch)
    monkeypatch.setattr(ddp, "_train_multi_node", unexpected_launch)
    for multi_node in (False, True):
        with pytest.raises(ValidationError, match="synchronized metrics"):
            ddp.train_supervised_module_ddp(
                lambda: None, None, config=TrainConfig(**settings),
                ddp_config=ddp.DDPConfig(multi_node=multi_node),
            )

def test_validation_with_buffers_completes_on_all_ranks():
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("This Torch build has no Gloo backend")
    example = next(
        example for example in inventory([ROOT / "buildml/dl/ddp.py"])
        if "result = train_supervised_module_ddp(" in example.source
    )
    source = example.source.replace(
        "return torch.nn.Linear(4, 2)",
        "return torch.nn.Sequential(torch.nn.BatchNorm1d(4), torch.nn.Linear(4, 2))",
    ).replace("test_size=0.2, stratify=True", "test_size=0.2, validation_size=0.2, stratify=True")
    source += "\nif __name__ == '__main__':\n    assert result.train_result.history[0]['val_loss'] >= 0\n"
    result = execute(dataclasses.replace(example, source=source), timeout=180)
    assert result["status"] == "passed", result
