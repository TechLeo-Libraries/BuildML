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
    result = execute(dataclasses.replace(example, source=source), timeout=600, max_output_chars=64000)
    if result["status"] != "passed":
        pytest.fail(str(result), pytrace=False)


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

def test_validation_with_buffers_completes_on_all_ranks(monkeypatch):
    monkeypatch.setenv("BUILDML_DDP_TRACE", "1")
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
    result = execute(dataclasses.replace(example, source=source), timeout=600, max_output_chars=64000)
    if result["status"] != "passed":
        pytest.fail(str(result), pytrace=False)
    for rank in (0, 1):
        phases = [line.rsplit("phase=", 1)[1] for line in result["stderr"].splitlines()
                  if line.startswith(f"BuildML DDP rank={rank} phase=")]
        expected = ["initializing", "initialized", "training", "trained"]
        if rank == 0:
            expected.append("result_saved")
        expected.extend(["destroying_process_group", "process_group_destroyed", "returning"])
        assert phases == expected
    print(result["stderr"])


def test_spawn_payload_does_not_consume_or_replace_caller_generator(monkeypatch):
    torch = pytest.importorskip("torch")
    import pandas as pd

    from buildml import Session
    from buildml.dl import ddp
    from buildml.dl.loaders import make_loaders
    from buildml.dl.types import TrainConfig

    session = Session.ingest(pd.DataFrame({"x": range(40), "y": [0, 1] * 20})).set_roles({"y": "target"})
    session.split(test_size=.2, random_state=3)
    bundle = make_loaders(session.dataset, session.split_plan, task="classification")
    original_loader = bundle.loaders["train"]
    original_generator = original_loader.generator
    original_state = original_generator.get_state().clone()

    class CapturedLaunch(Exception):
        pass

    def capture(function, *, args, nprocs, join):
        payload = args[5]
        assert payload is not bundle
        assert payload.loaders["train"] is not original_loader
        assert payload.loaders["train"].dataset is original_loader.dataset
        assert payload.loaders["train"].batch_size == original_loader.batch_size
        assert payload.loaders["train"].generator is None
        assert payload.loaders["test"] is bundle.loaders["test"]
        assert nprocs == 2 and join
        raise CapturedLaunch

    monkeypatch.setattr(ddp, "ddp_cuda_device_count", lambda: 0)
    monkeypatch.setattr(torch.multiprocessing, "spawn", capture)
    with pytest.raises(CapturedLaunch):
        ddp.train_supervised_module_ddp(
            lambda: torch.nn.Linear(1, 2), bundle,
            config=TrainConfig(epochs=1, device="cpu"),
            ddp_config=ddp.DDPConfig(world_size=2, allow_cpu_ddp=True),
        )
    assert bundle.loaders["train"] is original_loader
    assert original_loader.generator is original_generator
    assert torch.equal(original_generator.get_state(), original_state)
