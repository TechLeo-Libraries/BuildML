"""Real two-rank CPU DDP updates match independent globally averaged Adam steps."""

import os
import pickle
import socket

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.dl.ddp import DDPConfig, train_supervised_module_ddp
from buildml.dl.loaders import make_loaders
from buildml.dl.types import LoaderConfig, TrainConfig


def linear_factory():
    import torch

    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.1, -0.2]]))
        model.bias.fill_(0.05)
    return model


def bundle(shuffle):
    x = np.asarray(
        [
            [-1.0, 0.5],
            [0.2, -0.3],
            [0.7, 0.8],
            [-0.4, -0.2],
            [1.1, 0.1],
            [-0.8, 0.9],
            [0.4, -1.0],
            [0.9, 0.6],
            [1.5, -0.7],
            [-1.2, 0.2],
            [2.0, 1.0],
            [-2.0, -0.5],
        ]
    )
    y = x @ np.asarray([1.7, -0.4]) + 0.3
    frame = pd.DataFrame({"x": x[:, 0], "z": x[:, 1], "target": y})
    s = Session.ingest(frame).set_roles({"x": "feature", "z": "feature", "target": "target"})
    s.inject_split(train_indices=list(range(8)), validation_indices=[8, 9], test_indices=[10, 11])
    return (
        make_loaders(
            s.dataset,
            s.split_plan,
            task="regression",
            config=LoaderConfig(batch_size=4, shuffle_train=shuffle, normalize=False),
        ),
        x,
        y,
    )


def configuration(epochs):
    return TrainConfig(
        epochs=epochs, learning_rate=0.03, device="cpu", seed=11, restore_best_weights=False
    )


def multi_worker(rank, port, epochs, shuffle, destination):
    import torch

    os.environ.update(
        WORLD_SIZE="2",
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    loaders, _, _ = bundle(shuffle)
    result = train_supervised_module_ddp(
        linear_factory,
        loaders,
        config=configuration(epochs),
        ddp_config=DDPConfig(multi_node=True, backend="gloo", allow_cpu_ddp=True),
    )
    assert not torch.distributed.is_initialized()
    if rank == 0:
        with open(destination, "wb") as stream:
            pickle.dump(result, stream)
    else:
        assert result.train_result is None


def adam_oracle(x, y, epochs):
    design = np.column_stack([x[:8], np.ones(8)])
    theta = np.asarray([0.1, -0.2, 0.05])
    first = np.zeros(3)
    second = np.zeros(3)
    for step in range(1, epochs + 1):
        gradient = 2 * design.T @ (design @ theta - y[:8]) / 8
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient**2
        theta -= 0.03 * (first / (1 - 0.9**step)) / (np.sqrt(second / (1 - 0.999**step)) + 1e-8)
    return theta


@pytest.mark.parametrize("mode", ["single", "torchrun_env"])
@pytest.mark.parametrize("epochs", [1, 3])
@pytest.mark.parametrize("shuffle", [False, True])
def test_two_rank_update_matches_global_gradient_oracle(mode, epochs, shuffle, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("Actual Torch Gloo backend required")
    if torch.cuda.is_available():
        pytest.skip("This numerical contract targets the CPU-only Gloo runtime")
    loaders, x, y = bundle(shuffle)
    loader = loaders.loaders["train"]
    generator_state = loader.generator.get_state().clone() if loader.generator is not None else None
    children = {child.pid for child in torch.multiprocessing.active_children()}
    with socket.socket() as available:
        available.bind(("127.0.0.1", 0))
        port = available.getsockname()[1]
    if mode == "single":
        result = train_supervised_module_ddp(
            linear_factory,
            loaders,
            config=configuration(epochs),
            ddp_config=DDPConfig(
                world_size=2,
                backend="gloo",
                allow_cpu_ddp=True,
                master_addr="127.0.0.1",
                master_port=str(port),
            ),
        )
    else:
        path = tmp_path / "rank-zero.pkl"
        context = torch.multiprocessing.start_processes(
            multi_worker,
            args=(port, epochs, shuffle, str(path)),
            nprocs=2,
            join=False,
            start_method="spawn",
        )
        import time

        deadline = time.monotonic() + 120
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                for process in context.processes:
                    process.join(timeout=10)
                pytest.fail("Own torchrun-env workers did not finish within120 seconds")
        with path.open("rb") as stream:
            result = pickle.load(stream)
    expected = adam_oracle(x, y, epochs)
    model = result.train_result.module
    actual = np.r_[model.weight.detach().numpy().ravel(), model.bias.detach().numpy()]
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    probe = torch.tensor(x[10:], dtype=torch.float32)
    with torch.no_grad():
        prediction = model(probe).numpy().ravel()
    np.testing.assert_allclose(
        prediction, x[10:] @ expected[:2] + expected[2], rtol=2e-5, atol=2e-6
    )
    validation_loss = np.mean((x[8:10] @ expected[:2] + expected[2] - y[8:10]) ** 2)
    assert result.train_result.history[-1]["val_loss"] == pytest.approx(validation_loss, rel=2e-5)
    assert result.world_size == 2 and result.backend == "gloo" and not result.device_ids
    assert result.train_result.n_epochs_ran == epochs
    assert result.train_result.optimizer_state is None
    assert not torch.distributed.is_initialized()
    assert {child.pid for child in torch.multiprocessing.active_children()} == children
    assert loaders.loaders["train"] is loader
    if generator_state is not None:
        assert torch.equal(loader.generator.get_state(), generator_state)
