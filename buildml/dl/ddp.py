"""Single-node DistributedDataParallel training utilities (alpha).

Practical multi-process / multi-GPU entry for the supervised train loop.
Cluster / multi-node orchestration is out of scope; tests skip cleanly when
CUDA device count < 2.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from buildml.core.errors import ValidationError
from buildml.dl.extras import require_torch
from buildml.dl.results import TorchLoaderBundle, TrainResult
from buildml.dl.types import TrainConfig

ModuleFactory = Callable[[], Any]


@dataclass(slots=True)
class DDPConfig:
    """Single-node DDP launch knobs."""

    backend: Literal["gloo", "nccl", "auto"] = "auto"
    world_size: int | None = None
    master_addr: str = "127.0.0.1"
    master_port: str = "29500"
    find_unused_parameters: bool = False
    allow_cpu_ddp: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "world_size": self.world_size,
            "master_addr": self.master_addr,
            "master_port": self.master_port,
            "find_unused_parameters": self.find_unused_parameters,
            "allow_cpu_ddp": self.allow_cpu_ddp,
        }


@dataclass(slots=True)
class DDPTrainResult:
    """Rank-0 outcome from a single-node DDP training spawn."""

    train_result: TrainResult | None
    world_size: int
    backend: str
    device_ids: tuple[int, ...]
    disclosures: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_result": None if self.train_result is None else self.train_result.to_dict(),
            "world_size": self.world_size,
            "backend": self.backend,
            "device_ids": list(self.device_ids),
            "disclosures": list(self.disclosures),
            "limitations": list(self.limitations),
            "warnings": list(self.warnings),
            "meta": dict(self.meta),
        }


def ddp_cuda_device_count() -> int:
    """Return ``torch.cuda.device_count()`` or 0 when Torch/CUDA unavailable."""
    try:
        torch = require_torch(feature="DDP device probe")
    except Exception:
        return 0
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.device_count())


def resolve_ddp_backend(requested: str, *, use_cuda: bool) -> str:
    if requested == "auto":
        return "nccl" if use_cuda else "gloo"
    if requested not in {"gloo", "nccl"}:
        raise ValidationError("DDP backend must be gloo, nccl, or auto")
    if requested == "nccl" and not use_cuda:
        raise ValidationError("nccl backend requires CUDA")
    return requested


def _shard_loader(loader: Any, *, rank: int, world_size: int, seed: int) -> Any:
    torch = require_torch(feature="DDP loader shard")
    dataset = loader.dataset
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=getattr(loader, "shuffle", True) or True,
        seed=seed,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=loader.batch_size,
        sampler=sampler,
        num_workers=getattr(loader, "num_workers", 0),
        pin_memory=getattr(loader, "pin_memory", False),
        drop_last=getattr(loader, "drop_last", False),
    )


def _worker(
    rank: int,
    world_size: int,
    backend: str,
    master_addr: str,
    master_port: str,
    module_factory: ModuleFactory,
    loader_bundle: TorchLoaderBundle,
    config: TrainConfig,
    find_unused_parameters: bool,
    result_queue: Any,
) -> None:
    torch = require_torch(feature="DDP worker")
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    from buildml.dl.train import train_supervised_module

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    use_cuda = backend == "nccl" and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(rank)
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        device = f"cuda:{rank}"
    else:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        device = "cpu"

    try:
        module = module_factory().to(device)
        if use_cuda:
            module = DDP(
                module,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=find_unused_parameters,
            )
        else:
            module = DDP(module, find_unused_parameters=find_unused_parameters)

        train_loader = loader_bundle.loaders.get("train")
        if train_loader is None:
            raise ValidationError("DDP requires a train loader")
        sharded = dict(loader_bundle.loaders)
        sharded["train"] = _shard_loader(
            train_loader, rank=rank, world_size=world_size, seed=config.seed
        )
        # Validation is rank-0 only to avoid duplicated metric noise.
        if rank != 0:
            sharded.pop("validation", None)
        local_bundle = TorchLoaderBundle(
            loaders=sharded,
            contract=loader_bundle.contract,
            report=loader_bundle.report,
        )
        cfg = TrainConfig(**{**config.to_dict(), "device": device})  # type: ignore[arg-type]
        # Unwrap DDP for TrainResult.module on rank 0.
        result = train_supervised_module(module, local_bundle, config=cfg)
        if rank == 0:
            raw = module.module if hasattr(module, "module") else module
            result.module = raw.cpu()
            # Drop optimizer state tensors that may pin CUDA devices across the queue.
            if result.optimizer_state is not None:
                result.optimizer_state = None
            result_queue.put(
                {
                    "ok": True,
                    "train_result": result,
                    "device": device,
                }
            )
    except Exception as exc:  # noqa: BLE001
        if rank == 0:
            result_queue.put({"ok": False, "error": str(exc)})
        raise
    finally:
        dist.destroy_process_group()


def train_supervised_module_ddp(
    module_factory: ModuleFactory,
    loader_bundle: TorchLoaderBundle,
    *,
    config: TrainConfig | None = None,
    ddp_config: DDPConfig | None = None,
) -> DDPTrainResult:
    """Spawn single-node DDP training and return the rank-0 :class:`TrainResult`.

    Parameters
    ----------
    module_factory:
        Zero-arg callable that builds a **fresh** ``nn.Module`` in each process.
    loader_bundle:
        Shared in-memory loaders (datasets must be picklable for spawn).
    config:
        Train loop knobs. ``device`` is overridden per-rank.
    ddp_config:
        Backend / world-size / rendezvous knobs.

    Notes
    -----
    - Requires ``torch.cuda.device_count() >= 2`` for the CUDA/NCCL happy path.
    - CPU ``gloo`` multi-process is supported for smoke tests but is not a
      performance path.
    - Multi-node / Slurm / Kubernetes launchers are out of scope.
    """
    torch = require_torch(feature="DDP training")
    cfg = config or TrainConfig()
    dcfg = ddp_config or DDPConfig()
    n_cuda = ddp_cuda_device_count()
    use_cuda = n_cuda >= 2
    if not use_cuda and not dcfg.allow_cpu_ddp:
        raise ValidationError(
            "DDP requires torch.cuda.device_count() >= 2 for the NCCL path. "
            "Pass DDPConfig(allow_cpu_ddp=True) only for gloo multi-process smoke tests "
            "(not a performance path). Multi-node cluster launch is out of scope."
        )
    if dcfg.world_size is None:
        world_size = n_cuda if use_cuda else 2
    else:
        world_size = int(dcfg.world_size)
    if world_size < 2:
        raise ValidationError("DDP world_size must be >= 2")
    if use_cuda and world_size > n_cuda:
        raise ValidationError(
            f"Requested world_size={world_size} exceeds CUDA device count={n_cuda}"
        )
    backend = resolve_ddp_backend(dcfg.backend, use_cuda=use_cuda)
    warnings: list[str] = []
    if not use_cuda:
        warnings.append(
            "CUDA device count < 2; running CPU gloo multi-process smoke path "
            "(not a throughput optimization)."
        )

    ctx = torch.multiprocessing.get_context("spawn")
    queue: Any = ctx.Queue()
    torch.multiprocessing.spawn(
        _worker,
        args=(
            world_size,
            backend,
            dcfg.master_addr,
            dcfg.master_port,
            module_factory,
            loader_bundle,
            cfg,
            dcfg.find_unused_parameters,
            queue,
        ),
        nprocs=world_size,
        join=True,
    )
    if queue.empty():
        raise ValidationError("DDP training produced no rank-0 result")
    payload = queue.get()
    if not payload.get("ok"):
        raise ValidationError(f"DDP training failed: {payload.get('error')}")
    train_result: TrainResult = payload["train_result"]
    device_ids = tuple(range(world_size)) if use_cuda else ()
    return DDPTrainResult(
        train_result=train_result,
        world_size=world_size,
        backend=backend,
        device_ids=device_ids,
        disclosures=(
            f"Single-node DDP with world_size={world_size}, backend={backend}.",
            "Train sampler shards batches per rank; validation metrics come from rank 0.",
        ),
        limitations=(
            "Single-node only — no multi-node rendezvous, elastic launch, or cluster scheduler.",
            "Loader datasets must be picklable under the spawn start method.",
            "Alpha quality: prefer single-process fit_torch unless you need multi-GPU scale.",
        ),
        warnings=tuple(warnings),
        meta={"use_cuda": use_cuda, "n_cuda": n_cuda},
    )
