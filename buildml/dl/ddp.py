"""DistributedDataParallel training utilities (single-node + multi-node alpha).

Modes
-----
* **Single-node** (default): spawn ``world_size`` local processes (NCCL when
  ``cuda.device_count() >= 2``; CPU ``gloo`` only with ``allow_cpu_ddp=True``).
* **Multi-node**: join an existing ``torchrun`` / ``torch.distributed``
  rendezvous using ``WORLD_SIZE``, ``RANK``, ``LOCAL_RANK``, ``MASTER_ADDR``,
  ``MASTER_PORT``. Launch with::

      torchrun --nnodes=2 --nproc_per_node=2 --rdzv_backend=c10d \\
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT your_train_script.py

Does **not** provide Kubernetes multi-cluster orchestration or elastic
auto-scaling. Clear misconfig errors when env is incomplete.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from buildml.core.errors import ValidationError
from buildml.dl.extras import require_torch
from buildml.dl.results import TorchLoaderBundle, TrainResult
from buildml.dl.types import TrainConfig

ModuleFactory = Callable[[], Any]


@dataclass(slots=True)
class DistributedEnv:
    """Parsed torchrun / torch.distributed environment variables."""

    world_size: int
    rank: int
    local_rank: int
    master_addr: str
    master_port: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_size": self.world_size,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "master_addr": self.master_addr,
            "master_port": self.master_port,
        }


@dataclass(slots=True)
class DDPConfig:
    """DDP launch knobs for single-node spawn or multi-node torchrun join."""

    backend: Literal["gloo", "nccl", "auto"] = "auto"
    world_size: int | None = None
    master_addr: str = "127.0.0.1"
    master_port: str = "29500"
    find_unused_parameters: bool = False
    allow_cpu_ddp: bool = False
    multi_node: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "world_size": self.world_size,
            "master_addr": self.master_addr,
            "master_port": self.master_port,
            "find_unused_parameters": self.find_unused_parameters,
            "allow_cpu_ddp": self.allow_cpu_ddp,
            "multi_node": self.multi_node,
        }


@dataclass(slots=True)
class DDPTrainResult:
    """Rank-0 outcome from a DDP training run."""

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


def parse_torchrun_env(environ: Mapping[str, str] | None = None) -> DistributedEnv:
    """Parse torchrun-compatible distributed environment variables.

    Required: ``WORLD_SIZE``, ``RANK``, ``MASTER_ADDR``, ``MASTER_PORT``.
    ``LOCAL_RANK`` defaults to ``RANK`` when unset (single-node multi-proc).
    """
    env = dict(os.environ if environ is None else environ)
    missing = [
        key
        for key in ("WORLD_SIZE", "RANK", "MASTER_ADDR", "MASTER_PORT")
        if not str(env.get(key, "")).strip()
    ]
    if missing:
        raise ValidationError(
            "Multi-node / torchrun DDP requires environment variables "
            f"{missing} (also recommended: LOCAL_RANK). "
            "Launch with torchrun --nnodes ... --nproc_per_node ... "
            "or export WORLD_SIZE/RANK/LOCAL_RANK/MASTER_ADDR/MASTER_PORT."
        )
    try:
        world_size = int(env["WORLD_SIZE"])
        rank = int(env["RANK"])
        local_rank = int(env["LOCAL_RANK"]) if str(env.get("LOCAL_RANK", "")).strip() else rank
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "WORLD_SIZE, RANK, and LOCAL_RANK must be integers for torchrun DDP"
        ) from exc
    if world_size < 1:
        raise ValidationError("WORLD_SIZE must be >= 1")
    if rank < 0 or rank >= world_size:
        raise ValidationError(f"RANK={rank} out of range for WORLD_SIZE={world_size}")
    if local_rank < 0:
        raise ValidationError("LOCAL_RANK must be >= 0")
    master_addr = str(env["MASTER_ADDR"]).strip()
    master_port = str(env["MASTER_PORT"]).strip()
    if not master_addr:
        raise ValidationError("MASTER_ADDR must be non-empty")
    if not master_port:
        raise ValidationError("MASTER_PORT must be non-empty")
    return DistributedEnv(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        master_addr=master_addr,
        master_port=master_port,
    )


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


def _run_rank_training(
    *,
    rank: int,
    local_rank: int,
    world_size: int,
    backend: str,
    module_factory: ModuleFactory,
    loader_bundle: TorchLoaderBundle,
    config: TrainConfig,
    find_unused_parameters: bool,
    use_cuda: bool,
) -> TrainResult | None:
    torch = require_torch(feature="DDP worker")
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    from buildml.dl.train import train_supervised_module

    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    module = module_factory().to(device)
    if use_cuda:
        module = DDP(
            module,
            device_ids=[local_rank],
            output_device=local_rank,
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
    result = train_supervised_module(module, local_bundle, config=cfg)
    if rank == 0:
        raw = module.module if hasattr(module, "module") else module
        result.module = raw.cpu()
        if result.optimizer_state is not None:
            result.optimizer_state = None
        return result
    return None


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

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    use_cuda = backend == "nccl" and torch.cuda.is_available()
    if use_cuda:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    else:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    try:
        train_result = _run_rank_training(
            rank=rank,
            local_rank=rank,
            world_size=world_size,
            backend=backend,
            module_factory=module_factory,
            loader_bundle=loader_bundle,
            config=config,
            find_unused_parameters=find_unused_parameters,
            use_cuda=use_cuda,
        )
        if rank == 0:
            result_queue.put(
                {
                    "ok": True,
                    "train_result": train_result,
                    "device": f"cuda:{rank}" if use_cuda else "cpu",
                }
            )
    except Exception as exc:  # noqa: BLE001
        if rank == 0:
            result_queue.put({"ok": False, "error": str(exc)})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _train_single_node(
    module_factory: ModuleFactory,
    loader_bundle: TorchLoaderBundle,
    *,
    config: TrainConfig,
    dcfg: DDPConfig,
) -> DDPTrainResult:
    torch = require_torch(feature="DDP training")
    n_cuda = ddp_cuda_device_count()
    use_cuda = n_cuda >= 2
    if not use_cuda and not dcfg.allow_cpu_ddp:
        raise ValidationError(
            "Single-node DDP requires torch.cuda.device_count() >= 2 for the NCCL path. "
            "Pass allow_cpu_ddp=True only for gloo multi-process smoke tests "
            "(not a performance path). For multi-node, call with multi_node=True "
            "under torchrun (WORLD_SIZE/RANK/MASTER_ADDR/MASTER_PORT)."
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
            config,
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
            "Single-node spawn mode — for multi-node use multi_node=True under torchrun.",
            "Loader datasets must be picklable under the spawn start method.",
            "Alpha quality: prefer single-process fit_torch unless you need multi-GPU scale.",
        ),
        warnings=tuple(warnings),
        meta={"use_cuda": use_cuda, "n_cuda": n_cuda, "mode": "single_node"},
    )


def _train_multi_node(
    module_factory: ModuleFactory,
    loader_bundle: TorchLoaderBundle,
    *,
    config: TrainConfig,
    dcfg: DDPConfig,
    environ: Mapping[str, str] | None = None,
) -> DDPTrainResult:
    torch = require_torch(feature="Multi-node DDP training")
    import torch.distributed as dist

    dist_env = parse_torchrun_env(environ)
    n_cuda = ddp_cuda_device_count()
    use_cuda = n_cuda >= 1 and torch.cuda.is_available()
    if not use_cuda and not dcfg.allow_cpu_ddp:
        raise ValidationError(
            "Multi-node DDP on CPU requires allow_cpu_ddp=True (gloo smoke / debug only; "
            "not a performance path). For GPU multi-node, ensure CUDA is visible on each "
            "node and launch with torchrun."
        )
    if use_cuda and dist_env.local_rank >= n_cuda:
        raise ValidationError(
            f"LOCAL_RANK={dist_env.local_rank} exceeds CUDA device count={n_cuda} "
            "on this node. Check --nproc_per_node vs visible GPUs."
        )
    backend = resolve_ddp_backend(dcfg.backend, use_cuda=use_cuda)
    warnings: list[str] = []
    if not use_cuda:
        warnings.append(
            "Multi-node CPU gloo path enabled via allow_cpu_ddp=True "
            "(debug/smoke only)."
        )

    # Honour env rendezvous; optionally overlay explicit master from DDPConfig
    # when callers set non-default values (env still wins if already set).
    os.environ.setdefault("MASTER_ADDR", dist_env.master_addr)
    os.environ.setdefault("MASTER_PORT", dist_env.master_port)

    if use_cuda:
        torch.cuda.set_device(dist_env.local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            rank=dist_env.rank,
            world_size=dist_env.world_size,
        )
    try:
        train_result = _run_rank_training(
            rank=dist_env.rank,
            local_rank=dist_env.local_rank,
            world_size=dist_env.world_size,
            backend=backend,
            module_factory=module_factory,
            loader_bundle=loader_bundle,
            config=config,
            find_unused_parameters=dcfg.find_unused_parameters,
            use_cuda=use_cuda,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

    device_ids = (dist_env.local_rank,) if use_cuda else ()
    return DDPTrainResult(
        train_result=train_result,
        world_size=dist_env.world_size,
        backend=backend,
        device_ids=device_ids,
        disclosures=(
            f"Multi-node / torchrun DDP with world_size={dist_env.world_size}, "
            f"rank={dist_env.rank}, local_rank={dist_env.local_rank}, backend={backend}.",
            "Train sampler shards batches per global rank; validation on rank 0.",
        ),
        limitations=(
            "Requires torchrun-compatible env (WORLD_SIZE/RANK/LOCAL_RANK/"
            "MASTER_ADDR/MASTER_PORT). Not a Kubernetes multi-cluster orchestrator.",
            "Each process must build/load the same SplitPlan loaders before join.",
            "Alpha quality: verify NCCL connectivity and firewall rules yourself.",
        ),
        warnings=tuple(warnings),
        meta={
            "use_cuda": use_cuda,
            "n_cuda": n_cuda,
            "mode": "multi_node",
            "distributed_env": dist_env.to_dict(),
        },
    )


def train_supervised_module_ddp(
    module_factory: ModuleFactory,
    loader_bundle: TorchLoaderBundle,
    *,
    config: TrainConfig | None = None,
    ddp_config: DDPConfig | None = None,
    environ: Mapping[str, str] | None = None,
) -> DDPTrainResult:
    """Run DDP training (single-node spawn or multi-node torchrun join).

    Parameters
    ----------
    module_factory:
        Zero-arg callable that builds a **fresh** ``nn.Module`` in each process.
    loader_bundle:
        Shared in-memory loaders (datasets must be picklable for single-node spawn).
    config:
        Train loop knobs. ``device`` is overridden per-rank.
    ddp_config:
        Backend / world-size / rendezvous / ``multi_node`` knobs.
    environ:
        Optional env mapping for multi-node parsing (defaults to ``os.environ``).
    """
    require_torch(feature="DDP training")
    cfg = config or TrainConfig()
    dcfg = ddp_config or DDPConfig()
    if dcfg.multi_node:
        return _train_multi_node(
            module_factory,
            loader_bundle,
            config=cfg,
            dcfg=dcfg,
            environ=environ,
        )
    return _train_single_node(
        module_factory,
        loader_bundle,
        config=cfg,
        dcfg=dcfg,
    )
