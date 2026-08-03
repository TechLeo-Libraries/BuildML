"""Train one model across several GPUs or machines.

DistributedDataParallel is the standard way to scale Torch training. Each
process holds a full copy of the model and a distinct slice of the data. After
every backward pass the processes average their gradients, which keeps the
copies identical while spreading the work — so N processes get through an epoch
in roughly a fraction of the time, at the cost of an effective batch size N
times larger.

Two ways in. **Single-node** spawns the processes for you, one per visible GPU,
and is the simpler path when everything fits on one machine. **Multi-node**
joins a rendezvous that ``torchrun`` set up, reading its placement from the
environment.

The scope is deliberately narrow. This joins or spawns a process group and
trains; it does not schedule Kubernetes pods, handle nodes joining and leaving
mid-run, or manage a cluster. When the environment is incomplete it says which
variable is missing rather than hanging at the rendezvous, which is the failure
mode most worth avoiding.

Treat the whole module as alpha. Single-process training handles most datasets,
and the operational surface here — NCCL connectivity, firewall rules, pickling
across the spawn boundary — is genuinely more than it appears.

See Also
--------
buildml.dl.train : Single-process training.
buildml.dl.k8s : Manifests for running this under Kubernetes.
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
    """Where this process sits in a distributed run.

    Attributes
    ----------
    world_size:
        Total processes across all nodes.
    rank:
        This process's global index, ``0`` to ``world_size - 1``. Rank 0 is
        conventionally the one that reports results and saves checkpoints.
    local_rank:
        This process's index **on this machine**, which is also its GPU index.
    master_addr, master_port:
        Where the processes find each other to coordinate.

    Notes
    -----
    **``rank`` and ``local_rank`` differ across nodes, and confusing them is the
    classic multi-node bug.** On the second node of a two-GPU-per-node run,
    ranks 2 and 3 have local ranks 0 and 1. Using the global rank as a CUDA
    device index there asks for ``cuda:2`` on a machine with two GPUs.

    See Also
    --------
    parse_torchrun_env : Builds this from the environment.
    """

    world_size: int
    rank: int
    local_rank: int
    master_addr: str
    master_port: str

    def to_dict(self) -> dict[str, Any]:
        """Return the distributed placement as JSON-safe values.

        Useful in logs when diagnosing a multi-node run, where knowing which
        process produced which line is most of the work.

        Returns
        -------
        dict
            World size, global rank, local rank, master address, master port.
        """
        return {
            "world_size": self.world_size,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "master_addr": self.master_addr,
            "master_port": self.master_port,
        }


@dataclass(slots=True)
class DDPConfig:
    """How to launch a distributed run.

    Attributes
    ----------
    backend:
        How processes communicate. ``'nccl'`` for NVIDIA GPUs, ``'gloo'`` for
        CPU, ``'auto'`` to pick by whether CUDA is in use. NCCL is dramatically
        faster on GPUs and does not work without them.
    world_size:
        How many processes to spawn. ``None`` uses the visible GPU count.
        Single-node only; multi-node reads this from the environment.
    master_addr, master_port:
        Rendezvous point for single-node spawn. Change the port if it is
        already taken.
    find_unused_parameters:
        Let DDP tolerate parameters that receive no gradient. Costs an extra
        pass over the graph each step, so leave off unless your model has
        conditional branches that skip layers — DDP will otherwise hang waiting
        for gradients that never arrive.
    allow_cpu_ddp:
        Permit the CPU gloo path. This is for testing distributed code without
        GPUs; it is slower than single-process training, not faster.
    multi_node:
        Join an existing torchrun rendezvous instead of spawning locally.

    See Also
    --------
    train_supervised_module_ddp : Consumes this.
    """

    backend: Literal["gloo", "nccl", "auto"] = "auto"
    world_size: int | None = None
    master_addr: str = "127.0.0.1"
    master_port: str = "29500"
    find_unused_parameters: bool = False
    allow_cpu_ddp: bool = False
    multi_node: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the launch settings as JSON-safe values.

        Records how a distributed run was configured, so a later run can be
        arranged the same way.

        Returns
        -------
        dict
            Backend, world size, master address and port, and the three
            boolean flags.
        """
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
    """What a distributed run produced, as seen from rank 0.

    Attributes
    ----------
    train_result:
        The trained model and its history, with the DDP wrapper removed and the
        module moved to CPU. ``None`` on non-zero ranks, which train but do not
        report.
    world_size:
        How many processes participated.
    backend:
        Which communication backend was used.
    device_ids:
        The CUDA devices involved. Empty on the CPU path.
    disclosures:
        How the run was arranged.
    limitations:
        What this path does not cover.
    warnings:
        Notably, whether the run fell back to the CPU gloo path.
    meta:
        Whether CUDA was used, how many devices were visible, the mode, and the
        parsed environment for multi-node runs.

    Notes
    -----
    **The returned module is unwrapped and on CPU.** DDP wraps your module in a
    layer that only makes sense inside the process group, so the wrapper is
    stripped before returning. Optimiser state is cleared for the same reason —
    it refers to distributed parameters and would not restore meaningfully.

    See Also
    --------
    train_supervised_module_ddp : Produces this.
    """

    train_result: TrainResult | None
    world_size: int
    backend: str
    device_ids: tuple[int, ...]
    disclosures: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the distributed run as JSON-safe values.

        The nested training result is summarised through its own ``to_dict``,
        so weights and optimiser state are described rather than embedded.

        Returns
        -------
        dict
            The nested training result (or ``None``), world size, backend,
            device ids, disclosures, limitations, warnings, and metadata.
        """
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
    """Count usable CUDA devices, returning 0 rather than failing.

    A probe for capability checks and launch decisions. Any failure — Torch
    absent, CUDA absent, a broken driver — reports zero devices, because from
    the caller's point of view all of those mean the same thing.

    Returns
    -------
    int
        Number of visible CUDA devices, or 0.

    Notes
    -----
    Respects ``CUDA_VISIBLE_DEVICES``, so this counts what the process can
    actually use rather than what is physically installed.
    """
    try:
        torch = require_torch(feature="DDP device probe")
    except Exception:
        return 0
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.device_count())


def resolve_ddp_backend(requested: str, *, use_cuda: bool) -> str:
    """Choose the communication backend, refusing impossible combinations.

    Processes in a distributed run exchange gradients every step, and the
    backend is how. NCCL uses direct GPU-to-GPU transfers and is the only
    sensible choice with CUDA; gloo goes through the CPU and works anywhere.

    Parameters
    ----------
    requested:
        ``'nccl'``, ``'gloo'``, or ``'auto'``.
    use_cuda:
        Whether the run will use GPUs.

    Returns
    -------
    str
        The resolved backend name.

    Raises
    ------
    ValidationError
        If the name is unrecognised, or if NCCL is requested without CUDA.

    Notes
    -----
    **NCCL without CUDA raises rather than falling back.** Someone asking for
    NCCL expects GPU throughput, and silently giving them a slower CPU path
    would turn a configuration error into a mysterious performance problem.
    """
    if requested == "auto":
        return "nccl" if use_cuda else "gloo"
    if requested not in {"gloo", "nccl"}:
        raise ValidationError("DDP backend must be gloo, nccl, or auto")
    if requested == "nccl" and not use_cuda:
        raise ValidationError("nccl backend requires CUDA")
    return requested


def parse_torchrun_env(
    environ: Mapping[str, str] | None = None,
    *,
    require_local_rank: bool = False,
) -> DistributedEnv:
    """Read this process's distributed placement from the environment.

    ``torchrun`` communicates placement through environment variables. This
    parses them, validates them against each other, and returns a typed record
    — so a misconfigured launch fails with a clear message rather than a
    confusing hang or a wrong-device error deep in training.

    Parameters
    ----------
    environ:
        The mapping to read. Defaults to ``os.environ``. Supplying one is
        useful for testing.
    require_local_rank:
        Insist that ``LOCAL_RANK`` is present. Multi-node paths must set this.

    Returns
    -------
    DistributedEnv
        The parsed placement.

    Raises
    ------
    ValidationError
        If ``WORLD_SIZE``, ``RANK``, ``MASTER_ADDR``, or ``MASTER_PORT`` is
        missing or empty; if ``LOCAL_RANK`` is required and absent; if any rank
        value is not an integer; or if the ranks are inconsistent with the world
        size.

    Notes
    -----
    **``LOCAL_RANK`` defaults to ``RANK``, which is correct on one node and
    wrong on several.** Single-node runs have identical global and local ranks,
    so the default is safe and convenient. On multiple nodes the two diverge,
    and using the global rank as a CUDA device index selects a device that does
    not exist — hence ``require_local_rank=True`` on the multi-node path, which
    turns that silent misplacement into an explicit error.

    Examples
    --------
    Read a torchrun launch::

        env = parse_torchrun_env(require_local_rank=True)
        env.rank, env.local_rank

    See Also
    --------
    DistributedEnv : The returned record.
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
            f"{missing} (also required for multi-node: LOCAL_RANK). "
            "Launch with torchrun --nnodes ... --nproc_per_node ... "
            "or export WORLD_SIZE/RANK/LOCAL_RANK/MASTER_ADDR/MASTER_PORT."
        )
    local_raw = str(env.get("LOCAL_RANK", "")).strip()
    if not local_raw and require_local_rank:
        raise ValidationError(
            "Multi-node DDP requires LOCAL_RANK (torchrun sets this per process). "
            "Omitting LOCAL_RANK is unsafe: global RANK must not be used as a "
            "local CUDA device index across nodes."
        )
    try:
        world_size = int(env["WORLD_SIZE"])
        rank = int(env["RANK"])
        local_rank = int(local_raw) if local_raw else rank
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
        text_vocab=getattr(loader_bundle, "text_vocab", None),
        text_contract=getattr(loader_bundle, "text_contract", None),
        multimodal_contract=getattr(loader_bundle, "multimodal_contract", None),
        speech_contract=getattr(loader_bundle, "speech_contract", None),
        modality=getattr(loader_bundle, "modality", None),
        input_layout=getattr(loader_bundle, "input_layout", None),
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

    dist_env = parse_torchrun_env(environ, require_local_rank=True)
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

    # Honour the parsed torchrun rendezvous (explicit environ wins over stale
    # process-level MASTER_* leftovers from prior launches).
    os.environ["MASTER_ADDR"] = dist_env.master_addr
    os.environ["MASTER_PORT"] = dist_env.master_port

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
    """Train one model across several processes, each holding a slice of data.

    Every process builds its own copy of the module and trains on a distinct
    shard of the training rows. After each backward pass the processes average
    their gradients, so all copies stay identical and the effective batch is
    the per-process batch times the world size.

    Two modes. Single-node spawns the processes for you. Multi-node joins a
    rendezvous that ``torchrun`` already established, which is how you span
    machines::

        torchrun --nnodes=2 --nproc_per_node=2 --rdzv_backend=c10d \\
          --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT your_train_script.py

    Parameters
    ----------
    module_factory:
        A zero-argument callable returning a **fresh** module. Called once per
        process. It must construct rather than return a shared object — under
        the spawn start method the factory is pickled and re-executed, and a
        captured module would not survive that meaningfully.
    loader_bundle:
        The loaders. The training loader is re-wrapped with a distributed
        sampler so each rank sees a different shard. Datasets must be picklable
        for single-node spawn.
    config:
        Training settings. ``device`` is overridden per rank.
    ddp_config:
        Backend, world size, rendezvous, and mode.
    environ:
        Environment mapping for multi-node parsing. Defaults to ``os.environ``.

    Returns
    -------
    DDPTrainResult
        Rank 0's outcome, with the module unwrapped and on CPU. Other ranks
        return a result whose ``train_result`` is ``None``.

    Raises
    ------
    MissingExtraError
        If PyTorch is not installed.
    ValidationError
        If single-node DDP is attempted with fewer than two GPUs and
        ``allow_cpu_ddp`` is not set; if the world size is below 2 or exceeds
        the GPU count; if the multi-node environment is incomplete; if
        ``LOCAL_RANK`` exceeds this node's GPU count; if there is no training
        loader; or if rank 0 produces no result.

    Notes
    -----
    **The effective batch size is multiplied by the world size**, and this
    changes training. Four processes at batch 32 apply gradients averaged over
    128 rows, which usually means fewer, smoother updates per epoch — often
    worth raising the learning rate to compensate.

    **Validation runs on rank 0 only.** Every rank evaluating the same
    validation set would produce identical numbers logged four times, so the
    other ranks skip it.

    **This path is alpha.** Single-process ``fit_torch`` is better tested and
    fast enough for most datasets; reach for DDP when a model genuinely does not
    fit or train in reasonable time on one GPU.

    **Kubernetes orchestration and elastic scaling are out of scope.** This
    joins a rendezvous you arranged; it does not schedule pods or handle nodes
    joining and leaving mid-run.

    Examples
    --------
    Single-node across the visible GPUs::

        result = train_supervised_module_ddp(
            lambda: build_tabular_mlp(12, n_classes=3),
            bundle,
            config=TrainConfig(epochs=20),
        )
        result.train_result.n_epochs_ran

    See Also
    --------
    buildml.dl.train.train_supervised_module : The single-process loop.
    parse_torchrun_env : How multi-node placement is read.
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
