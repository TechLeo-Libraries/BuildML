"""Environment probes for optional extras and Torch health."""

from __future__ import annotations

from typing import Any


def probe_torch() -> dict[str, Any]:
    """Import Torch and run a tiny tensor op; report status for proof JSON."""
    status: dict[str, Any] = {
        "available": False,
        "version": None,
        "cuda_available": False,
        "smoke_ok": False,
        "error": None,
        "skip_torch_paths": True,
    }
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 — proof harness must never crash here
        status["error"] = f"{type(exc).__name__}: {exc}"
        return status

    status["available"] = True
    status["version"] = str(getattr(torch, "__version__", "unknown"))
    try:
        status["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001
        status["error"] = f"cuda_probe: {type(exc).__name__}: {exc}"

    try:
        x = torch.randn(2, 3)
        _ = float((x @ x.T).sum().detach())
        status["smoke_ok"] = True
        status["skip_torch_paths"] = False
    except Exception as exc:  # noqa: BLE001
        status["error"] = f"tensor_smoke: {type(exc).__name__}: {exc}"
        status["skip_torch_paths"] = True
    return status


TORCH_STATUS: dict[str, Any] = probe_torch()


def extra_available(module_name: str) -> bool:
    """Return True if ``import module_name`` succeeds."""
    try:
        __import__(module_name)
        return True
    except Exception:  # noqa: BLE001
        return False


def skip_reason(module_name: str, *, feature: str) -> str:
    """Human-readable skip message for missing optional deps."""
    return (
        f"Skipped {feature}: optional module '{module_name}' is not importable. "
        f"Install the matching buildml extra and re-run."
    )
