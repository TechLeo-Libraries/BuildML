"""Federated backend catalog and honest capability matrix."""

from __future__ import annotations

from typing import Any, Literal

from buildml.federated.extras import federated_industry_available, flwr_available

FederatedBackendName = Literal["native", "flower"]
FederatedMethodName = Literal["fedavg", "fedprox"]

_NATIVE_METHODS = ("fedavg", "fedprox")
_FLOWER_METHODS = ("fedavg", "fedprox")


def federated_capability_matrix() -> dict[str, Any]:
    """Honest capability matrix for federated backends and optional extras."""
    return {
        "backends": {
            "native": {
                "available": True,
                "extra": None,
                "methods": list(_NATIVE_METHODS),
                "estimators": [
                    "sgd_classifier",
                    "sgd_regressor",
                    "logistic_regression",
                    "ridge",
                    "linear_regression",
                ],
                "aggregation": "in-process weighted coef_/intercept_ averaging",
                "client_weighting": "sample_size (n_k / Σ n_k)",
                "secure_aggregation": False,
                "network_runtime": False,
                "notes": (
                    "Default core path: local FedAvg/FedProx simulation on Session "
                    "train rows partitioned by a client/group column. No optional "
                    "extra required."
                ),
            },
            "flower": {
                "available": flwr_available(),
                "extra": "federated-industry",
                "methods": list(_FLOWER_METHODS),
                "estimators": [
                    "sgd_classifier",
                    "sgd_regressor",
                    "logistic_regression",
                    "ridge",
                    "linear_regression",
                ],
                "aggregation": (
                    "Flower NumPyClient local fit + flwr weighted ndarray "
                    "aggregation (still in-process simulation)"
                ),
                "client_weighting": "sample_size via flwr.server.strategy.aggregate",
                "secure_aggregation": False,
                "network_runtime": False,
                "notes": (
                    "Uses Flower (flwr) NumPyClient wrappers over Session client "
                    "partitions and Flower's weighted aggregation helpers. Still "
                    "runs locally unless you deploy a real Flower ServerApp/ClientApp "
                    "yourself — not a turnkey production FL network stack."
                ),
            },
        },
        "default_backend_when_installed": _default_backend_when_installed(),
        "install_hints": {
            "federated-industry": (
                "pip install 'buildml[federated-industry]'  "
                "# Flower (flwr) NumPyClient + aggregation adapter"
            ),
        },
        "non_goals": [
            "Cryptographic secure aggregation (not implemented on any backend)",
            "Differential privacy guarantees from simulation alone",
            "Ray / gRPC production Flower deployment from Session.fit_federated",
            "Non-linear tree/neural FedAvg zoo without coef_/intercept_ path",
            "OpenFL / TensorFlow Federated replacement claims",
        ],
        "industry_extra_present": federated_industry_available(),
        "honesty": (
            "Both backends are honest local simulations on Session data. "
            "backend='flower' uses Flower libraries for client/aggregation wiring "
            "but does not start a networked FL deployment unless you operate one "
            "separately."
        ),
    }


def _default_backend_when_installed() -> str:
    if flwr_available():
        return "flower"
    return "native"


def list_federated_methods(*, backend: FederatedBackendName | None = None) -> list[str]:
    """List federation methods for a backend (or all when backend is None)."""
    matrix = federated_capability_matrix()["backends"]
    if backend is not None:
        entry = matrix.get(backend)
        if entry is None:
            return []
        return list(entry.get("methods") or [])
    out: list[str] = []
    for entry in matrix.values():
        for method in entry.get("methods") or []:
            if method not in out:
                out.append(method)
    return out


def backend_available(name: FederatedBackendName) -> bool:
    matrix = federated_capability_matrix()["backends"]
    entry = matrix.get(name)
    if entry is None:
        return False
    return bool(entry.get("available"))


def resolve_backend(
    backend: FederatedBackendName | None,
    *,
    method: str | None = None,
) -> FederatedBackendName:
    """Validate backend availability and apply honest defaults."""
    from buildml.core.errors import MissingExtraError, ValidationError

    method_key = None if method is None else str(method).lower().replace("-", "_")
    if method_key is not None and method_key not in set(_NATIVE_METHODS):
        raise ValidationError(
            f"Unknown federated method={method!r}. Supported: {sorted(_NATIVE_METHODS)}."
        )

    resolved: FederatedBackendName
    if backend is None:
        resolved = _default_backend_when_installed()  # type: ignore[assignment]
    else:
        resolved = str(backend).lower().replace("-", "_")  # type: ignore[assignment]
        if resolved not in {"native", "flower"}:
            raise ValidationError(
                f"Unknown federated backend={backend!r}. Supported: 'native', 'flower'."
            )

    if not backend_available(resolved):
        extra = federated_capability_matrix()["backends"][resolved].get("extra")
        raise MissingExtraError(
            str(extra or "federated-industry"),
            f"backend='{resolved}'",
        )
    return resolved
