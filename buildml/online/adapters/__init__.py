"""Online / continual backend adapters."""

from __future__ import annotations

from typing import Any

from buildml.online.adapters.river import build_river_estimator, resolve_river_task
from buildml.online.adapters.sklearn import build_sklearn_estimator, resolve_sklearn_task
from buildml.online.adapters.torch_continual import (
    build_torch_continual_estimator,
    resolve_torch_task,
)
from buildml.online.types import OnlineBackend, OnlineTask

__all__ = [
    "build_online_estimator",
    "resolve_online_task",
]


def resolve_online_task(
    backend: OnlineBackend,
    estimator: str,
    task: OnlineTask | None,
) -> OnlineTask:
    """Resolve the online task type for a backend/estimator pair.

    Delegates to the backend-specific resolver and validates classifier vs
    regressor compatibility.

    Parameters
    ----------
    backend:
        Online backend (``sklearn``, ``industry``, or ``torch``).
    estimator:
        Estimator key for the backend.
    task:
        Optional explicit task override.

    Returns
    -------
    OnlineTask
        Resolved ``classification`` or ``regression`` task.

    Raises
    ------
    ValidationError
        When the estimator/task pairing is invalid.
    """
    if backend == "sklearn":
        return resolve_sklearn_task(estimator, task)
    if backend == "industry":
        return resolve_river_task(estimator, task)
    return resolve_torch_task(estimator, task)


def build_online_estimator(
    backend: OnlineBackend,
    estimator: str,
    *,
    random_state: int | None = 0,
    drift_detector: str = "mean_shift",
    n_features: int = 0,
    buffer_size: int = 512,
    epochs_per_update: int = 5,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    ewc_lambda: float = 100.0,
    hidden_dim: int = 64,
    device: str = "cpu",
) -> Any:
    """Build a backend-specific online estimator with a partial_fit interface.

    Dispatches to sklearn, River, or torch continual adapters based on
    ``backend``.

    Parameters
    ----------
    backend:
        Online backend (``sklearn``, ``industry``, or ``torch``).
    estimator:
        Estimator key for the backend.
    random_state:
        Seed for stochastic estimators.
    drift_detector:
        Drift disclosure mode for industry backends.
    n_features:
        Feature dimensionality (required for River lazy init).
    buffer_size:
        Replay buffer size for torch continual backends.
    epochs_per_update:
        Training epochs per partial_fit for torch backends.
    batch_size:
        Minibatch size for torch backends.
    learning_rate:
        Optimizer learning rate for torch backends.
    ewc_lambda:
        EWC penalty weight for ``ewc_mlp``.
    hidden_dim:
        MLP hidden width for torch backends.
    device:
        Torch device string (``cpu`` or ``cuda``).

    Returns
    -------
    Any
        Backend-specific estimator object supporting ``partial_fit`` (or refit
        fallback when disclosed).
    """
    if backend == "sklearn":
        return build_sklearn_estimator(estimator, random_state)
    if backend == "industry":
        return build_river_estimator(
            estimator,
            random_state=random_state,
            drift_detector=drift_detector,
            n_features=n_features,
        )
    return build_torch_continual_estimator(
        estimator,  # type: ignore[arg-type]
        random_state=random_state,
        buffer_size=buffer_size,
        epochs_per_update=epochs_per_update,
        batch_size=batch_size,
        learning_rate=learning_rate,
        ewc_lambda=ewc_lambda,
        hidden_dim=hidden_dim,
        device=device,
    )
