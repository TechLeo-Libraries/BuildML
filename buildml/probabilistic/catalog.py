"""Probabilistic backend catalog and honest capability matrix."""

from __future__ import annotations

from typing import Any, Literal

from buildml.probabilistic.extras import (
    mapie_available,
    mapie_spec_present,
    ngboost_available,
    ngboost_spec_present,
    probabilistic_industry_available,
)

ProbabilisticBackendName = Literal["native", "mapie", "ngboost"]

_NATIVE_ESTIMATORS = (
    "bayesian_ridge",
    "gaussian_process_regressor",
    "gaussian_process_classifier",
    "gaussian_nb",
)
_MAPIE_METHODS = ("split", "cv_plus", "jackknife_plus")
_NGBOOST_ESTIMATORS = ("ngboost_regressor", "ngboost_classifier")


def probabilistic_capability_matrix() -> dict[str, Any]:
    """Build the honest capability matrix for probabilistic backends.

    Reports which native, MAPIE, and NGBoost paths are installed, supported
    uncertainty outputs, evaluation metrics, and explicit non-goals for teaching
    overlays and Session walkthrough panels.

    Returns
    -------
    dict[str, Any]
        Nested backend entries, interval methods, install hints, and boundaries.
    """
    mapie_methods = list(_MAPIE_METHODS)
    if not mapie_available():
        jackknife_note = "install buildml[probabilistic-industry] for MAPIE"
    else:
        jackknife_note = "jackknife_plus via MAPIE method='plus' with cv='prefit' carve"

    return {
        "backends": {
            "native": {
                "available": True,
                "extra": None,
                "estimators": list(_NATIVE_ESTIMATORS),
                "tasks": ["regression", "classification"],
                "uncertainty": [
                    "posterior_std (BayesianRidge/GPR)",
                    "predict_proba (GPC/GaussianNB)",
                    "split_conformal (in-tree absolute-residual / 1−p(y))",
                ],
                "conformal": "in-tree split conformal carved from Session train only",
            },
            "mapie": {
                "available": mapie_available(),
                "extra": "probabilistic-industry",
                "methods": mapie_methods,
                "tasks": ["regression", "classification"],
                "uncertainty": [
                    "split conformal (prefit + train carve)",
                    "cv_plus (cross-validation+ on train)",
                    "jackknife_plus (when MAPIE supports prefit jackknife+)",
                ],
                "notes": jackknife_note,
            },
            "ngboost": {
                "available": ngboost_available(),
                "extra": "probabilistic-industry",
                "estimators": list(_NGBOOST_ESTIMATORS),
                "tasks": ["regression", "classification"],
                "uncertainty": [
                    "natural-gradient boosting predictive distributions",
                    "NLL / CRPS from pred_dist",
                    "optional in-tree split conformal overlay",
                ],
            },
        },
        "evaluation_metrics": {
            "regression": [
                "mae",
                "rmse",
                "r2",
                "nll",
                "crps",
                "interval_coverage",
                "mean_interval_width",
                "interval_score",
            ],
            "classification": [
                "accuracy",
                "f1_macro",
                "f1_weighted",
                "nll",
                "brier",
                "ece",
                "set_coverage",
                "mean_set_size",
            ],
        },
        "interval_methods": [
            "posterior_std",
            "split_conformal",
            "both",
            "none",
            "mapie_cv_plus",
            "mapie_jackknife_plus",
        ],
        "default_backend_when_installed": _default_backend_when_installed(),
        "default_mapie_method": "split",
        "install_hints": {
            "probabilistic-industry": (
                "pip install 'buildml[probabilistic-industry]'  "
                "# MAPIE conformal (split/CV+/jackknife+) + NGBoost uncertainty"
            ),
        },
        "non_goals": [
            "PyMC / Stan / NumPyro MCMC platform",
            "Bayesian deep nets",
            "Full MAPIE algorithm zoo beyond split/CV+/jackknife+",
        ],
        "industry_extra_present": mapie_spec_present() or ngboost_spec_present(),
        "industry_runtime_present": probabilistic_industry_available(),
        "mapie_spec_present": mapie_spec_present(),
        "ngboost_spec_present": ngboost_spec_present(),
        "industry_import_honesty": (
            "mapie / ngboost backend 'available' and industry_runtime_present "
            "use subprocess import probes. industry_extra_present / "
            "*_spec_present are find_spec only — spec-present but broken "
            "wheels report available=False."
        ),
        "classical_calibration_unchanged": (
            "Session.calibration() remains for classical fit(...) classifiers; "
            "evaluate_probabilistic reports NLL/Brier/ECE for probabilistic plans."
        ),
    }


def _default_backend_when_installed() -> str:
    if mapie_available():
        return "mapie"
    if ngboost_available():
        return "ngboost"
    return "native"


def list_probabilistic_estimators(
    *,
    backend: ProbabilisticBackendName | None = None,
) -> list[str]:
    """List estimator or conformal method keys for a probabilistic backend.

    Reads :func:`probabilistic_capability_matrix` so callers only offer keys
    that exist for the requested backend.

    Parameters
    ----------
    backend:
        ``native``, ``mapie``, ``ngboost``, or ``None`` for the combined list.

    Returns
    -------
    list[str]
        Valid estimator or method names for the backend.
    """
    matrix = probabilistic_capability_matrix()
    if backend == "native":
        return list(matrix["backends"]["native"]["estimators"])
    if backend == "mapie":
        return list(matrix["backends"]["mapie"]["methods"])
    if backend == "ngboost":
        return list(matrix["backends"]["ngboost"]["estimators"])
    if backend is not None:
        return []
    out: list[str] = []
    for key in _NATIVE_ESTIMATORS + _NGBOOST_ESTIMATORS:
        out.append(key)
    for method in _MAPIE_METHODS:
        if method not in out:
            out.append(method)
    return out


def backend_available(name: ProbabilisticBackendName) -> bool:
    """Return whether a probabilistic backend is available on this machine.

    Checks the ``available`` flag in :func:`probabilistic_capability_matrix`
    for native, MAPIE, or NGBoost entries.

    Parameters
    ----------
    name:
        Backend key such as ``native``, ``mapie``, or ``ngboost``.

    Returns
    -------
    bool
        ``True`` when the backend can be used for fit without missing extras.
    """
    entry = probabilistic_capability_matrix()["backends"].get(name)
    if entry is None:
        return False
    return bool(entry.get("available"))


def resolve_backend_estimator(
    *,
    backend: ProbabilisticBackendName | None,
    estimator: str,
    task: str | None = None,
) -> tuple[ProbabilisticBackendName, str, str]:
    """Validate backend/estimator pairing and infer task when possible.

    Normalises MAPIE method aliases, checks install status, and resolves the
    regression vs classification task before fit proceeds.

    Parameters
    ----------
    backend:
        Explicit backend or ``None`` to infer from ``estimator``.
    estimator:
        Estimator or conformal method key from the catalog.
    task:
        Optional task override; inferred when omitted for most backends.

    Returns
    -------
    tuple[str, str, str]
        Resolved ``(backend, estimator_key, task)`` triple.

    Raises
    ------
    ValidationError
        When the estimator does not belong to the backend or task conflicts.
    MissingExtraError
        When the resolved backend requires an industry extra that is missing.
    """
    from buildml.core.errors import MissingExtraError, ValidationError

    est_key = str(estimator).lower().replace("-", "_")

    resolved_backend: ProbabilisticBackendName
    if backend is None:
        if est_key in _NATIVE_ESTIMATORS:
            resolved_backend = "native"
        elif est_key in _MAPIE_METHODS or est_key.startswith("mapie_"):
            resolved_backend = "mapie"
        elif est_key in _NGBOOST_ESTIMATORS:
            resolved_backend = "ngboost"
        else:
            resolved_backend = _default_backend_when_installed()  # type: ignore[assignment]
    else:
        resolved_backend = backend

    # Normalize mapie-prefixed estimator names to method keys.
    estimator_key = est_key
    if resolved_backend == "mapie":
        if est_key.startswith("mapie_"):
            estimator_key = est_key.removeprefix("mapie_")
        if estimator_key not in _MAPIE_METHODS:
            if backend is not None or est_key not in _NATIVE_ESTIMATORS:
                raise ValidationError(
                    f"method='{estimator_key}' is not valid for backend='mapie'. "
                    f"Choose from {list(_MAPIE_METHODS)}."
                )

    if resolved_backend == "native":
        allowed = list(_NATIVE_ESTIMATORS)
        if estimator_key not in allowed:
            raise ValidationError(
                f"estimator='{estimator_key}' is not valid for backend='native'. "
                f"Choose from {allowed}."
            )
    elif resolved_backend == "mapie":
        if estimator_key not in _MAPIE_METHODS:
            raise ValidationError(
                f"method='{estimator_key}' is not valid for backend='mapie'. "
                f"Choose from {list(_MAPIE_METHODS)}."
            )
    elif resolved_backend == "ngboost":
        allowed = list(_NGBOOST_ESTIMATORS)
        if estimator_key not in allowed:
            raise ValidationError(
                f"estimator='{estimator_key}' is not valid for backend='ngboost'. "
                f"Choose from {allowed}."
            )

    if not backend_available(resolved_backend):
        extra = probabilistic_capability_matrix()["backends"][resolved_backend].get("extra")
        raise MissingExtraError(
            str(extra or "probabilistic-industry"),
            f"backend='{resolved_backend}' estimator='{estimator_key}'",
        )

    resolved_task = _infer_task(resolved_backend, estimator_key, task)
    return resolved_backend, estimator_key, resolved_task


def _infer_task(
    backend: ProbabilisticBackendName,
    estimator: str,
    task: str | None,
) -> str:
    from buildml.core.errors import ValidationError

    if backend == "native":
        if estimator in {"gaussian_process_classifier", "gaussian_nb"}:
            if task == "regression":
                raise ValidationError(
                    f"Estimator {estimator!r} is a classifier; task cannot be 'regression'."
                )
            return "classification"
        if estimator in {"bayesian_ridge", "gaussian_process_regressor"}:
            if task == "classification":
                raise ValidationError(
                    f"Estimator {estimator!r} is a regressor; task cannot be 'classification'."
                )
            return "regression"
    if backend == "ngboost":
        if estimator == "ngboost_classifier":
            if task == "regression":
                raise ValidationError("ngboost_classifier requires task='classification'.")
            return "classification"
        if estimator == "ngboost_regressor":
            if task == "classification":
                raise ValidationError("ngboost_regressor requires task='regression'.")
            return "regression"
    if backend == "mapie":
        if task is None:
            return "regression"
        if task not in {"regression", "classification"}:
            raise ValidationError("task must be 'regression' or 'classification'.")
        return task
    raise ValidationError(f"Unknown backend={backend!r}.")
