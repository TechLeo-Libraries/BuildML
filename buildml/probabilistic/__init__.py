"""Bayesian / probabilistic ML domain (sklearn uncertainty + split conformal).

Behavior and limitations:
  - **native** backend: sklearn ``BayesianRidge`` / ``GaussianProcessRegressor`` /
    ``GaussianProcessClassifier`` / ``GaussianNB`` with predictive std /
    proba and optional **in-tree split conformal** intervals/sets.
  - **mapie** backend (``buildml[probabilistic-industry]``): MAPIE conformal
    regression/classification: split, CV+, jackknife+ when installed.
  - **ngboost** backend (``buildml[probabilistic-industry]``): natural gradient
    boosting with predictive distributions (NLL / CRPS).
  - Conformal calibration is carved from the Session **train** partition only;
    validation/test are never used for fit or conformal calibration.
  - Classical ``Session.calibration()`` (reliability / ECE for classical
    ``fit(...)`` classifiers) is unchanged and complementary.
  - **Not** a PyMC / Stan / NumPyro MCMC platform and **not** Bayesian deep nets.

Dependency policy: core stays numpy/pandas/pyarrow/sklearn. In-tree split
conformal requires no optional extra. MAPIE / NGBoost are behind
``buildml[probabilistic-industry]``.

Lazy imports: core never grows heavy probabilistic-programming stacks.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "IntervalMethod",
    "ProbabilisticBackend",
    "ProbabilisticConfig",
    "ProbabilisticEstimator",
    "ProbabilisticEvalResult",
    "ProbabilisticFitResult",
    "ProbabilisticIntervalResult",
    "ProbabilisticPlan",
    "ProbabilisticPredictResult",
    "ProbabilisticTask",
    "evaluate_probabilistic",
    "fit_probabilistic",
    "list_probabilistic_estimators",
    "load_probabilistic_bundle",
    "predict_interval",
    "predict_probabilistic",
    "probabilistic_capability_matrix",
    "probabilistic_status",
    "probabilistic_status_for_session",
    "save_probabilistic_bundle",
]


def __getattr__(name: str) -> Any:
    if name in {
        "ProbabilisticBackend",
        "ProbabilisticEstimator",
        "ProbabilisticTask",
        "IntervalMethod",
        "ProbabilisticConfig",
    }:
        from buildml.probabilistic import types as types_mod

        return getattr(types_mod, name)
    if name in {
        "ProbabilisticPlan",
        "ProbabilisticFitResult",
        "ProbabilisticEvalResult",
        "ProbabilisticPredictResult",
        "ProbabilisticIntervalResult",
    }:
        from buildml.probabilistic import results as results_mod

        return getattr(results_mod, name)
    if name == "fit_probabilistic":
        from buildml.probabilistic.fit import fit_probabilistic

        return fit_probabilistic
    if name == "evaluate_probabilistic":
        from buildml.probabilistic.evaluate import evaluate_probabilistic

        return evaluate_probabilistic
    if name == "predict_probabilistic":
        from buildml.probabilistic.predict import predict_probabilistic

        return predict_probabilistic
    if name == "predict_interval":
        from buildml.probabilistic.predict import predict_interval

        return predict_interval
    if name in {
        "probabilistic_capability_matrix",
        "list_probabilistic_estimators",
    }:
        from buildml.probabilistic import catalog as catalog_mod

        return getattr(catalog_mod, name)
    if name in {
        "BUNDLE_FORMAT",
        "CHECKPOINT_BOUNDARY",
        "save_probabilistic_bundle",
        "load_probabilistic_bundle",
    }:
        from buildml.probabilistic import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    if name in {"probabilistic_status", "probabilistic_status_for_session"}:
        from buildml.probabilistic import explain_hooks as hooks

        return getattr(hooks, name)
    raise AttributeError(f"module 'buildml.probabilistic' has no attribute {name!r}")
