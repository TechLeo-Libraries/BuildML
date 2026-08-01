"""Classical model fit / predict / evaluate."""

from buildml.model.compare import ModelComparison, compare_estimators
from buildml.model.diagnostics import DiagnosticReport
from buildml.model.plot_boards import PlotBoardReport, build_eval_plot_board
from buildml.model.selection import (
    CVScoreResult,
    NestedCVResult,
    OuterFoldResult,
    SearchResult,
    cv_score,
    grid_search,
    nested_cv_score,
    optuna_search,
    randomized_search,
)
from buildml.model.supervised import (
    EvaluateResult,
    FitResult,
    evaluate_estimator,
    fit_estimator,
    materialize_partition_design,
    predict_estimator,
)

__all__ = [
    "CVScoreResult",
    "DiagnosticReport",
    "EvaluateResult",
    "FitResult",
    "ModelComparison",
    "NestedCVResult",
    "OuterFoldResult",
    "PlotBoardReport",
    "SearchResult",
    "build_eval_plot_board",
    "compare_estimators",
    "cv_score",
    "evaluate_estimator",
    "fit_estimator",
    "grid_search",
    "materialize_partition_design",
    "nested_cv_score",
    "optuna_search",
    "predict_estimator",
    "randomized_search",
]
