"""Session mixin: fairness disparity reporting."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from buildml.session import fairness_ops
from buildml.session.mixins._shared import *  # noqa: F403

if TYPE_CHECKING:
    from buildml.fairness.results import FairnessReport


class FairnessSessionMixin:
    """Public Session methods for observational fairness reporting.

    Preferred namespaced API: ``session.fairness.*`` (domain flat actions emit
    DeprecationWarning until BuildML 3.0).
    """

    if TYPE_CHECKING:
        _fairness_report: Any
        _fairness_mitigation_suggestion: Any

    @staticmethod
    def fairness_capability_matrix() -> dict[str, Any]:
        """Honest fairness metric / non-goal matrix.

        Prefer ``session.fairness.capability_matrix()``. Canonical docs live on
        :func:`buildml.session.fairness_ops.fairness_capability_matrix`.

        Returns
        -------
        dict[str, Any]
            Backends, metrics, disclosures, and explicit non-goals.
        """
        return fairness_ops.fairness_capability_matrix()

    def evaluate_fairness(
        self,
        *,
        sensitive_column: str | Sequence[str],
        partition: str = "test",
        positive_label: Any = 1,
        include_classical_metrics: bool = True,
        bootstrap_samples: int = 0,
        stability_method: str = "bootstrap",
        subsample_fraction: float = 0.8,
        confidence_level: float = 0.95,
        random_state: int | None = 0,
        include_scores: bool = True,
    ) -> FairnessReport:
        """Report group disparity metrics on a holdout partition.

        Prefer ``session.fairness.evaluate(...)``. Session facade over
        :func:`buildml.session.fairness_ops.evaluate_fairness_op`.

        ``sensitive_column`` may be a single column or a sequence of columns
        for intersectional group keys. Set ``bootstrap_samples > 1`` for
        disclosed stability bands.

        Returns
        -------
        FairnessReport
            Selection rates, demographic parity, disparate impact, equalized
            odds, optional classical per-group metrics and stability bands.
        """
        return cast(
            "FairnessReport",
            fairness_ops.evaluate_fairness_op(
                self,
                sensitive_column=sensitive_column,
                partition=partition,
                positive_label=positive_label,
                include_classical_metrics=include_classical_metrics,
                bootstrap_samples=bootstrap_samples,
                stability_method=stability_method,  # type: ignore[arg-type]
                subsample_fraction=subsample_fraction,
                confidence_level=confidence_level,
                random_state=random_state,
                include_scores=include_scores,
            ),
        )

    def attach_fairness_to_last_eval(
        self,
        *,
        sensitive_column: str | Sequence[str],
        positive_label: Any = 1,
        include_classical_metrics: bool = True,
        bootstrap_samples: int = 0,
        stability_method: str = "bootstrap",
        subsample_fraction: float = 0.8,
        confidence_level: float = 0.95,
        random_state: int | None = 0,
        include_scores: bool = True,
        partition: str | None = None,
    ) -> FairnessReport:
        """Attach a fairness report using the latest classical evaluate partition.

        Prefer ``session.fairness.attach_to_last_eval(...)``. Does not modify
        classical evaluate results; stores ``session.fairness.last_report``.
        """
        return cast(
            "FairnessReport",
            fairness_ops.attach_fairness_to_last_eval_op(
                self,
                sensitive_column=sensitive_column,
                positive_label=positive_label,
                include_classical_metrics=include_classical_metrics,
                bootstrap_samples=bootstrap_samples,
                stability_method=stability_method,  # type: ignore[arg-type]
                subsample_fraction=subsample_fraction,
                confidence_level=confidence_level,
                random_state=random_state,
                include_scores=include_scores,
                partition=partition,
            ),
        )

    def suggest_fairness_thresholds(
        self,
        *,
        sensitive_column: str | Sequence[str],
        partition: str = "validation",
        positive_label: Any = 1,
        target: str = "demographic_parity",
        grid_size: int = 101,
    ) -> Any:
        """Suggest per-group thresholds (opt-in; not auto-applied).

        Prefer ``session.fairness.suggest_thresholds(...)``.
        """
        return fairness_ops.suggest_fairness_thresholds_op(
            self,
            sensitive_column=sensitive_column,
            partition=partition,
            positive_label=positive_label,
            target=target,
            grid_size=grid_size,
        )

    def suggest_fairness_reweighing(
        self,
        *,
        sensitive_column: str | Sequence[str],
        partition: str = "train",
        positive_label: Any = 1,
    ) -> Any:
        """Suggest reweighing sample weights (opt-in; not auto-applied).

        Prefer ``session.fairness.suggest_reweighing(...)``.
        """
        return fairness_ops.suggest_fairness_reweighing_op(
            self,
            sensitive_column=sensitive_column,
            partition=partition,
            positive_label=positive_label,
        )

    @property
    def last_fairness(self) -> FairnessReport | None:
        """Most recent fairness report, or ``None`` before the first audit.

        Prefer ``session.fairness.last_report``. ``None`` until
        :meth:`evaluate_fairness` / ``session.fairness.evaluate`` populates it.
        """
        return cast("FairnessReport | None", getattr(self, "_fairness_report", None))
