"""Session mixin: fairness disparity reporting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from buildml.session import fairness_ops
from buildml.session.mixins._shared import *  # noqa: F403


class FairnessSessionMixin:
    """Public Session methods for observational fairness reporting."""

    if TYPE_CHECKING:
        _fairness_report: Any

    @staticmethod
    def fairness_capability_matrix() -> dict[str, Any]:
        """Honest fairness metric / non-goal matrix."""
        return fairness_ops.fairness_capability_matrix()

    def evaluate_fairness(
        self,
        *,
        sensitive_column: str,
        partition: str = "test",
        positive_label: Any = 1,
    ) -> FairnessReport:
        """Report group disparity metrics on a holdout partition.

        Session facade over :func:`buildml.session.fairness_ops.evaluate_fairness_op`.

        Returns
        -------
        FairnessReport
            Selection rates, demographic parity, disparate impact, equalized odds.
        """
        return cast(
            "FairnessReport",
            fairness_ops.evaluate_fairness_op(
                self,
                sensitive_column=sensitive_column,
                partition=partition,
                positive_label=positive_label,
            ),
        )

    @property
    def last_fairness(self) -> FairnessReport | None:
        """Most recent :meth:`evaluate_fairness` report, if any."""
        return cast("FairnessReport | None", getattr(self, "_fairness_report", None))
