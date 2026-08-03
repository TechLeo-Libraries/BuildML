"""Fairness report result types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class FairnessReport:
    """Observational group disparity report for binary classification."""

    partition: str
    sensitive_column: str
    positive_label: Any
    n_rows: int
    groups: tuple[str, ...]
    selection_rate_by_group: dict[str, float]
    demographic_parity_difference: float
    disparate_impact_ratio: float | None
    equalized_odds_tpr_difference: float | None
    equalized_odds_fpr_difference: float | None
    tpr_by_group: dict[str, float] = field(default_factory=dict)
    fpr_by_group: dict[str, float] = field(default_factory=dict)
    support_by_group: dict[str, int] = field(default_factory=dict)
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe serialization."""
        return asdict(self)
