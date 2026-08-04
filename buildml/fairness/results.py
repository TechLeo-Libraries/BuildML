"""Fairness report result types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from buildml.fairness.stability import FairnessStability


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
    sensitive_columns: tuple[str, ...] = ()
    intersectional: bool = False
    classical_metrics_by_group: dict[str, dict[str, float | None]] = field(
        default_factory=dict
    )
    stability: FairnessStability | None = None
    scope: dict[str, Any] = field(default_factory=dict)
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe serialization including stability and classical bridge."""
        payload = asdict(self)
        if self.stability is not None:
            payload["stability"] = self.stability.to_dict()
        return payload

    def to_markdown(self) -> str:
        """Readable Markdown digest with scope, gaps, groups, and warnings."""
        lines: list[str] = [
            "# Fairness report (observational)",
            "",
            f"- **Partition:** `{self.partition}`",
            f"- **Sensitive column(s):** `{self.sensitive_column}`"
            + (" (intersectional)" if self.intersectional else ""),
            f"- **Positive label:** `{self.positive_label!r}`",
            f"- **Rows:** {self.n_rows}",
            f"- **Groups:** {len(self.groups)}",
            "",
            "## Gap metrics",
            "",
            f"- Demographic parity difference: "
            f"**{_fmt(self.demographic_parity_difference)}**",
            f"- Disparate impact ratio: **{_fmt(self.disparate_impact_ratio)}**",
            f"- Equalized odds ΔTPR: "
            f"**{_fmt(self.equalized_odds_tpr_difference)}**",
            f"- Equalized odds ΔFPR: "
            f"**{_fmt(self.equalized_odds_fpr_difference)}**",
            "",
            "## Per-group selection / rates",
            "",
            "| Group | Support | Selection rate | TPR | FPR |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for g in self.groups:
            lines.append(
                f"| `{g}` | {self.support_by_group.get(g, 0)} | "
                f"{_fmt(self.selection_rate_by_group.get(g))} | "
                f"{_fmt(self.tpr_by_group.get(g))} | "
                f"{_fmt(self.fpr_by_group.get(g))} |"
            )

        if self.classical_metrics_by_group:
            lines.extend(
                [
                    "",
                    "## Per-group classical metrics",
                    "",
                    "| Group | Acc | Precision | Recall | F1 | ROC-AUC |",
                    "| --- | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for g in self.groups:
                m = self.classical_metrics_by_group.get(g, {})
                lines.append(
                    f"| `{g}` | {_fmt(m.get('accuracy'))} | "
                    f"{_fmt(m.get('precision'))} | {_fmt(m.get('recall'))} | "
                    f"{_fmt(m.get('f1'))} | {_fmt(m.get('roc_auc'))} |"
                )

        if self.stability is not None:
            lines.extend(
                [
                    "",
                    "## Stability bands",
                    "",
                    f"- Method: `{self.stability.method}` "
                    f"(n={self.stability.n_resamples}, "
                    f"CI={self.stability.confidence_level:.0%})",
                    "",
                    "| Metric | Point | CI low | CI high | Std |",
                    "| --- | ---: | ---: | ---: | ---: |",
                ]
            )
            for name, band in self.stability.metrics.items():
                lines.append(
                    f"| `{name}` | {_fmt(band.get('point'))} | "
                    f"{_fmt(band.get('ci_low'))} | {_fmt(band.get('ci_high'))} | "
                    f"{_fmt(band.get('std'))} |"
                )

        if self.scope:
            lines.extend(["", "## Scope", ""])
            for key, value in self.scope.items():
                lines.append(f"- **{key}:** {value}")

        if self.warnings:
            lines.extend(["", "## Warnings", ""])
            for w in self.warnings:
                lines.append(f"- {w}")

        if self.disclosures:
            lines.extend(["", "## Disclosures", ""])
            for d in self.disclosures:
                lines.append(f"- {d}")

        lines.append("")
        return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    try:
        if value != value:  # NaN
            return "—"
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)
