"""SDMetrics quality evaluation adapter."""

from __future__ import annotations

from typing import Any

import pandas as pd

from buildml.synthetic.extras import require_sdmetrics


def sdmetrics_quality_scores(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    metadata: Any | None = None,
) -> tuple[dict[str, float], list[str]]:
    """Run SDMetrics QualityReport; return scalar metrics + warnings.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
real:
    real (pd.DataFrame).
synthetic:
    synthetic (pd.DataFrame).
metadata:
    metadata (Any | None).

Returns
-------
tuple[dict[str, float], list[str]]
    Tuple of results (tuple[dict[str, float], list[str]]) for downstream Session steps.
    """
    require_sdmetrics()
    from sdmetrics.reports.single_table import QualityReport

    warnings: list[str] = []
    if metadata is None:
        from sdv.metadata import SingleTableMetadata

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(real)
        warnings.append(
            "SDMetrics metadata inferred from real partition (not the frozen plan metadata)."
        )

    report = QualityReport()
    report.generate(real.reset_index(drop=True), synthetic.reset_index(drop=True), metadata)

    metrics: dict[str, float] = {"sdmetrics_overall": float(report.get_score())}
    try:
        props = report.get_properties()
        if props is not None and not getattr(props, "empty", True):
            for _, row in props.iterrows():
                prop = str(row.get("Property") or row.get("property") or "")
                score = row.get("Score") if "Score" in row else row.get("score")
                if prop and score is not None:
                    key = f"sdmetrics_{prop.lower().replace(' ', '_')}"
                    metrics[key] = float(score)
    except Exception as exc:  # noqa: BLE001: SDMetrics API drift across versions
        warnings.append(f"SDMetrics property breakdown unavailable: {exc}")

    return metrics, warnings
