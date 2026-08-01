"""Structured EDA report object."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from buildml.explain import Finding, Recommendation


@dataclass(slots=True)
class EDAReport:
    """Structured exploratory analysis result."""

    overview: dict[str, Any]
    quality: dict[str, Any]
    univariate: dict[str, Any]
    bivariate: dict[str, Any]
    multivariate: dict[str, Any]
    target: dict[str, Any]
    outliers: dict[str, Any]
    drift: dict[str, Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    recommendation_details: list[Recommendation] = field(default_factory=list)
    narrative: list[str] = field(default_factory=list)
    adaptive_plan: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    figures: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    html_path: str | None = None
    figure_dir: str | None = None
    figure_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        figure_meta = {
            key: ("figure" if value is not None and not isinstance(value, dict) else value)
            for key, value in self.figures.items()
        }
        return {
            "overview": self.overview,
            "quality": self.quality,
            "univariate": self.univariate,
            "bivariate": self.bivariate,
            "multivariate": self.multivariate,
            "target": self.target,
            "outliers": self.outliers,
            "drift": self.drift,
            "findings": [finding.to_dict() for finding in self.findings],
            "recommendation_details": [
                recommendation.to_dict() for recommendation in self.recommendation_details
            ],
            "narrative": list(self.narrative),
            "adaptive_plan": list(self.adaptive_plan),
            "recommendations": list(self.recommendations),
            "figures": figure_meta,
            "warnings": list(self.warnings),
            "html_path": self.html_path,
            "figure_dir": self.figure_dir,
            "figure_paths": dict(self.figure_paths),
        }

    def show(self, *, max_items: int = 12) -> None:
        """Print a high-signal console digest."""
        o = self.overview
        print(
            f"BuildML EDA · {o.get('n_rows')}×{o.get('n_columns')} · "
            f"sample={o.get('analysis_rows')} · "
            f"completeness={self.quality.get('completeness_score')}"
        )
        print("--- Narrative ---")
        for line in self.narrative[:max_items]:
            print(f"* {line}")
        print("--- Recommendations ---")
        for item in self.recommendations[:max_items]:
            print(f"- {item}")
        if self.html_path:
            print(f"HTML report: {self.html_path}")
        if self.figure_dir:
            print(f"Figures: {self.figure_dir}")

    def save_html(
        self,
        path: str | Path,
        *,
        html_format: str = "studio",
    ) -> Path:
        """Export this report to HTML.

        Default ``html_format="studio"`` writes an offline Teaching Studio
        snapshot. Use ``html_format="research"`` for the layered research shell.
        """
        if html_format == "studio":
            try:
                from buildml.dashboard.offline import export_studio_html

                destination = export_studio_html(self.to_dict(), path)
                self.html_path = str(destination)
                return destination
            except Exception:
                pass
        from buildml.eda.html_report import export_eda_html

        destination = export_eda_html(self.to_dict(), path, figures=self.figures)
        self.html_path = str(destination)
        return destination
