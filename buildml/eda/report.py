"""Everything an EDA pass found, in one object you can read or export.

A container rather than an analysis. It holds each analyzer's section, the
findings and recommendations derived from them, the plot plan, any rendered
figures, and the warnings about what was sampled or capped.

Keeping the sections separate rather than merging them into one flat dict is
deliberate. A caller who wants the missing-value counts should not have to know
which analyzer produced them, and a section that was skipped stays visibly
empty rather than silently absent.

See Also
--------
buildml.eda.profile.explore_dataset : What produces this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from buildml.explain import Finding, Recommendation


@dataclass(slots=True)
class EDAReport:
    """The complete result of exploring a dataset.

    Sections in the order they were produced, then the interpretation layer,
    then the artifacts. Every field has a default, so a partial pass — one
    without a split, or without plots — yields a report with empty sections
    rather than missing attributes.

    Attributes
    ----------
    overview:
        Shape, dtypes, roles, and the feature-eligibility bookkeeping — which
        columns were excluded and why, separated into explicit role exclusions
        and heuristic ones. Also records how many rows and columns the detailed
        analyzers actually saw.
    quality:
        Completeness, duplicates, constants, identifier-like columns, mixed
        types, string patterns. Computed on the **full** frame.
    univariate:
        Per-column distributions.
    bivariate:
        Correlations, categorical associations, and mutual information against
        the target.
    multivariate:
        Correlation clusters, VIF, and PCA.
    target:
        Target profile and feature associations. Empty when no target role.
    outliers:
        Per-column and multivariate outlier screens.
    drift:
        Train/test distribution comparison. ``{'available': False, ...}`` when
        no split was supplied.
    findings:
        Structured claims with severity and evidence. **The place to start.**
    recommendation_details:
        Structured next steps with priorities and caveats.
    narrative:
        The findings as plain sentences, for older callers.
    adaptive_plan:
        The plot specifications that were chosen.
    recommendations:
        The recommendations as plain strings.
    figures:
        Rendered figures, when ``include_plots`` was set.
    warnings:
        What was sampled, what was capped. **Read these before the numbers.**
    html_path:
        Where an HTML report was written, if one was.
    figure_dir:
        Where figures were written, if they were.
    figure_paths:
        Individual figure files by key.

    Notes
    -----
    **``findings`` and ``narrative`` are the same content twice**, structured
    and flat. Same for ``recommendation_details`` and ``recommendations``. The
    structured forms carry severity, evidence, and caveats; prefer them.

    **A section can be empty for two different reasons** — nothing was found, or
    nothing was run. The warnings and the ``available`` flags disambiguate.

    See Also
    --------
    buildml.eda.profile.explore_dataset : What produces this.
    """

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
        """Flatten to a JSON-safe dict, replacing figures with placeholders.

        A Matplotlib figure cannot be serialised, so each becomes the string
        ``'figure'`` — enough to know one was rendered under that key, and small
        enough that the dict stays writable. Dict-valued figure entries, which
        are already data, pass through unchanged. The figures themselves stay on
        the report object.

        Findings and recommendations are converted through their own
        ``to_dict``, so the evidence and caveats survive.

        Returns
        -------
        dict
            The report as plain data, ready for JSON, an HTML template, or a
            comparison against a previous run.

        Notes
        -----
        **Figures are lost here.** Use ``figure_paths`` if you need the images,
        or render from ``adaptive_plan``.

        See Also
        --------
        save_html : Producing a readable document instead.
        """
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
        """Print the headline numbers, the narrative, and the recommendations.

        For the notebook, right after the pass. One line of shape and
        completeness, then the findings as sentences, then the next steps, then
        the paths to anything that was exported.

        Parameters
        ----------
        max_items:
            Cap on narrative lines and recommendations shown. Twelve of each is
            about what fits on a screen; the report holds the rest.

        Returns
        -------
        None
            Prints to stdout.

        Notes
        -----
        **``warnings`` is not printed.** Check it separately — a report built
        from a 1% sample looks exactly like one built from everything.

        See Also
        --------
        save_html : The full picture, including the plots.
        """
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
        """Write this report as a self-contained HTML file.

        Two shells for two audiences. ``'studio'`` produces the offline Teaching
        Studio snapshot — guided, explanatory, the same surface as
        ``session.eda_app()`` — and suits someone who needs the analysis
        explained as well as shown. ``'research'`` produces the layered shell
        with embedded figures, denser and aimed at a reader who already knows
        what they are looking at.

        Either way the output is one file with everything inlined, so it can be
        emailed or archived and will still render years later.

        Parameters
        ----------
        path:
            Where to write. Parent directories are created.
        html_format:
            ``'studio'`` or ``'research'``.

        Returns
        -------
        Path
            The file written. Also recorded on ``html_path``.

        Raises
        ------
        MissingExtraError
            If the research shell is used without the reporting extra.
        OSError
            If the file cannot be written.

        Notes
        -----
        **The studio shell falls back to the research shell** if it cannot be
        built — a missing optional dependency, say. You get a report either way,
        which is the right behaviour, but it does mean the format you asked for
        is not guaranteed. Check the output if it matters.

        **Figures appear only if they were rendered.** Pass ``include_plots``
        to :func:`~buildml.eda.profile.explore_dataset`, or the report is
        numbers and prose.

        See Also
        --------
        buildml.eda.html_report.export_eda_html : The research shell directly.
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
