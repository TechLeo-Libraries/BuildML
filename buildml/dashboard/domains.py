"""Domain board registry for the EDA Teaching Studio app."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DomainSpec:
    """One board in the studio: what it shows, teaches, and exports.

    The dashboard is organised by question rather than by analyzer — "is my data
    clean", "what predicts the target", "will this generalise" — because that is
    how someone arrives at a report. Each spec maps one of those questions onto
    the report sections that answer it, the concepts a reader might need
    explained, and the tables they can export.

    Declaring this as data rather than building it in code is what keeps the
    studio, the offline export, and the CSV endpoints consistent. Adding a board
    is adding a spec; there is no second place to update.

    Attributes
    ----------
    key:
        Stable identifier, used in URLs and element ids. Not renamed once
        published, since links into a report are shared.
    title:
        The board's heading.
    short:
        A phrase saying what is on it, shown in navigation.
    icon:
        Icon name for the navigation entry.
    report_keys:
        Which report sections feed this board. A board whose sections are all
        empty renders as empty rather than being hidden — an absent board would
        look like an omission.
    concept_keys:
        Glossary entries offered alongside, from :mod:`buildml.explain`. This is
        the teaching half of the studio: the analysis and its explanation on the
        same screen.
    csv_sections:
        Which tables can be downloaded from this board.

    Notes
    -----
    **Frozen and slotted.** These are module-level constants shared across
    requests; immutability means a handler cannot corrupt the registry.

    See Also
    --------
    buildml.dashboard.teaching : Where ``concept_keys`` are resolved.
    buildml.dashboard.exports : Where ``csv_sections`` are resolved.
    """

    key: str
    title: str
    short: str
    icon: str
    report_keys: tuple[str, ...]
    concept_keys: tuple[str, ...]
    csv_sections: tuple[str, ...]


DOMAINS: tuple[DomainSpec, ...] = (
    DomainSpec(
        key="cockpit",
        title="Command cockpit",
        short="Readiness, severity map, next actions",
        icon="gauge",
        report_keys=("overview", "findings", "recommendation_details", "warnings"),
        concept_keys=("column-roles", "leakage-boundary", "data-splitting"),
        csv_sections=("findings", "recommendations", "roles"),
    ),
    DomainSpec(
        key="quality",
        title="Data quality",
        short="Completeness, constants, identifiers",
        icon="shield-check",
        report_keys=("quality",),
        concept_keys=("missing-data", "column-roles", "feature-schema"),
        csv_sections=("missing_rates", "quality_flags"),
    ),
    DomainSpec(
        key="features",
        title="Feature profiles",
        short="Univariate shape, entropy, normality",
        icon="bar-chart-3",
        report_keys=("univariate",),
        concept_keys=("normality-screens", "feature-scaling", "missing-data"),
        csv_sections=("univariate_numeric", "univariate_categorical"),
    ),
    DomainSpec(
        key="relationships",
        title="Relationships",
        short="Correlation and mutual information",
        icon="git-branch",
        report_keys=("bivariate",),
        concept_keys=("mutual-information", "feature-importance", "leakage-boundary"),
        csv_sections=("correlations", "spearman", "cramers_v", "mutual_information"),
    ),
    DomainSpec(
        key="multivariate",
        title="Multivariate structure",
        short="VIF, clusters, PCA summary",
        icon="layers",
        report_keys=("multivariate",),
        concept_keys=("variance-inflation", "principal-components", "feature-schema"),
        csv_sections=("vif", "pca"),
    ),
    DomainSpec(
        key="target",
        title="Target & validation",
        short="Target screens and train/test drift",
        icon="crosshair",
        report_keys=("target", "drift"),
        concept_keys=("class-imbalance", "dataset-drift", "evaluation-partitions"),
        csv_sections=("target_summary", "drift"),
    ),
    DomainSpec(
        key="outliers",
        title="Outliers",
        short="Univariate and multivariate screens",
        icon="scan-search",
        report_keys=("outliers",),
        concept_keys=("diagnostic-uncertainty", "missing-data"),
        csv_sections=("outliers",),
    ),
    DomainSpec(
        key="visuals",
        title="Visual evidence",
        short="Interactive Plotly boards",
        icon="image",
        report_keys=("adaptive_plan", "figures"),
        concept_keys=("diagnostic-uncertainty", "reproducibility"),
        csv_sections=("adaptive_plan",),
    ),
    DomainSpec(
        key="academy",
        title="Concept Academy",
        short="Searchable teaching library",
        icon="graduation-cap",
        report_keys=(),
        concept_keys=(),
        csv_sections=("concepts",),
    ),
)


DOMAIN_BY_KEY = {domain.key: domain for domain in DOMAINS}
