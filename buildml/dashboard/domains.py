"""Domain board registry for the EDA Teaching Studio app."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DomainSpec:
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
