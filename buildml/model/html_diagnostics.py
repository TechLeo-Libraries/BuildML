# ruff: noqa: E501
"""Render a diagnostic report as a single HTML file that works offline.

Model results usually have to leave the notebook: to a reviewer, a stakeholder,
an audit trail, a pull request. A screenshot loses the numbers and a JSON dump
loses the reading.

This renders the whole report as one self-contained file. Figures are embedded
as data URIs rather than referenced, so nothing breaks when the file is emailed
or committed, and it opens with no network access at all.

The output keeps the structure that makes a report reviewable: findings with
their evidence, recommendations with the findings they rest on, methods, and
limitations. A dashboard of numbers without its caveats is how a qualified
result becomes an unqualified claim on the way to someone else's desk.

See Also
--------
buildml.model.diagnostics.DiagnosticReport.export_html : The usual entry point.
buildml.reporting.html : The shared report shell.
"""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from buildml.reporting.html import (
    ReportSection,
    element_id,
    encode_asset,
    escape,
    render_badge,
    render_reading_frame,
    render_report,
    render_table,
    severity_tone,
)

MODEL_SECTION_IDS = (
    "summary",
    "metrics",
    "evidence",
    "interpretation",
    "visuals",
    "actions",
    "methods",
    "skipped",
)


def export_diagnostics_html(
    report_dict: dict[str, Any],
    path: str | Path,
    *,
    title: str = "BuildML Diagnostics Dashboard",
    figures: Mapping[str, Any] | None = None,
) -> Path:
    """Write the report to one HTML file with everything embedded.

    Builds the standard sections: summary, metrics, evidence, interpretation,
    visuals, actions, methods, and anything skipped: using the shared BuildML
    report shell, so diagnostics look the same wherever they come from.

    Figures are encoded into the document rather than linked, which is what
    makes the file portable: one attachment, no directory of images travelling
    beside it.

    Parameters
    ----------
    report_dict:
        The report as plain data, typically from
        :meth:`~buildml.model.diagnostics.DiagnosticReport.to_dict`.
    path:
        Where to write. Parent directories are created as needed.
    title:
        Page title and heading.
    figures:
        Matplotlib figures to embed, keyed by panel name. Figures already
        recorded in the report are picked up automatically.

    Returns
    -------
    pathlib.Path
        The file written.

    Notes
    -----
    **Embedding makes the file large.** Several figures can push a report into
    the megabytes, which is the price of it working anywhere.

    **Skipped panels are rendered, not omitted.** An analysis that could not run
    appears with its reason, so the reader can tell the difference between "this
    was fine" and "this was never measured".

    **Missing sections are dropped rather than shown empty**, so a report with
    no figures does not carry a blank visuals heading.

    See Also
    --------
    buildml.model.diagnostics.DiagnosticReport.export_html : The wrapper.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = report_dict
    payload = report.get("payload") or {}
    metrics = report.get("metrics") or _scalar_metrics(payload)
    findings = report.get("findings") or []
    recommendations = report.get("recommendation_details") or []
    skipped = report.get("skipped") or []
    task = (
        report.get("task")
        or payload.get("task")
        or (
            "classification"
            if report.get("kind") in {"calibration", "threshold_sweep"}
            else "model"
        )
    )
    sections = [
        _summary_section(report, findings, skipped),
        _metrics_section(metrics, payload),
        _evidence_section(findings),
        _interpretation_section(report),
        _assets_section(report, figures or {}),
        _actions_section(recommendations),
        _methods_section(report),
        _skipped_section(skipped),
    ]
    assert [section.key for section in sections] == list(MODEL_SECTION_IDS)
    document = render_report(
        title,
        sections,
        subtitle=(
            "Task-appropriate evidence, exact values, interpretation, "
            "limitations, and API-linked next actions."
        ),
        metadata={
            "Task": task,
            "Partition": report.get("partition") or payload.get("partition", "not available"),
            "Diagnostic": report.get("kind", "evaluation plot board"),
            "Generator": "BuildML model diagnostics",
        },
    )
    destination.write_text(document, encoding="utf-8")
    return destination


def _summary_section(
    report: Mapping[str, Any],
    findings: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
) -> ReportSection:
    body = render_reading_frame(
        examined="The fitted model's task-appropriate evaluation and diagnostic evidence.",
        observed=(
            f"{len(findings)} structured findings and {len(skipped)} skipped or "
            "degraded panels were retained."
        ),
        why="The summary identifies what deserves review; exact values and evidence remain in later sections.",
        limits="No diagnostic proves future performance, causality, fairness, or deployment suitability.",
        next_step="Open the evidence behind each finding before acting.",
    )
    rows = [
        {
            "finding": item.get("title"),
            "severity": item.get("severity"),
            "detail": item.get("detail"),
        }
        for item in findings
    ]
    body += render_table(rows, caption="Layered finding summary")
    return ReportSection("summary", "Summary and scope", body)


def _metrics_section(metrics: Mapping[str, Any], payload: Mapping[str, Any]) -> ReportSection:
    rows = [{"metric": key, "value": value} for key, value in metrics.items()]
    body = render_reading_frame(
        examined="Exact retained metric values and structured curve or importance rows.",
        observed=f"{len(rows)} headline metric values are available.",
        why="Exact values keep visual impressions auditable.",
        limits="Metric meaning depends on task, partition, prevalence, score definition, and sample support.",
        next_step="Compare values with the stated method and limitation before making a model choice.",
    )
    body += render_table(rows, caption="Exact metrics")
    for key in (
        "calibration_curve",
        "rows",
        "per_class_brier",
        "segments",
        "small_segments",
        "operating_points",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            body += render_table(value[:100], caption=f"{key.replace('_', ' ').title()}")
        elif isinstance(value, Mapping):
            if key == "operating_points":
                point_rows = [
                    {"name": name, **(dict(point) if isinstance(point, Mapping) else {"value": point})}
                    for name, point in value.items()
                ]
                body += render_table(point_rows, caption="Operating points")
            else:
                body += (
                    f"<details><summary>{escape(key.replace('_', ' ').title())}</summary>"
                    f"{_json(value)}</details>"
                )
    recommended = payload.get("recommended_threshold")
    if isinstance(recommended, Mapping):
        body += render_table([dict(recommended)], caption="Recommended threshold")
    if "train_sizes" in payload:
        curve_rows = [
            {
                "train_size": size,
                "train_mean": train,
                "validation_mean": valid,
                "train_std": train_std,
                "validation_std": valid_std,
            }
            for size, train, valid, train_std, valid_std in zip(
                payload.get("train_sizes", []),
                payload.get("train_scores_mean", []),
                payload.get("valid_scores_mean", []),
                payload.get("train_scores_std", []),
                payload.get("valid_scores_std", []),
                strict=False,
            )
        ]
        body += render_table(curve_rows, caption="Learning curve values")
    return ReportSection("metrics", "Exact metrics and data", body)


def _evidence_section(findings: list[dict[str, Any]]) -> ReportSection:
    cards: list[str] = []
    for finding in findings:
        key = str(finding.get("key", "finding"))
        severity = str(finding.get("severity", "info")).lower()
        evidence = finding.get("evidence") or []
        evidence_rows = [
            {
                "evidence": item.get("summary"),
                "value": item.get("value"),
                "source": item.get("source"),
                "limitations": item.get("limitations"),
            }
            for item in evidence
        ]
        cards.append(
            f'<article class="bml-finding severity-{escape(severity)}" '
            f'id="{escape(element_id(key, prefix="finding"))}">'
            f"<h3>{escape(finding.get('title'))} "
            f"{render_badge(finding.get('severity', 'info'), tone=severity_tone(severity))}</h3>"
            f"<p>{escape(finding.get('detail'))}</p>"
            f"{render_table(evidence_rows, caption='Supporting evidence')}</article>"
        )
    body = render_reading_frame(
        examined="Observed values supporting each interpretation.",
        observed=f"{len(findings)} evidence-linked findings are shown.",
        why="Evidence links separate measurement from editorial interpretation.",
        limits="Evidence is bounded by each row's source and limitations.",
        next_step="Trace any recommendation back to these finding IDs.",
    ) + "".join(cards)
    return ReportSection("evidence", "Top findings and supporting evidence", body)


def _interpretation_section(report: Mapping[str, Any]) -> ReportSection:
    interpretation = report.get("interpretation") or []
    limitations = report.get("limitations") or []
    body = render_reading_frame(
        examined="Interpretive statements retained by the diagnostic.",
        observed=(
            f"{len(interpretation)} interpretations and {len(limitations)} "
            "explicit limitations are available."
        ),
        why="Interpretation explains model behavior without hiding the measurement boundary.",
        limits="; ".join(str(item) for item in limitations)
        or "No additional limitation was recorded.",
        next_step="Use interpretation to form a focused validation question, not as a conclusion by itself.",
    )
    body += render_table(
        [{"type": "interpretation", "statement": item} for item in interpretation]
        + [{"type": "limitation", "statement": item} for item in limitations],
        caption="Interpretation and limitations",
    )
    return ReportSection("interpretation", "Interpretation and limitations", body)


def _assets_section(report: Mapping[str, Any], figures: Mapping[str, Any]) -> ReportSection:
    assets: list[tuple[str, str]] = []
    failures: list[dict[str, str]] = []
    for name, figure in figures.items():
        if isinstance(figure, Mapping):
            failures.append({"panel": str(name), "reason": str(figure.get("error", "not rendered"))})
            continue
        try:
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
            assets.append((str(name), encode_asset(buffer.getvalue(), media_type="image/png")))
        except Exception as exc:  # noqa: BLE001
            failures.append({"panel": str(name), "reason": str(exc)})
    for name, raw_path in (report.get("figure_paths") or {}).items():
        if any(existing == str(name) for existing, _ in assets):
            continue
        try:
            assets.append((str(name), encode_asset(Path(raw_path))))
        except OSError as exc:
            failures.append({"panel": str(name), "reason": str(exc)})
    gallery = "".join(
        '<figure class="bml-figure">'
        f'<button type="button" class="bml-figure__expand" aria-label="Expand {escape(name)}">'
        f'<img src="{uri}" alt="{escape(name)} diagnostic plot"></button>'
        f"<figcaption>{escape(name.replace('_', ' '))}</figcaption></figure>"
        for name, uri in assets
    )
    body = render_reading_frame(
        examined="Task-appropriate visual diagnostics.",
        observed=f"{len(assets)} assets were embedded directly in this HTML file.",
        why="Visuals complement, rather than replace, the exact metric tables.",
        limits="Plots can conceal support and uncertainty; use the evidence tables for exact values.",
        next_step="Expand a figure, then compare its pattern with the corresponding metric.",
    )
    body += f'<div class="bml-gallery">{gallery}</div>'
    body += render_table(failures, caption="Asset rendering failures")
    return ReportSection("visuals", "Visual evidence: Figure board", body)


def _actions_section(recommendations: list[dict[str, Any]]) -> ReportSection:
    items: list[str] = []
    for recommendation in recommendations:
        links = ", ".join(
            f'<a href="#{escape(element_id(key, prefix="finding"))}">{escape(key)}</a>'
            for key in recommendation.get("based_on") or []
        )
        action = recommendation.get("action") or {}
        items.append(
            "<li>"
            f"<strong>{escape(recommendation.get('title'))}</strong> "
            f"{render_badge(recommendation.get('priority'), tone='info')}"
            f"<p>{escape(recommendation.get('rationale'))}</p>"
            f"<p>Observed evidence: {links or 'unavailable'}</p>"
            f"<p>API action: <code>{escape(action.get('label'))}</code></p>"
            + (
                f"<p><small>Limits: {escape('; '.join(recommendation.get('caveats') or []))}</small></p>"
                if recommendation.get("caveats")
                else ""
            )
            + "</li>"
        )
    body = render_reading_frame(
        examined="Recommendations derived from observed findings.",
        observed=f"{len(items)} evidence-linked API actions are available.",
        why="Each action exposes both its evidence and the public API entry point.",
        limits="Actions remain subject to domain costs, validation design, and deployment constraints.",
        next_step="Run an action only after confirming its linked evidence applies.",
    ) + "<ol>" + "".join(items) + "</ol>"
    return ReportSection("actions", "Evidence-linked recommendations and next actions", body)


def _methods_section(report: Mapping[str, Any]) -> ReportSection:
    methods = report.get("methods") or []
    body = render_reading_frame(
        examined="Retained calculation and evaluation methods.",
        observed=f"{len(methods)} method notes were recorded.",
        why="Method details define what the reported values estimate.",
        limits="Library defaults and environment versions are not fully captured here.",
        next_step="Retain this report with the model, split, and environment metadata.",
    )
    body += render_table([{"method": item} for item in methods], caption="Methods")
    return ReportSection("methods", "Methods", body)


def _skipped_section(skipped: list[dict[str, Any]]) -> ReportSection:
    body = render_reading_frame(
        examined="Panels that were disabled, inapplicable, unsupported, or failed.",
        observed=f"{len(skipped)} skipped or degraded panels are listed.",
        why="An absent panel is not evidence that the corresponding risk is absent.",
        limits="Some task/estimator combinations cannot support every diagnostic.",
        next_step="Resolve the stated prerequisite only when that panel answers a real decision question.",
    )
    body += render_table(skipped, caption="Skipped and degraded panels")
    return ReportSection("skipped", "Skipped and degraded panels", body)


def _scalar_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in payload.items()
        if value is None or isinstance(value, (bool, int, float, str))
    }


def _json(value: Any) -> str:
    return f'<pre class="bml-json">{escape(json.dumps(value, indent=2, default=str))}</pre>'
