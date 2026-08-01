"""Model cards summarizing fitted pipeline artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from buildml._version import __version__
from buildml.core.errors import ValidationError


@dataclass(slots=True)
class ModelCard:
    """Structured card for a persisted preprocess+estimator bundle.

    Parameters
    ----------
    title:
        Short display name.
    task:
        Classification or regression.
    estimator_name:
        Fitted estimator class name (or pipeline summary).
    feature_columns / target_column:
        Feature contract.
    metrics:
        Optional evaluation metrics keyed by partition.
    schema:
        Column schema snapshot at save time.
    preprocess_summary:
        Which Session preprocess plans were included.
    history_summary:
        Compact operation-history digest (not full provenance).
    lineage:
        Explicit artifact relationships and caveats.
    created_at / buildml_version:
        Save metadata.
    notes:
        Free-form limitations or operator notes.
    """

    title: str
    task: str
    estimator_name: str
    feature_columns: tuple[str, ...]
    target_column: str
    n_train_rows: int
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    schema: dict[str, Any] = field(default_factory=dict)
    preprocess_summary: dict[str, Any] = field(default_factory=dict)
    history_summary: list[dict[str, Any]] = field(default_factory=list)
    lineage: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    buildml_version: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "task": self.task,
            "estimator_name": self.estimator_name,
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "n_train_rows": self.n_train_rows,
            "metrics": {k: dict(v) for k, v in self.metrics.items()},
            "schema": dict(self.schema),
            "preprocess_summary": dict(self.preprocess_summary),
            "history_summary": list(self.history_summary),
            "lineage": dict(self.lineage),
            "created_at": self.created_at,
            "buildml_version": self.buildml_version,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelCard:
        return cls(
            title=str(payload.get("title", "model-card")),
            task=str(payload["task"]),
            estimator_name=str(payload["estimator_name"]),
            feature_columns=tuple(str(c) for c in payload.get("feature_columns", [])),
            target_column=str(payload["target_column"]),
            n_train_rows=int(payload.get("n_train_rows", 0)),
            metrics={
                str(part): {str(k): float(v) for k, v in metrics.items()}
                for part, metrics in payload.get("metrics", {}).items()
            },
            schema=dict(payload.get("schema", {})),
            preprocess_summary=dict(payload.get("preprocess_summary", {})),
            history_summary=list(payload.get("history_summary", [])),
            lineage=dict(payload.get("lineage", {})),
            created_at=str(payload.get("created_at", "")),
            buildml_version=str(payload.get("buildml_version", "")),
            notes=list(payload.get("notes", [])),
        )

    def to_markdown(self) -> str:
        lines = [
            f"# {self.title}",
            "",
            f"- Task: {self.task}",
            f"- Estimator: {self.estimator_name}",
            f"- Target: `{self.target_column}`",
            f"- Train rows: {self.n_train_rows}",
            (
                f"- Features ({len(self.feature_columns)}): "
                + ", ".join(f"`{c}`" for c in self.feature_columns)
            ),
            f"- Created: {self.created_at or 'unknown'}",
            f"- BuildML: {self.buildml_version or 'unknown'}",
            "",
            "## Metrics",
        ]
        if not self.metrics:
            lines.append("No evaluation metrics were attached at save time.")
        else:
            for partition, values in self.metrics.items():
                lines.append(f"### Partition `{partition}`")
                for key, value in values.items():
                    lines.append(f"- {key}: {value:.6f}")
        lines.extend(["", "## Preprocess"])
        if not self.preprocess_summary:
            lines.append("No Session preprocess plans were included in this bundle.")
        else:
            present = [
                key for key, value in self.preprocess_summary.items() if value is not None
            ]
            absent = [
                key for key, value in self.preprocess_summary.items() if value is None
            ]
            if present:
                lines.append(f"- Plans present: {', '.join(present)}")
            if absent:
                lines.append(f"- Plans absent: {', '.join(absent)}")
            for key, value in self.preprocess_summary.items():
                if value is None:
                    continue
                label = _short_plan_label(value)
                lines.append(f"- {key}: {label}")
        lines.extend(["", "## History summary"])
        if not self.history_summary:
            lines.append("No operation history summary was attached.")
        else:
            for item in self.history_summary:
                op = item.get("operation_id", "operation")
                seq = item.get("sequence", "?")
                lines.append(f"- [{seq}] {op}")
        lines.extend(["", "## Lineage"])
        for key, value in self.lineage.items():
            lines.append(f"- {key}: {value}")
        if self.notes:
            lines.extend(["", "## Notes"])
            for note in self.notes:
                lines.append(f"- {note}")
        lines.append("")
        return "\n".join(lines)


def build_model_card(
    *,
    fit_result: Any,
    dataset_schema: dict[str, Any] | None = None,
    preprocess_summary: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
    metrics: dict[str, dict[str, float]] | None = None,
    title: str | None = None,
    notes: list[str] | None = None,
    lineage: dict[str, Any] | None = None,
) -> ModelCard:
    """Construct a model card from a fitted result and optional Session context."""
    estimator = fit_result.estimator
    estimator_name = type(estimator).__name__
    if hasattr(estimator, "named_steps"):
        estimator_name = "Pipeline(" + ", ".join(estimator.named_steps) + ")"

    history_summary = _summarize_history(history or [])
    return ModelCard(
        title=title or f"{estimator_name} model card",
        task=str(fit_result.task),
        estimator_name=estimator_name,
        feature_columns=tuple(fit_result.feature_columns),
        target_column=str(fit_result.target_column),
        n_train_rows=int(fit_result.n_train_rows),
        metrics=metrics or {},
        schema=dataset_schema or {},
        preprocess_summary=preprocess_summary or {},
        history_summary=history_summary,
        lineage=lineage
        or {
            "artifact": "pipeline_bundle",
            "contains_checkpoint": False,
            "contains_raw_dataset": False,
        },
        created_at=datetime.now(timezone.utc).isoformat(),
        buildml_version=__version__,
        notes=list(
            notes
            or [
                "A pipeline bundle stores fitted preprocess plans and the estimator; "
                "it is not a Session checkpoint.",
                "Reload requires a feature contract compatible with the saved schema.",
                "Resample plans are lineage metadata only; they are not reapplied at inference.",
            ]
        ),
    )


def save_model_card(path: str | Path, card: ModelCard) -> Path:
    """Write ``model_card.json`` and ``model_card.md`` under ``path``."""
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "model_card.json"
    md_path = root / "model_card.md"
    json_path.write_text(json.dumps(card.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(card.to_markdown(), encoding="utf-8")
    return root


def load_model_card(path: str | Path) -> ModelCard:
    """Load a model card from a bundle directory or JSON file."""
    root = Path(path)
    json_path = root if root.suffix.lower() == ".json" else root / "model_card.json"
    if not json_path.exists():
        raise ValidationError(f"Model card not found at '{json_path}'")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    return ModelCard.from_dict(payload)


def _summarize_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for record in history[-40:]:
        summary.append(
            {
                "sequence": record.get("sequence"),
                "operation_id": record.get("operation_id") or record.get("action"),
                "decision_origin": record.get("decision_origin"),
            }
        )
    return summary


def _short_plan_label(value: Any) -> str:
    if not isinstance(value, dict):
        return "present"
    for key in ("method", "strategy", "sampler", "action"):
        if key in value and value[key] is not None:
            return f"present ({key}={value[key]})"
    if "columns" in value:
        return f"present (columns={len(value.get('columns') or [])})"
    if "selected_features_" in value:
        return f"present (selected={len(value.get('selected_features_') or [])})"
    return "present"
