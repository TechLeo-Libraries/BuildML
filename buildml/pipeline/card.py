"""Record what a model is, so the artifact can be understood without you.

A saved estimator is opaque. Six months later, nobody can tell from the file
what it was trained on, what it scored, which transforms it assumes, or whether
the number it returns is a probability or a count. The questions get asked
anyway, usually urgently, and the usual answer is to retrain from scratch.

A model card is written at save time and answers them: the task, the estimator,
the features and target, the metrics that were attached, a digest of the
operations that produced it, and explicit notes about what the artifact is not.

Cards are written twice, as JSON for tooling and Markdown for people. The
Markdown is not decoration — the audience for a model card is frequently someone
without a Python environment, and a card nobody can open explains nothing.

See Also
--------
buildml.pipeline.bundle : Where cards are written and read.
buildml.pipeline.contract : The machine-checkable half of the same idea.
"""

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
    """What a saved model is, what it scored, and what it does not cover.

    Descriptive, not enforcing — nothing here is checked at score time, which is
    the schema contract's job. The card is for the human trying to decide
    whether an artifact can be trusted for a purpose.

    The ``notes`` field is the part people skip and should not. It is where the
    limitations live: that a bundle is not a checkpoint, that resample plans are
    not replayed, that reload needs a compatible feature contract.

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

    Notes
    -----
    **An empty ``metrics`` is a real signal.** It means nothing was attached at
    save time, so the artifact carries no stated performance at all — worth
    treating as a reason to re-evaluate before deploying, not as an oversight to
    ignore.

    **``history_summary`` is a digest, not provenance.** It keeps the last forty
    operations with their identifiers, enough to see the shape of the workflow
    and not enough to reproduce it. The full history belongs in a checkpoint.

    See Also
    --------
    build_model_card : Producing one from a fit result.
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
        """Convert the card to JSON-safe plain data for ``model_card.json``.

        Every collection is copied rather than referenced, so the written record
        cannot change if the card is mutated afterwards.

        Returns
        -------
        dict
            All fields, with tuples as lists and nested mappings copied.

        See Also
        --------
        from_dict : The inverse.
        """
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
        """Rebuild a card from stored JSON, tolerating older or partial records.

        Deliberately forgiving about the descriptive fields — a card written by
        an earlier version may lack ``lineage`` or ``notes``, and refusing to
        read it would lose the information it does carry.

        Parameters
        ----------
        payload:
            The parsed contents of ``model_card.json``.

        Returns
        -------
        ModelCard
            The reconstructed card, with absent optional fields defaulted empty.

        Raises
        ------
        KeyError
            If ``task``, ``estimator_name``, or ``target_column`` is missing.
            These identify what the model does; a card without them describes
            nothing and is better rejected than half-read.
        ValueError
            If ``n_train_rows`` or a metric value cannot be converted to a
            number.

        See Also
        --------
        to_dict : The inverse.
        """
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
        """Render the card as Markdown, stating absences as plainly as contents.

        Where a section has nothing to show, it says so — "No evaluation metrics
        were attached at save time" rather than an empty heading. A blank
        section reads like a formatting glitch; a sentence reads like a fact,
        and in this file the absences are facts worth knowing.

        The preprocessing section lists which plans are present *and* which are
        absent, for the same reason: knowing a model was trained without scaling
        is as useful as knowing it was trained with it.

        Returns
        -------
        str
            The card as Markdown, suitable for ``model_card.md``, a pull request
            comment, or a wiki page.

        Notes
        -----
        **Metrics are formatted to six decimal places.** Enough to distinguish
        close models without implying more precision than a metric on a finite
        test set actually has.

        See Also
        --------
        save_model_card : Writing this alongside the JSON.
        """
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
    """Assemble a card from a fit result and whatever context is available.

    Called automatically by :func:`~buildml.pipeline.bundle.save_pipeline_bundle`
    when no card is supplied, so every bundle carries one even if nothing was
    passed. Call it directly to add notes or a lineage record.

    Two details are handled for you. A scikit-learn ``Pipeline`` is named by its
    steps rather than reported as the useless ``'Pipeline'``, so the card says
    what is actually inside it. And the default notes state the limitations that
    matter — a bundle is not a checkpoint, reload needs a compatible contract,
    resample plans are not replayed at inference.

    Parameters
    ----------
    fit_result:
        The fitted result, read for its estimator, task, feature columns,
        target, and training row count. Typed loosely to avoid a circular
        import.
    dataset_schema:
        Column types at training time.
    preprocess_summary:
        Which plans were included, as dictionaries keyed by step name. Both
        present and absent entries are rendered.
    history:
        The operation log. The last forty entries are digested.
    metrics:
        Scores by partition. Worth supplying — a card without metrics states no
        performance at all.
    title:
        A display name. Defaults to one derived from the estimator.
    notes:
        Replaces the default limitations entirely, so include them yourself if
        you override. Add deployment caveats here: the population the model was
        trained on, known blind spots, the date the data ends.
    lineage:
        Replaces the default artifact relationships.

    Returns
    -------
    ModelCard
        The card, stamped with the current UTC time and BuildML version.

    Notes
    -----
    **``notes`` and ``lineage`` replace rather than extend the defaults.** If
    you pass notes, the standard limitations disappear from the card unless you
    repeat them.

    **The timestamp is UTC and ISO 8601**, so cards from different machines sort
    and compare correctly.

    See Also
    --------
    ModelCard : What ends up on the card.
    """
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
    """Write the card twice, as JSON for tooling and Markdown for people.

    Both files come from the same object, so they cannot disagree. The JSON is
    sorted and indented, which keeps a card diffable — two versions of a model
    can be compared line by line in review.

    Parameters
    ----------
    path:
        The directory to write into, created if missing. Normally the bundle
        root, so the card sits beside the model.
    card:
        The card to write.

    Returns
    -------
    Path
        The directory written to.

    Raises
    ------
    OSError
        If the directory cannot be created or either file written.

    See Also
    --------
    load_model_card : Reading the JSON back.
    """
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "model_card.json"
    md_path = root / "model_card.md"
    json_path.write_text(json.dumps(card.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(card.to_markdown(), encoding="utf-8")
    return root


def load_model_card(path: str | Path) -> ModelCard:
    """Read a card, given either the bundle directory or the JSON file itself.

    Accepting both saves callers a join. A path ending in ``.json`` is read
    directly; anything else is treated as a directory containing
    ``model_card.json``.

    Parameters
    ----------
    path:
        The bundle directory, or the card file.

    Returns
    -------
    ModelCard
        The loaded card.

    Raises
    ------
    ValidationError
        If no card file exists at the resolved location. Named explicitly in the
        message, since the two accepted forms make it easy to point at the
        wrong one.
    json.JSONDecodeError
        If the file is not valid JSON.
    KeyError
        If the payload is missing a field the card cannot do without.

    Notes
    -----
    **Only the JSON is read.** The Markdown is a rendering, not a source, and is
    regenerated from the object on the next save.

    See Also
    --------
    save_model_card : Writing it.
    """
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
