"""Rule / predicate knowledge base for symbolic tabular inference."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.symbolic.types import PredicateOp, RuleSource


@dataclass(slots=True)
class Predicate:
    """Atomic condition over a column value."""

    column: str
    op: PredicateOp
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise the predicate for rule export and bundle persistence.

        Used when serialising rule antecedents inside knowledge bases and
        teaching overlays that display individual conditions.

        Returns
        -------
        dict[str, Any]
            Column name, operator, and comparison value.
        """
        return {"column": self.column, "op": self.op, "value": self.value}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> Predicate:
        """Build a :class:`Predicate` from a mapping with ``column`` and ``op`` keys.

        Parses declarative rule fragments from Session payloads and bundle
        files into typed predicate objects for compilation.

        Parameters
        ----------
        payload:
            Dict with ``column``, ``op``, and optional ``value``.

        Returns
        -------
        Predicate
            Parsed atomic condition.

        Raises
        ------
        ValidationError
            When required keys are missing or ``op`` is unsupported.
        """
        if "column" not in payload or "op" not in payload:
            raise ValidationError(
                "Predicate requires 'column' and 'op' keys."
            )
        op = str(payload["op"])
        allowed = {
            "<",
            "<=",
            ">",
            ">=",
            "==",
            "!=",
            "in",
            "not_in",
            "isna",
            "notna",
        }
        if op not in allowed:
            raise ValidationError(
                f"Unsupported predicate op {op!r}; expected one of {sorted(allowed)}."
            )
        return cls(
            column=str(payload["column"]),
            op=op,  # type: ignore[arg-type]
            value=payload.get("value"),
        )


@dataclass(slots=True)
class Rule:
    """If-then rule: AND(antecedents) → consequent.

    ``hardness``:
      - ``hard``: override / repair predictions when fired
      - ``soft``: blend / prefer with ``strength`` in neuro-symbolic modes
    ``kind``:
      - ``classification`` / ``regression``: prediction rules
      - ``constraint``: constraint overlay / repair (neuro-symbolic)
    """

    rule_id: str
    antecedents: tuple[Predicate, ...]
    consequent: Any
    priority: int = 0
    source: RuleSource = "declared"
    strength: float = 1.0
    hardness: str = "hard"
    kind: str = "classification"
    support: int | None = None
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise the rule for knowledge-base export and teaching overlays.

        Embeds antecedents, consequent, priority, and optional support stats
        for history logs and walkthrough rule panels.

        Returns
        -------
        dict[str, Any]
            Rule metadata, antecedents, consequent, and optional support stats.
        """
        return {
            "rule_id": self.rule_id,
            "antecedents": [p.to_dict() for p in self.antecedents],
            "consequent": self.consequent,
            "priority": self.priority,
            "source": self.source,
            "strength": self.strength,
            "hardness": self.hardness,
            "kind": self.kind,
            "support": self.support,
            "confidence": self.confidence,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, default_id: str) -> Rule:
        """Build a :class:`Rule` from a declarative mapping.

        Accepts ``if``/``antecedents`` and ``then``/``consequent`` aliases used
        in Session declared-rule payloads.

        Parameters
        ----------
        payload:
            Rule definition mapping.
        default_id:
            Fallback rule id when none is supplied.

        Returns
        -------
        Rule
            Parsed if-then rule with validated hardness and strength.

        Raises
        ------
        ValidationError
            When consequent or antecedents are malformed.
        """
        if "consequent" not in payload and "then" not in payload:
            raise ValidationError(
                "Rule requires 'consequent' (or 'then') key."
            )
        antecedents_raw = payload.get("antecedents") or payload.get("if") or []
        if not isinstance(antecedents_raw, Sequence) or isinstance(
            antecedents_raw, (str, bytes)
        ):
            raise ValidationError("Rule 'antecedents' / 'if' must be a sequence.")
        antecedents = tuple(
            Predicate.from_mapping(item) for item in antecedents_raw
        )
        hardness = str(payload.get("hardness", "hard")).lower()
        if hardness not in {"hard", "soft"}:
            raise ValidationError("Rule hardness must be 'hard' or 'soft'.")
        strength = float(payload.get("strength", 1.0))
        if not 0.0 <= strength <= 1.0:
            raise ValidationError("Rule strength must be in [0, 1].")
        return cls(
            rule_id=str(payload.get("rule_id") or payload.get("id") or default_id),
            antecedents=antecedents,
            consequent=payload.get("consequent", payload.get("then")),
            priority=int(payload.get("priority", 0)),
            source=str(payload.get("source", "declared")),  # type: ignore[arg-type]
            strength=strength,
            hardness=hardness,
            kind=str(payload.get("kind", "classification")),
            support=(
                None
                if payload.get("support") is None
                else int(payload["support"])
            ),
            confidence=(
                None
                if payload.get("confidence") is None
                else float(payload["confidence"])
            ),
        )


@dataclass(slots=True)
class RuleTrace:
    """Which rules fired for a single row (explanation trace)."""

    row_index: Any
    fired_rule_ids: tuple[str, ...]
    chosen_rule_id: str | None
    prediction: Any
    neural_prediction: Any | None = None
    repaired: bool = False
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialise a per-row explanation trace for predict/eval history.

        Records which rules fired, which rule won the decision list, and any
        repair notes for neuro-symbolic overlays.

        Returns
        -------
        dict[str, Any]
            Fired rules, chosen rule, predictions, and repair notes.
        """
        return {
            "row_index": self.row_index,
            "fired_rule_ids": list(self.fired_rule_ids),
            "chosen_rule_id": self.chosen_rule_id,
            "prediction": self.prediction,
            "neural_prediction": self.neural_prediction,
            "repaired": self.repaired,
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class RuleKnowledgeBase:
    """Ordered rule base with optional default consequent."""

    rules: tuple[Rule, ...]
    default_consequent: Any = None
    columns_used: tuple[str, ...] = ()
    disclosures: tuple[str, ...] = ()
    provenance: str = "declared"

    def to_dict(self) -> dict[str, Any]:
        """Serialise the knowledge base for bundles and walkthrough panels.

        Exports the full ordered rule set plus provenance disclosures so
        downstream evaluate and predict steps can replay the same logic.

        Returns
        -------
        dict[str, Any]
            Rules, default consequent, provenance, and column contract.
        """
        return {
            "rules": [r.to_dict() for r in self.rules],
            "default_consequent": self.default_consequent,
            "columns_used": list(self.columns_used),
            "disclosures": list(self.disclosures),
            "provenance": self.provenance,
            "n_rules": len(self.rules),
        }


def parse_declared_rules(
    rules: Sequence[Mapping[str, Any] | Rule],
    *,
    default_consequent: Any = None,
    provenance: str = "declared",
) -> RuleKnowledgeBase:
    """Compile caller-declared rules into a :class:`RuleKnowledgeBase`.

    Expert-supplied rules are sorted by descending priority and validated for
    column references before fit continues.

    Parameters
    ----------
    rules:
        Sequence of :class:`Rule` objects or declarative mappings.
    default_consequent:
        Prediction when no rule antecedents match a row.
    provenance:
        Provenance label recorded on the knowledge base.

    Returns
    -------
    RuleKnowledgeBase
        Ordered, validated rule base ready for predict and evaluate.

    Raises
    ------
    ValidationError
        When ``rules`` is empty or an entry is neither Rule nor mapping.
    """
    if not rules:
        raise ValidationError(
            "Declared symbolic rules require a non-empty rules sequence."
        )
    compiled: list[Rule] = []
    for i, item in enumerate(rules):
        if isinstance(item, Rule):
            compiled.append(item)
        elif isinstance(item, Mapping):
            compiled.append(Rule.from_mapping(item, default_id=f"declared_{i}"))
        else:
            raise ValidationError(
                "Each rule must be a mapping or Rule instance."
            )
    # Higher priority first; stable within priority.
    compiled.sort(key=lambda r: (-int(r.priority), r.rule_id))
    columns = tuple(
        sorted(
            {
                pred.column
                for rule in compiled
                for pred in rule.antecedents
            }
        )
    )
    disclosures = (
        "Rule provenance=declared (expert / caller-supplied). "
        "Not induced from Session train unless source marks induced_*.",
    )
    return RuleKnowledgeBase(
        rules=tuple(compiled),
        default_consequent=default_consequent,
        columns_used=columns,
        disclosures=disclosures,
        provenance=provenance,
    )


def evaluate_predicate(series: pd.Series, predicate: Predicate) -> np.ndarray:
    """Evaluate a single predicate as a boolean mask over a column.

    Supports numeric comparisons, equality, membership, and null checks.

    Parameters
    ----------
    series:
        Column values for one partition row set.
    predicate:
        Atomic condition to evaluate.

    Returns
    -------
    numpy.ndarray
        Boolean mask aligned with ``series``.

    Raises
    ------
    ValidationError
        When ``in``/``not_in`` lack sequence values or numeric ops get non-numeric rhs.
    """
    op = predicate.op
    if op == "isna":
        return series.isna().to_numpy()
    if op == "notna":
        return series.notna().to_numpy()
    values = series
    rhs = predicate.value
    if op == "in":
        if not isinstance(rhs, (list, tuple, set)):
            raise ValidationError("Predicate op 'in' requires a sequence value.")
        return values.isin(list(rhs)).to_numpy()
    if op == "not_in":
        if not isinstance(rhs, (list, tuple, set)):
            raise ValidationError(
                "Predicate op 'not_in' requires a sequence value."
            )
        return (~values.isin(list(rhs))).to_numpy()
    # Numeric / equality comparisons; coerce when both sides numeric-looking.
    if op in {"<", "<=", ">", ">="}:
        left = pd.to_numeric(values, errors="coerce")
        try:
            right = float(rhs)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"Predicate {op} on column {predicate.column!r} needs a numeric value."
            ) from exc
        if op == "<":
            return (left < right).fillna(False).to_numpy()
        if op == "<=":
            return (left <= right).fillna(False).to_numpy()
        if op == ">":
            return (left > right).fillna(False).to_numpy()
        return (left >= right).fillna(False).to_numpy()
    # == / !=
    if op == "==":
        return (values == rhs).fillna(False).to_numpy()
    return (values != rhs).fillna(False).to_numpy()


def fire_rules(
    frame: pd.DataFrame,
    knowledge_base: RuleKnowledgeBase,
    *,
    row_indices: Sequence[Any] | None = None,
) -> tuple[list[Any], list[RuleTrace], np.ndarray]:
    """Apply decision-list semantics and return predictions with traces.

    Rules are evaluated in priority order; the first matching rule sets the
    prediction unless only the default consequent applies.

    Parameters
    ----------
    frame:
        Partition frame containing columns referenced by rules.
    knowledge_base:
        Ordered rule base with optional default consequent.
    row_indices:
        Row index labels aligned with ``frame`` rows for trace metadata.

    Returns
    -------
    tuple[list, list[RuleTrace], numpy.ndarray]
        Predictions, per-row traces, and a boolean fire matrix shaped
        ``(n_rows, n_rules)``.

    Raises
    ------
    ValidationError
        When a rule references a missing column or ``row_indices`` length mismatches.
    """
    n = len(frame)
    rules = knowledge_base.rules
    fire_matrix = np.zeros((n, len(rules)), dtype=bool)
    for j, rule in enumerate(rules):
        if not rule.antecedents:
            mask = np.ones(n, dtype=bool)
        else:
            mask = np.ones(n, dtype=bool)
            for pred in rule.antecedents:
                if pred.column not in frame.columns:
                    raise ValidationError(
                        f"Rule {rule.rule_id!r} references missing column "
                        f"{pred.column!r}."
                    )
                mask &= evaluate_predicate(frame[pred.column], pred)
        fire_matrix[:, j] = mask

    indices = (
        list(row_indices)
        if row_indices is not None
        else list(frame.index)
    )
    if len(indices) != n:
        raise ValidationError("row_indices length must match frame length.")

    predictions: list[Any] = []
    traces: list[RuleTrace] = []
    for i in range(n):
        fired_ids: list[str] = []
        chosen: str | None = None
        pred = knowledge_base.default_consequent
        for j, rule in enumerate(rules):
            if fire_matrix[i, j]:
                fired_ids.append(rule.rule_id)
                if chosen is None:
                    chosen = rule.rule_id
                    pred = rule.consequent
        notes: list[str] = []
        if chosen is None:
            notes.append("No rule fired; used default_consequent.")
        predictions.append(pred)
        traces.append(
            RuleTrace(
                row_index=indices[i],
                fired_rule_ids=tuple(fired_ids),
                chosen_rule_id=chosen,
                prediction=pred,
                notes=tuple(notes),
            )
        )
    return predictions, traces, fire_matrix


def rule_feature_matrix(
    frame: pd.DataFrame,
    knowledge_base: RuleKnowledgeBase,
) -> tuple[np.ndarray, list[str]]:
    """Build binary rule-fire features for rules-as-features neuro-symbolic mode.

    Each column indicates whether a row satisfies a rule's antecedents.

    Parameters
    ----------
    frame:
        Partition frame.
    knowledge_base:
        Rule base whose antecedents define the features.

    Returns
    -------
    tuple[numpy.ndarray, list[str]]
        Float matrix shaped ``(n_rows, n_rules)`` and feature names
        ``rule__<rule_id>``.
    """
    _, _, fire_matrix = fire_rules(frame, knowledge_base)
    names = [f"rule__{rule.rule_id}" for rule in knowledge_base.rules]
    return fire_matrix.astype(float), names


def validate_rule_columns(
    knowledge_base: RuleKnowledgeBase,
    available: Iterable[str],
) -> None:
    """Refuse rules that reference columns absent from the training frame.

    Called during fit before rule induction or compilation so missing columns
    fail fast with a clear validation error.

    Parameters
    ----------
    knowledge_base:
        Compiled or declared rule base.
    available:
        Column names present on the train partition frame.

    Raises
    ------
    ValidationError
        When any referenced column is missing.
    """
    available_set = {str(c) for c in available}
    missing = [
        col for col in knowledge_base.columns_used if col not in available_set
    ]
    if missing:
        raise ValidationError(
            "Symbolic rules reference columns not present in the frame: "
            f"{missing}."
        )
