"""Rule / predicate knowledge base for symbolic tabular inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

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
        return {"column": self.column, "op": self.op, "value": self.value}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> Predicate:
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
      - ``hard`` — override / repair predictions when fired
      - ``soft`` — blend / prefer with ``strength`` in neuro-symbolic modes
    ``kind``:
      - ``classification`` / ``regression`` — prediction rules
      - ``constraint`` — constraint overlay / repair (neuro-symbolic)
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
    """Compile caller-declared rules into a :class:`RuleKnowledgeBase`."""
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
    """Vectorized predicate mask over a Series."""
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
    """Apply decision-list semantics: first matching rule (by priority order).

    Returns (predictions, traces, rule_fire_matrix[n_rows, n_rules]).
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
    """Binary features: one column per rule (1 if antecedents match)."""
    _, _, fire_matrix = fire_rules(frame, knowledge_base)
    names = [f"rule__{rule.rule_id}" for rule in knowledge_base.rules]
    return fire_matrix.astype(float), names


def validate_rule_columns(
    knowledge_base: RuleKnowledgeBase,
    available: Iterable[str],
) -> None:
    """Refuse rules that reference unknown columns."""
    available_set = {str(c) for c in available}
    missing = [
        col for col in knowledge_base.columns_used if col not in available_set
    ]
    if missing:
        raise ValidationError(
            "Symbolic rules reference columns not present in the frame: "
            f"{missing}."
        )
