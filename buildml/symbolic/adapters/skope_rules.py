"""SkopeRules (skope-rules) → BuildML RuleKnowledgeBase adapter."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.symbolic.extras import require_skope_rules
from buildml.symbolic.rules import Predicate, Rule, RuleKnowledgeBase


def induce_skope_rules(
    frame: pd.DataFrame,
    columns: list[str],
    y: np.ndarray,
    *,
    task: str,
    max_rules: int = 32,
    random_state: int | None = 0,
    class_names: tuple[Any, ...] | None = None,
    precision_min: float = 0.3,
    recall_min: float = 0.05,
    n_estimators: int = 30,
) -> tuple[RuleKnowledgeBase, Any]:
    """Fit SkopeRules on train and export rules into BuildML rule objects.

    Runs one-vs-rest SkopeRules mining per class on Session train, parses rule
    strings into predicates, and returns a :class:`RuleKnowledgeBase` with
    ``induced_skope`` provenance.

    Parameters
    ----------
    frame:
        Train partition frame.
    columns:
        Numeric feature columns for SkopeRules.
    y:
        Encoded train targets.
    task:
        Must be ``classification`` (SkopeRules is binary per class).
    max_rules:
        Cap on exported rules across all classes.
    random_state:
        Seed for SkopeRules.
    class_names:
        Class labels for consequents.
    precision_min, recall_min:
        SkopeRules quality thresholds.
    n_estimators:
        Number of SkopeRules base estimators.

    Returns
    -------
    tuple[RuleKnowledgeBase, SkopeRules estimator]
        Induced rules and the last fitted SkopeRules instance.

    Raises
    ------
    ValidationError
        When task is not classification, fewer than two classes exist, or no
        rules pass the precision/recall thresholds.
    """
    if task != "classification":
        raise ValidationError(
            "SkopeRules backend supports classification only; "
            "use rulefit for regression or backend='sklearn'."
        )
    SkopeRules = require_skope_rules()
    x = frame[columns].to_numpy(dtype=float)
    y_arr = np.asarray(y)
    unique = np.unique(y_arr)
    if len(unique) < 2:
        raise ValidationError("SkopeRules requires at least 2 classes.")

    rules_out: list[Rule] = []
    est = None
    # Multi-class: one-vs-rest rule mining per class (honest disclosure).
    for cls_idx, cls_val in enumerate(unique):
        binary_y = (y_arr == cls_val).astype(int)
        est = SkopeRules(
            feature_names=columns,
            precision_min=float(precision_min),
            recall_min=float(recall_min),
            n_estimators=int(n_estimators),
            random_state=random_state,
        )
        est.fit(x, binary_y)
        consequent = _class_label(cls_val, class_names)
        for rank, entry in enumerate(getattr(est, "rules_", [])[: max_rules // len(unique) + 1]):
            rule_str = entry[0] if isinstance(entry, (list, tuple)) else str(entry)
            precision = float(entry[1]) if isinstance(entry, (list, tuple)) and len(entry) > 1 else None
            recall = float(entry[2]) if isinstance(entry, (list, tuple)) and len(entry) > 2 else None
            antecedents = _parse_skope_rule_string(rule_str, columns)
            if not antecedents:
                continue
            rules_out.append(
                Rule(
                    rule_id=f"skope_{cls_idx}_{rank}",
                    antecedents=antecedents,
                    consequent=consequent,
                    priority=int(1000 - rank - cls_idx * 100),
                    source="induced_skope",
                    confidence=precision,
                    support=None,
                )
            )
            if len(rules_out) >= max_rules:
                break
        if len(rules_out) >= max_rules:
            break

    if not rules_out:
        raise ValidationError(
            "SkopeRules produced no rules at current precision_min/recall_min; "
            "relax thresholds or use backend='sklearn'."
        )

    default = _majority_class(y_arr, class_names)
    kb = RuleKnowledgeBase(
        rules=tuple(rules_out[:max_rules]),
        default_consequent=default,
        columns_used=tuple(columns),
        disclosures=(
            "Rule provenance=induced_skope: skope-rules SkopeRules fitted on "
            "Session train only, exported as if-then rules. Not expert-declared.",
            f"SkopeRules precision_min={precision_min}, recall_min={recall_min}, "
            f"n_estimators={n_estimators}.",
            "Multi-class uses one-vs-rest rule mining per class.",
        ),
        provenance="induced_skope",
    )
    return kb, est


def _parse_skope_rule_string(rule_str: str, columns: list[str]) -> tuple[Predicate, ...]:
    """Parse skope-rules string like 'a <= 1.0 and b > 0.5' into predicates."""
    parts = re.split(r"\s+and\s+", str(rule_str).strip(), flags=re.IGNORECASE)
    preds: list[Predicate] = []
    col_map = {f"c{i}": col for i, col in enumerate(columns)}
    col_map.update({col: col for col in columns})
    pattern = re.compile(
        r"^(?P<col>[a-zA-Z_][\w]*)\s*(?P<op><=|>=|<|>|==|!=)\s*(?P<val>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$"
    )
    for part in parts:
        m = pattern.match(part.strip())
        if not m:
            continue
        raw_col = m.group("col")
        col = col_map.get(raw_col, raw_col)
        op = m.group("op")
        val: Any = float(m.group("val"))
        preds.append(Predicate(column=str(col), op=op, value=val))  # type: ignore[arg-type]
    return tuple(preds)


def _class_label(val: Any, class_names: tuple[Any, ...] | None) -> Any:
    if class_names is None:
        return val
    try:
        idx = int(val)
        if 0 <= idx < len(class_names):
            return class_names[idx]
    except (TypeError, ValueError):
        pass
    return val


def _majority_class(y: np.ndarray, class_names: tuple[Any, ...] | None) -> Any:
    vals, counts = np.unique(y, return_counts=True)
    winner = vals[int(np.argmax(counts))]
    return _class_label(winner, class_names)
