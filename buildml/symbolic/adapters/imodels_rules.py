"""imodels RuleFit / BoostedRules → BuildML RuleKnowledgeBase adapter."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.symbolic.extras import require_imodels
from buildml.symbolic.rules import Predicate, Rule, RuleKnowledgeBase


def induce_imodels_rules(
    frame: pd.DataFrame,
    columns: list[str],
    y: np.ndarray,
    *,
    task: str,
    method: str,
    max_rules: int = 32,
    random_state: int | None = 0,
    class_names: tuple[Any, ...] | None = None,
    max_depth: int = 3,
) -> tuple[RuleKnowledgeBase, Any]:
    """Fit imodels interpretable model and export rules."""
    imodels = require_imodels()
    x = frame[columns].to_numpy(dtype=float)
    y_arr = np.asarray(y)
    method_key = str(method).lower().replace("-", "_")

    if method_key == "rulefit":
        est = _build_rulefit(imodels, task=task, random_state=random_state, max_depth=max_depth)
    elif method_key == "boosted_rules":
        if task != "classification":
            raise ValidationError(
                "boosted_rules supports classification only; use rulefit for regression."
            )
        est = imodels.BoostedRulesClassifier(
            n_estimators=min(32, max_rules),
            max_depth=int(max_depth),
            random_state=random_state,
        )
    else:
        raise ValidationError(f"Unknown imodels method {method!r}.")

    est.fit(x, y_arr)
    rules_df = getattr(est, "rules_", None)
    if rules_df is None or len(rules_df) == 0:
        raise ValidationError(
            f"imodels {method_key} produced no exportable rules; "
            "try backend='sklearn' or relax max_depth."
        )

    rules_out: list[Rule] = []
    for rank, row in rules_df.head(max_rules).iterrows():
        rule_str = str(row.get("rule", row.iloc[0] if len(row) else ""))
        coef = float(row.get("coef", row.get("coefficient", 1.0)) or 1.0)
        antecedents = _parse_imodels_rule_string(rule_str, columns)
        if not antecedents:
            continue
        if task == "classification":
            if coef >= 0:
                consequent = _positive_class(class_names, y_arr)
            else:
                consequent = _negative_class(class_names, y_arr)
        else:
            consequent = coef
        rules_out.append(
            Rule(
                rule_id=f"imodels_{method_key}_{rank}",
                antecedents=antecedents,
                consequent=consequent,
                priority=int(1000 - int(rank) if isinstance(rank, (int, np.integer)) else rank),
                source=f"induced_{method_key}",
                confidence=abs(coef) if task == "classification" else None,
            )
        )

    if not rules_out:
        raise ValidationError("Could not parse imodels rules into predicates.")

    if task == "classification":
        default = _majority_class(y_arr, class_names)
    else:
        default = float(np.mean(y_arr))

    provenance = f"induced_{method_key}"
    kb = RuleKnowledgeBase(
        rules=tuple(rules_out),
        default_consequent=default,
        columns_used=tuple(columns),
        disclosures=(
            f"Rule provenance={provenance}: imodels {method_key} fitted on "
            "Session train only, exported as if-then rules.",
            f"imodels max_depth={max_depth}, max_rules={max_rules}.",
        ),
        provenance=provenance,
    )
    return kb, est


def _build_rulefit(imodels: Any, *, task: str, random_state: int | None, max_depth: int) -> Any:
    if task == "classification":
        return imodels.RuleFitClassifier(
            max_rules=32,
            tree_size=int(max_depth),
            random_state=random_state,
        )
    return imodels.RuleFitRegressor(
        max_rules=32,
        tree_size=int(max_depth),
        random_state=random_state,
    )


def _parse_imodels_rule_string(rule_str: str, columns: list[str]) -> tuple[Predicate, ...]:
    """Parse imodels rule strings (similar to skope format)."""
    from buildml.symbolic.adapters.skope_rules import _parse_skope_rule_string

    preds = _parse_skope_rule_string(rule_str, columns)
    if preds:
        return preds
    # Fallback: single feature threshold patterns like "feature_0 <= 0.5"
    parts = re.split(r"\s+and\s+", str(rule_str).strip(), flags=re.IGNORECASE)
    out: list[Predicate] = []
    pattern = re.compile(
        r"^(?P<col>[\w.]+)\s*(?P<op><=|>=|<|>|==|!=)\s*(?P<val>-?\d+(?:\.\d+)?)$"
    )
    for part in parts:
        m = pattern.match(part.strip())
        if not m:
            continue
        col_raw = m.group("col")
        col = columns[int(col_raw.split("_")[-1])] if col_raw.startswith("feature_") and col_raw.split("_")[-1].isdigit() else col_raw
        if col not in columns:
            idx_match = re.search(r"(\d+)$", col_raw)
            if idx_match:
                idx = int(idx_match.group(1))
                if 0 <= idx < len(columns):
                    col = columns[idx]
        out.append(
            Predicate(column=str(col), op=m.group("op"), value=float(m.group("val")))  # type: ignore[arg-type]
        )
    return tuple(out)


def _positive_class(class_names: tuple[Any, ...] | None, y: np.ndarray) -> Any:
    if class_names and len(class_names) >= 2:
        return class_names[1]
    vals = np.unique(y)
    return vals[-1] if len(vals) else 1


def _negative_class(class_names: tuple[Any, ...] | None, y: np.ndarray) -> Any:
    if class_names and len(class_names) >= 1:
        return class_names[0]
    vals = np.unique(y)
    return vals[0] if len(vals) else 0


def _majority_class(y: np.ndarray, class_names: tuple[Any, ...] | None) -> Any:
    vals, counts = np.unique(y, return_counts=True)
    winner = vals[int(np.argmax(counts))]
    if class_names is not None:
        try:
            idx = int(winner)
            if 0 <= idx < len(class_names):
                return class_names[idx]
        except (TypeError, ValueError):
            pass
    return winner
