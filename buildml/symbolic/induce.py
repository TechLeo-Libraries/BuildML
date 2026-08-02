"""Train-only rule induction: decision-tree export + sequential covering list."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from buildml.core.errors import ValidationError
from buildml.symbolic.rules import Predicate, Rule, RuleKnowledgeBase


def induce_decision_tree_rules(
    frame: pd.DataFrame,
    columns: list[str],
    y: np.ndarray,
    *,
    task: str,
    max_depth: int = 4,
    min_samples_leaf: int = 5,
    max_rules: int = 32,
    random_state: int | None = 0,
    class_names: tuple[Any, ...] | None = None,
) -> tuple[RuleKnowledgeBase, Any]:
    """Fit a shallow sklearn DecisionTree on train and export path rules.

    Each root-to-leaf path becomes one if-then rule. Provenance is disclosed
    as ``induced_tree`` (data-induced from Session train only).
    """
    if max_depth < 1:
        raise ValidationError("max_depth must be >= 1.")
    if min_samples_leaf < 1:
        raise ValidationError("min_samples_leaf must be >= 1.")
    x = frame[columns]
    if task == "classification":
        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
    elif task == "regression":
        tree = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
    else:
        raise ValidationError(f"Unknown symbolic task {task!r}.")
    tree.fit(x, y)
    rules = _export_tree_rules(
        tree,
        feature_names=columns,
        task=task,
        class_names=class_names,
        max_rules=max_rules,
    )
    default = _majority_or_mean(y, task=task, class_names=class_names)
    kb = RuleKnowledgeBase(
        rules=tuple(rules),
        default_consequent=default,
        columns_used=tuple(columns),
        disclosures=(
            "Rule provenance=induced_tree: sklearn DecisionTree fitted on "
            "Session train only, then exported as path rules. Not expert-declared.",
            f"Tree max_depth={max_depth}, min_samples_leaf={min_samples_leaf}.",
        ),
        provenance="induced_tree",
    )
    return kb, tree


def induce_decision_list(
    frame: pd.DataFrame,
    columns: list[str],
    y: np.ndarray,
    *,
    task: str = "classification",
    max_depth: int = 2,
    min_samples_leaf: int = 5,
    max_rules: int = 24,
    random_state: int | None = 0,
    class_names: tuple[Any, ...] | None = None,
) -> RuleKnowledgeBase:
    """Induce an ordered decision list via sequential covering (train-only).

    For each remaining uncovered class (classification) or residual slice
    (regression), fit a shallow tree stump / short path, take the single
    highest-support leaf rule for the target consequent, remove covered rows,
    and repeat. Ends with a majority / mean default rule.

    Honesty: RIPPER-/CN2-style covering lite — not a full ILP / Prolog inducer.
    """
    if task not in {"classification", "regression"}:
        raise ValidationError(f"Unknown symbolic task {task!r}.")
    if task == "regression":
        # Regression decision lists: bin targets into quantiles for covering,
        # then emit mean consequent per bin rule.
        return _induce_regression_list(
            frame,
            columns,
            y,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_rules=max_rules,
            random_state=random_state,
        )

    if class_names is None:
        raise ValidationError(
            "decision_list classification requires class_names."
        )
    remaining_mask = np.ones(len(frame), dtype=bool)
    rules: list[Rule] = []
    rng = np.random.default_rng(random_state)
    # Cover minority classes first for more informative lists.
    counts = {c: int(np.sum(y == i)) for i, c in enumerate(class_names)}

    class_order_idx = sorted(
        range(len(class_names)),
        key=lambda i: counts[class_names[i]],
    )

    for class_idx in class_order_idx:
        class_label = class_names[class_idx]
        for _ in range(max_rules):
            pos = remaining_mask & (y == class_idx)
            if int(np.sum(pos)) < min_samples_leaf:
                break
            if int(np.sum(remaining_mask)) < min_samples_leaf * 2:
                break
            sub = frame.loc[remaining_mask, columns]
            y_sub = y[remaining_mask]
            # Binary: target class vs rest among remaining.
            y_bin = (y_sub == class_idx).astype(int)
            if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                break
            stump = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
            )
            stump.fit(sub, y_bin)
            candidates = _export_tree_rules(
                stump,
                feature_names=columns,
                task="classification",
                class_names=("other", class_label),
                max_rules=max_rules,
                source="induced_list",
                only_consequent=class_label,
            )
            if not candidates:
                break
            # Pick highest-support rule for this class.
            candidates.sort(
                key=lambda r: (
                    -(r.support or 0),
                    -(r.confidence or 0.0),
                    r.rule_id,
                )
            )
            best = candidates[0]
            best = Rule(
                rule_id=f"list_{len(rules)}_{class_label}",
                antecedents=best.antecedents,
                consequent=class_label,
                priority=len(rules),
                source="induced_list",
                strength=1.0,
                hardness="hard",
                kind="classification",
                support=best.support,
                confidence=best.confidence,
            )
            rules.append(best)
            # Remove covered remaining rows that match this rule.
            covered = _mask_for_rule(frame, best) & remaining_mask
            # Only drop positives we intended to cover (and any others matched).
            if not covered.any():
                break
            remaining_mask = remaining_mask & ~covered
            if int(np.sum(remaining_mask)) == 0:
                break
        if len(rules) >= max_rules:
            break

    # Re-number priorities so earlier (rarer) covers stay first.
    ordered = []
    for i, rule in enumerate(rules[:max_rules]):
        ordered.append(
            Rule(
                rule_id=rule.rule_id,
                antecedents=rule.antecedents,
                consequent=rule.consequent,
                priority=len(rules) - i,
                source=rule.source,
                strength=rule.strength,
                hardness=rule.hardness,
                kind=rule.kind,
                support=rule.support,
                confidence=rule.confidence,
            )
        )
    default = class_names[int(np.bincount(y).argmax())]
    # Shuffle disclosure seed unused to keep deterministic API surface.
    _ = rng.integers(0, 1)
    return RuleKnowledgeBase(
        rules=tuple(ordered),
        default_consequent=default,
        columns_used=tuple(columns),
        disclosures=(
            "Rule provenance=induced_list: sequential covering on Session "
            "train only (shallow tree stumps per class). Not expert-declared; "
            "not a full RIPPER/ILP solver.",
            f"max_depth={max_depth}, min_samples_leaf={min_samples_leaf}, "
            f"max_rules={max_rules}.",
        ),
        provenance="induced_list",
    )


def _induce_regression_list(
    frame: pd.DataFrame,
    columns: list[str],
    y: np.ndarray,
    *,
    max_depth: int,
    min_samples_leaf: int,
    max_rules: int,
    random_state: int | None,
) -> RuleKnowledgeBase:
    """Regression covering via shallow tree leaves → mean consequents."""
    tree = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    tree.fit(frame[columns], y)
    rules = _export_tree_rules(
        tree,
        feature_names=columns,
        task="regression",
        class_names=None,
        max_rules=max_rules,
        source="induced_list",
    )
    # Sort by support descending → higher priority first.
    rules.sort(key=lambda r: (-(r.support or 0), r.rule_id))
    ordered = []
    for i, rule in enumerate(rules):
        ordered.append(
            Rule(
                rule_id=f"reglist_{i}",
                antecedents=rule.antecedents,
                consequent=rule.consequent,
                priority=len(rules) - i,
                source="induced_list",
                strength=1.0,
                hardness="hard",
                kind="regression",
                support=rule.support,
                confidence=rule.confidence,
            )
        )
    return RuleKnowledgeBase(
        rules=tuple(ordered),
        default_consequent=float(np.mean(y)),
        columns_used=tuple(columns),
        disclosures=(
            "Rule provenance=induced_list (regression): DecisionTreeRegressor "
            "leaves exported as ordered mean-consequent rules from train only.",
        ),
        provenance="induced_list",
    )


def _export_tree_rules(
    tree: Any,
    *,
    feature_names: list[str],
    task: str,
    class_names: tuple[Any, ...] | None,
    max_rules: int,
    source: str = "induced_tree",
    only_consequent: Any = None,
) -> list[Rule]:
    """Walk sklearn tree_ structure into Rule objects."""
    model = tree.tree_
    rules: list[Rule] = []

    def walk(node: int, antecedents: list[Predicate]) -> None:
        if len(rules) >= max_rules:
            return
        left = model.children_left[node]
        right = model.children_right[node]
        if left == right:  # leaf
            n_node = int(model.n_node_samples[node])
            if task == "classification":
                values = model.value[node][0]
                class_i = int(np.argmax(values))
                conf = float(values[class_i] / max(values.sum(), 1.0))
                if class_names is not None and class_i < len(class_names):
                    consequent = class_names[class_i]
                else:
                    consequent = class_i
                # Stump export uses binary class_names=("other", target).
                if only_consequent is not None and str(consequent) != str(
                    only_consequent
                ):
                    return
                kind = "classification"
            else:
                consequent = float(model.value[node][0][0])
                conf = 1.0
                kind = "regression"
            rules.append(
                Rule(
                    rule_id=f"{source}_leaf_{node}",
                    antecedents=tuple(antecedents),
                    consequent=consequent,
                    priority=0,
                    source=source,  # type: ignore[arg-type]
                    strength=1.0,
                    hardness="hard",
                    kind=kind,
                    support=n_node,
                    confidence=conf,
                )
            )
            return
        feat_i = int(model.feature[node])
        threshold = float(model.threshold[node])
        col = feature_names[feat_i]
        walk(
            left,
            antecedents
            + [Predicate(column=col, op="<=", value=threshold)],
        )
        walk(
            right,
            antecedents
            + [Predicate(column=col, op=">", value=threshold)],
        )

    walk(0, [])
    return rules


def _mask_for_rule(frame: pd.DataFrame, rule: Rule) -> np.ndarray:
    from buildml.symbolic.rules import evaluate_predicate

    mask = np.ones(len(frame), dtype=bool)
    for pred in rule.antecedents:
        mask &= evaluate_predicate(frame[pred.column], pred)
    return mask


def _majority_or_mean(
    y: np.ndarray,
    *,
    task: str,
    class_names: tuple[Any, ...] | None,
) -> Any:
    if task == "regression":
        return float(np.mean(y))
    counts = np.bincount(np.asarray(y, dtype=int))
    idx = int(np.argmax(counts))
    if class_names is not None and idx < len(class_names):
        return class_names[idx]
    return idx
