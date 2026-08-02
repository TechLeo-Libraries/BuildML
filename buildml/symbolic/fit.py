"""Fit / compile symbolic rule bases and neuro-symbolic hybrids (train-only)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition
from buildml.symbolic.features import (
    classification_accuracy,
    encode_classification_targets,
    matrix_from_frame,
    regression_metrics,
    regression_targets,
    resolve_symbolic_columns,
    train_partition_frame,
)
from buildml.symbolic.induce import induce_decision_list, induce_decision_tree_rules
from buildml.symbolic.results import (
    NeuroSymbolicFitResult,
    NeuroSymbolicPlan,
    SymbolicFitResult,
    SymbolicPlan,
)
from buildml.symbolic.rules import (
    Rule,
    RuleKnowledgeBase,
    fire_rules,
    parse_declared_rules,
    rule_feature_matrix,
    validate_rule_columns,
)
from buildml.symbolic.types import (
    BaseEstimatorName,
    NeuroSymbolicMode,
    SymbolicSource,
    SymbolicTask,
)


def fit_symbolic(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    source: SymbolicSource = "decision_tree",
    task: SymbolicTask | None = None,
    rules: Sequence[Mapping[str, Any] | Rule] | None = None,
    columns: list[str] | None = None,
    random_state: int | None = 0,
    max_depth: int = 4,
    min_samples_leaf: int = 5,
    max_rules: int = 32,
    default_consequent: Any = None,
    prefer_reduce_components: bool = True,
    reduce_plan: Any | None = None,
) -> tuple[SymbolicPlan, SymbolicFitResult]:
    """Compile or induce a symbolic rule knowledge base on Session train.

    Honesty
    -------
    Structured if-then rules over tabular columns. Sources:

    - ``declared`` — expert / caller rules (not learned)
    - ``decision_tree`` — sklearn DecisionTree path export (train-induced)
    - ``decision_list`` — sequential covering lite (train-induced)

    Not an AGI reasoner, Prolog engine, or Z3 SMT solver. Validation/test are
    never used for induction or compile-time statistics beyond disclosures.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    source_key = str(source).lower().replace("-", "_")
    if source_key not in {"declared", "decision_tree", "decision_list"}:
        raise ValidationError(
            f"Unknown symbolic source {source!r}; expected declared, "
            "decision_tree, or decision_list."
        )

    target = dataset.require_target()
    train = train_partition_frame(dataset, split_plan)
    resolved_task = _resolve_task(dataset, train[target], task)
    cols, used_reduce, disclosures = resolve_symbolic_columns(
        dataset,
        train,
        columns,
        reduce_plan=reduce_plan,
        prefer_reduce_components=prefer_reduce_components,
        target_column=target,
    )
    warnings: list[str] = []
    n_train = int(len(split_plan.train_indices))
    tree_est = None
    classes: tuple[Any, ...] | None = None
    label_encoder = None
    y_codes: np.ndarray | None = None

    if resolved_task == "classification":
        y_codes, label_encoder, classes = encode_classification_targets(
            train[target]
        )
        if len(classes) < 2:
            raise ValidationError(
                "Symbolic classification requires at least 2 classes."
            )
    else:
        y_num = regression_targets(train[target])

    if source_key == "declared":
        if not rules:
            raise ValidationError(
                "source='declared' requires a non-empty rules sequence."
            )
        kb = parse_declared_rules(
            rules, default_consequent=default_consequent, provenance="declared"
        )
        validate_rule_columns(kb, list(train.columns))
        disclosures.extend(kb.disclosures)
        disclosures.append(
            "Declared rules are expert/caller-supplied; they were not induced "
            "from Session train."
        )
    elif source_key == "decision_tree":
        if resolved_task == "classification":
            assert y_codes is not None
            kb, tree_est = induce_decision_tree_rules(
                train,
                cols,
                y_codes,
                task="classification",
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_rules=max_rules,
                random_state=random_state,
                class_names=classes,
            )
        else:
            kb, tree_est = induce_decision_tree_rules(
                train,
                cols,
                y_num,
                task="regression",
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_rules=max_rules,
                random_state=random_state,
            )
        if default_consequent is not None:
            kb = RuleKnowledgeBase(
                rules=kb.rules,
                default_consequent=default_consequent,
                columns_used=kb.columns_used,
                disclosures=kb.disclosures,
                provenance=kb.provenance,
            )
        disclosures.extend(kb.disclosures)
    else:  # decision_list
        if resolved_task == "classification":
            assert y_codes is not None and classes is not None
            kb = induce_decision_list(
                train,
                cols,
                y_codes,
                task="classification",
                max_depth=min(max_depth, 3),
                min_samples_leaf=min_samples_leaf,
                max_rules=max_rules,
                random_state=random_state,
                class_names=classes,
            )
        else:
            kb = induce_decision_list(
                train,
                cols,
                y_num,
                task="regression",
                max_depth=min(max_depth, 3),
                min_samples_leaf=min_samples_leaf,
                max_rules=max_rules,
                random_state=random_state,
            )
        disclosures.extend(kb.disclosures)

    disclosures.append(
        "Symbolic fit uses Session train only. Holdout is for "
        "evaluate_symbolic / predict_symbolic."
    )
    disclosures.append(
        "Honesty: structured tabular rules — not Prolog/Z3/AGI symbolic AI."
    )

    preds, _, _ = fire_rules(train, kb)
    train_acc: float | None = None
    if resolved_task == "classification":
        train_acc = classification_accuracy(train[target].tolist(), preds)
    else:
        # Store R2 as train_accuracy field analogue via metrics helper.
        try:
            y_hat = np.asarray(
                [float(p) if p is not None else float("nan") for p in preds],
                dtype=float,
            )
            train_acc = regression_metrics(
                regression_targets(train[target]), y_hat
            ).get("r2")
        except Exception:  # noqa: BLE001
            train_acc = None
            warnings.append("Could not compute train regression R2 for fit summary.")

    config = {
        "source": source_key,
        "task": resolved_task,
        "columns": cols,
        "random_state": random_state,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "max_rules": max_rules,
        "default_consequent": default_consequent,
        "prefer_reduce_components": prefer_reduce_components,
    }
    plan = SymbolicPlan(
        source=source_key,
        task=resolved_task,
        columns=tuple(cols),
        target_column=target,
        n_train_rows=n_train,
        n_rules=len(kb.rules),
        knowledge_base=kb,
        classes_=classes,
        tree_estimator_=tree_est,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        used_reduce_components=used_reduce,
        config=config,
    )
    result = SymbolicFitResult(
        source=source_key,
        task=resolved_task,
        n_train_rows=n_train,
        n_rules=len(kb.rules),
        columns=tuple(cols),
        target_column=target,
        provenance=kb.provenance,
        classes=classes,
        train_accuracy=train_acc,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
    return plan, result


def fit_neuro_symbolic(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    mode: NeuroSymbolicMode = "constraint_overlay",
    base_estimator: BaseEstimatorName = "logistic_regression",
    task: SymbolicTask | None = None,
    rules: Sequence[Mapping[str, Any] | Rule] | None = None,
    rule_source: SymbolicSource = "decision_tree",
    columns: list[str] | None = None,
    random_state: int | None = 0,
    soft_strength: float = 0.5,
    max_depth: int = 3,
    min_samples_leaf: int = 5,
    max_rules: int = 24,
    prefer_reduce_components: bool = True,
    reduce_plan: Any | None = None,
) -> tuple[NeuroSymbolicPlan, NeuroSymbolicFitResult]:
    """Fit a sklearn base model jointly with a symbolic rule component.

    Modes
    -----
    - ``constraint_overlay`` — train base model; at predict apply hard/soft rules
    - ``rules_as_features`` — fire rules as binary features; train on ``[X|R]``
    - ``constraint_repair`` — train base model; repair hard-constraint violations

    Rules may be expert-declared or train-induced (``rule_source``). Induction
    and base-model fitting use Session **train** only.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    mode_key = str(mode).lower().replace("-", "_")
    if mode_key not in {
        "constraint_overlay",
        "rules_as_features",
        "constraint_repair",
    }:
        raise ValidationError(
            f"Unknown neuro-symbolic mode {mode!r}; expected "
            "constraint_overlay, rules_as_features, or constraint_repair."
        )
    if not 0.0 <= float(soft_strength) <= 1.0:
        raise ValidationError("soft_strength must be in [0, 1].")

    target = dataset.require_target()
    train = train_partition_frame(dataset, split_plan)
    resolved_task = _resolve_task(dataset, train[target], task)
    cols, used_reduce, disclosures = resolve_symbolic_columns(
        dataset,
        train,
        columns,
        reduce_plan=reduce_plan,
        prefer_reduce_components=prefer_reduce_components,
        target_column=target,
    )
    warnings: list[str] = []
    n_train = int(len(split_plan.train_indices))

    # --- Rule component ---
    rule_src = str(rule_source).lower().replace("-", "_")
    classes: tuple[Any, ...] | None = None
    label_encoder = None
    if resolved_task == "classification":
        y_codes, label_encoder, classes = encode_classification_targets(
            train[target]
        )
        y_fit: Any = y_codes
    else:
        y_fit = regression_targets(train[target])

    if rule_src == "declared":
        if not rules:
            raise ValidationError(
                "neuro-symbolic rule_source='declared' requires rules=..."
            )
        kb = parse_declared_rules(rules, provenance="declared")
        validate_rule_columns(kb, list(train.columns))
        # Mark constraint kinds when hardness set.
        kb = _tag_constraint_kinds(kb)
        disclosures.append(
            "Neuro-symbolic rules are expert/caller-declared "
            "(provenance=declared)."
        )
    else:
        # Induce rules from train, optionally merge with declared constraints.
        if rule_src == "decision_tree":
            if resolved_task == "classification":
                kb, _ = induce_decision_tree_rules(
                    train,
                    cols,
                    y_fit,
                    task="classification",
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_rules=max_rules,
                    random_state=random_state,
                    class_names=classes,
                )
            else:
                kb, _ = induce_decision_tree_rules(
                    train,
                    cols,
                    y_fit,
                    task="regression",
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_rules=max_rules,
                    random_state=random_state,
                )
        elif rule_src == "decision_list":
            kb = induce_decision_list(
                train,
                cols,
                y_fit,
                task=resolved_task,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_rules=max_rules,
                random_state=random_state,
                class_names=classes,
            )
        else:
            raise ValidationError(f"Unknown rule_source {rule_source!r}.")
        disclosures.append(
            f"Neuro-symbolic rules induced via rule_source={rule_src} "
            "on Session train only (provenance disclosed on knowledge_base)."
        )
        if rules:
            declared = parse_declared_rules(rules, provenance="declared")
            merged = list(declared.rules) + list(kb.rules)
            kb = RuleKnowledgeBase(
                rules=tuple(merged),
                default_consequent=kb.default_consequent,
                columns_used=tuple(
                    sorted(set(declared.columns_used) | set(kb.columns_used))
                ),
                disclosures=declared.disclosures + kb.disclosures,
                provenance="mixed_declared_and_induced",
            )
            disclosures.append(
                "Additional declared rules were merged ahead of induced rules."
            )

    disclosures.extend(kb.disclosures)

    # --- Neural / sklearn component ---
    x = matrix_from_frame(train, cols)
    rule_feature_names: tuple[str, ...] = ()
    if mode_key == "rules_as_features":
        r_mat, r_names = rule_feature_matrix(train, kb)
        x = np.hstack([x, r_mat])
        rule_feature_names = tuple(r_names)
        disclosures.append(
            "Mode=rules_as_features: binary rule firings concatenated to "
            "numeric features before fitting the base estimator (train-only)."
        )
    else:
        disclosures.append(
            f"Mode={mode_key}: base estimator fitted on numeric features; "
            "rules applied at predict time as overlay/repair."
        )

    estimator = _build_base_estimator(
        base_estimator, task=resolved_task, random_state=random_state
    )
    estimator.fit(x, y_fit)

    train_score: float | None = None
    try:
        train_score = float(estimator.score(x, y_fit))
    except Exception:  # noqa: BLE001
        train_score = None

    disclosures.append(
        "Neuro-symbolic fit uses Session train only for induction and base "
        "estimator fitting. Holdout is for evaluate_neuro_symbolic / predict."
    )
    disclosures.append(
        "Honesty: sklearn + tabular rule hybrid — not a deep neuro-symbolic "
        "research platform, Prolog, or Z3."
    )

    config = {
        "mode": mode_key,
        "base_estimator": base_estimator,
        "task": resolved_task,
        "columns": cols,
        "random_state": random_state,
        "soft_strength": soft_strength,
        "rule_source": rule_src,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "max_rules": max_rules,
        "prefer_reduce_components": prefer_reduce_components,
    }
    plan = NeuroSymbolicPlan(
        mode=mode_key,
        base_estimator_name=str(base_estimator),
        task=resolved_task,
        columns=tuple(cols),
        target_column=target,
        n_train_rows=n_train,
        knowledge_base=kb,
        estimator_=estimator,
        label_encoder_=label_encoder,
        classes_=classes,
        rule_feature_names_=rule_feature_names,
        soft_strength=float(soft_strength),
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        used_reduce_components=used_reduce,
        config=config,
    )
    result = NeuroSymbolicFitResult(
        mode=mode_key,
        base_estimator_name=str(base_estimator),
        task=resolved_task,
        n_train_rows=n_train,
        n_rules=len(kb.rules),
        columns=tuple(cols),
        target_column=target,
        rule_provenance=kb.provenance,
        classes=classes,
        train_score=train_score,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
    return plan, result


def _tag_constraint_kinds(kb: RuleKnowledgeBase) -> RuleKnowledgeBase:
    """Ensure constraint-style rules keep kind=constraint when marked."""
    out: list[Rule] = []
    for rule in kb.rules:
        kind = rule.kind
        if kind == "constraint" or rule.hardness in {"hard", "soft"} and kind == "constraint":
            kind = "constraint"
        # Keep prediction rules as-is; callers may set kind=constraint explicitly.
        out.append(
            Rule(
                rule_id=rule.rule_id,
                antecedents=rule.antecedents,
                consequent=rule.consequent,
                priority=rule.priority,
                source=rule.source,
                strength=rule.strength,
                hardness=rule.hardness,
                kind=kind,
                support=rule.support,
                confidence=rule.confidence,
            )
        )
    return RuleKnowledgeBase(
        rules=tuple(out),
        default_consequent=kb.default_consequent,
        columns_used=kb.columns_used,
        disclosures=kb.disclosures,
        provenance=kb.provenance,
    )


def _resolve_task(
    dataset: Dataset, y: Any, task: SymbolicTask | None
) -> SymbolicTask:
    if task is not None:
        t = str(task).lower()
        if t not in {"classification", "regression"}:
            raise ValidationError(f"Unknown task {task!r}.")
        return t  # type: ignore[return-value]
    # Infer from target dtype / nunique.
    import pandas as pd

    series = y if isinstance(y, pd.Series) else pd.Series(y)
    if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 20:
        return "regression"
    return "classification"


def _build_base_estimator(
    name: BaseEstimatorName | str,
    *,
    task: str,
    random_state: int | None,
) -> Any:
    key = str(name).lower().replace("-", "_")
    if task == "classification":
        if key == "logistic_regression":
            return LogisticRegression(max_iter=500, random_state=random_state)
        if key == "random_forest":
            return RandomForestClassifier(
                n_estimators=50, random_state=random_state, max_depth=6
            )
        if key == "decision_tree":
            return DecisionTreeClassifier(
                max_depth=6, random_state=random_state
            )
        if key == "ridge":
            raise ValidationError(
                "base_estimator='ridge' is regression-only; use "
                "logistic_regression for classification."
            )
    else:
        if key == "ridge":
            return Ridge()
        if key == "random_forest":
            return RandomForestRegressor(
                n_estimators=50, random_state=random_state, max_depth=6
            )
        if key == "decision_tree":
            return DecisionTreeRegressor(
                max_depth=6, random_state=random_state
            )
        if key == "logistic_regression":
            raise ValidationError(
                "base_estimator='logistic_regression' is classification-only; "
                "use ridge for regression."
            )
    raise ValidationError(f"Unknown base_estimator {name!r}.")
