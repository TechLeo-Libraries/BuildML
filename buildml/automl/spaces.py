"""Default model-family and preprocess-recipe catalogs for AutoML.

These catalogs are deliberately finite and disclosed — not neural architecture
search, not an unbounded Autosklearn zoo, and not causal discovery.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from buildml.core.errors import ValidationError
from buildml.preprocess.fold import PreprocessRecipe

TaskName = Literal["classification", "regression"]
EstimatorFactory = Callable[[int | None], Any]


@dataclass(frozen=True, slots=True)
class ModelFamily:
    """One searchable estimator family with a modest param catalog."""

    name: str
    task: TaskName
    factory: EstimatorFactory
    param_grid: dict[str, list[Any]]
    param_distributions: dict[str, Any]

    def build(self, random_state: int | None = 0, **params: Any) -> Any:
        est = self.factory(random_state)
        if params:
            est.set_params(**params)
        return est


@dataclass(frozen=True, slots=True)
class RecipeStrategy:
    """Named fold-local preprocess recipe (strategy enums, not just knobs)."""

    name: str
    recipe: PreprocessRecipe
    description: str


def _rf_clf(rs: int | None) -> Any:
    return RandomForestClassifier(n_estimators=80, random_state=rs, n_jobs=1)


def _rf_reg(rs: int | None) -> Any:
    return RandomForestRegressor(n_estimators=80, random_state=rs, n_jobs=1)


def _gb_clf(rs: int | None) -> Any:
    return GradientBoostingClassifier(random_state=rs)


def _gb_reg(rs: int | None) -> Any:
    return GradientBoostingRegressor(random_state=rs)


def _lr(_rs: int | None) -> Any:
    return LogisticRegression(max_iter=500)


def _ridge(_rs: int | None) -> Any:
    return Ridge()


def _lasso(_rs: int | None) -> Any:
    return Lasso(max_iter=2000)


def _knn_clf(_rs: int | None) -> Any:
    return KNeighborsClassifier()


def _knn_reg(_rs: int | None) -> Any:
    return KNeighborsRegressor()


def _dt_clf(rs: int | None) -> Any:
    return DecisionTreeClassifier(random_state=rs)


def _dt_reg(rs: int | None) -> Any:
    return DecisionTreeRegressor(random_state=rs)


CLASSIFICATION_FAMILIES: tuple[ModelFamily, ...] = (
    ModelFamily(
        name="logistic",
        task="classification",
        factory=_lr,
        param_grid={"C": [0.1, 1.0, 10.0]},
        param_distributions={"C": [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]},
    ),
    ModelFamily(
        name="random_forest",
        task="classification",
        factory=_rf_clf,
        param_grid={"n_estimators": [40, 80], "max_depth": [None, 6]},
        param_distributions={
            "n_estimators": [40, 60, 80, 120],
            "max_depth": [None, 4, 6, 10],
            "min_samples_leaf": [1, 2, 4],
        },
    ),
    ModelFamily(
        name="gradient_boosting",
        task="classification",
        factory=_gb_clf,
        param_grid={"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]},
        param_distributions={
            "n_estimators": [40, 80, 120],
            "learning_rate": [0.03, 0.05, 0.1, 0.2],
            "max_depth": [2, 3, 4],
        },
    ),
    ModelFamily(
        name="knn",
        task="classification",
        factory=_knn_clf,
        param_grid={"n_neighbors": [3, 7]},
        param_distributions={"n_neighbors": [3, 5, 7, 11]},
    ),
    ModelFamily(
        name="decision_tree",
        task="classification",
        factory=_dt_clf,
        param_grid={"max_depth": [3, 6, None]},
        param_distributions={"max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 2, 4]},
    ),
)

REGRESSION_FAMILIES: tuple[ModelFamily, ...] = (
    ModelFamily(
        name="ridge",
        task="regression",
        factory=_ridge,
        param_grid={"alpha": [0.1, 1.0, 10.0]},
        param_distributions={"alpha": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]},
    ),
    ModelFamily(
        name="lasso",
        task="regression",
        factory=_lasso,
        param_grid={"alpha": [0.01, 0.1, 1.0]},
        param_distributions={"alpha": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]},
    ),
    ModelFamily(
        name="random_forest",
        task="regression",
        factory=_rf_reg,
        param_grid={"n_estimators": [40, 80], "max_depth": [None, 6]},
        param_distributions={
            "n_estimators": [40, 60, 80, 120],
            "max_depth": [None, 4, 6, 10],
            "min_samples_leaf": [1, 2, 4],
        },
    ),
    ModelFamily(
        name="gradient_boosting",
        task="regression",
        factory=_gb_reg,
        param_grid={"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]},
        param_distributions={
            "n_estimators": [40, 80, 120],
            "learning_rate": [0.03, 0.05, 0.1, 0.2],
            "max_depth": [2, 3, 4],
        },
    ),
    ModelFamily(
        name="knn",
        task="regression",
        factory=_knn_reg,
        param_grid={"n_neighbors": [3, 7]},
        param_distributions={"n_neighbors": [3, 5, 7, 11]},
    ),
    ModelFamily(
        name="decision_tree",
        task="regression",
        factory=_dt_reg,
        param_grid={"max_depth": [3, 6, None]},
        param_distributions={"max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 2, 4]},
    ),
)

DEFAULT_RECIPE_STRATEGIES: tuple[RecipeStrategy, ...] = (
    RecipeStrategy(
        name="passthrough",
        recipe=PreprocessRecipe(impute=None, scale=None, encode=None),
        description="No fold-local preprocess (use only when the design matrix is already clean).",
    ),
    RecipeStrategy(
        name="impute_median",
        recipe=PreprocessRecipe(impute="median", scale=None, encode=None),
        description="Median impute only.",
    ),
    RecipeStrategy(
        name="impute_scale",
        recipe=PreprocessRecipe(impute="median", scale="standard", encode=None),
        description="Median impute + standard scale.",
    ),
    RecipeStrategy(
        name="impute_minmax",
        recipe=PreprocessRecipe(impute="median", scale="minmax", encode=None),
        description="Median impute + minmax scale.",
    ),
    RecipeStrategy(
        name="impute_scale_onehot",
        recipe=PreprocessRecipe(impute="median", scale="standard", encode="onehot"),
        description="Median impute + standard scale + one-hot categoricals.",
    ),
    RecipeStrategy(
        name="impute_scale_ordinal",
        recipe=PreprocessRecipe(impute="median", scale="standard", encode="ordinal"),
        description="Median impute + standard scale + ordinal categoricals.",
    ),
    RecipeStrategy(
        name="impute_scale_select",
        recipe=PreprocessRecipe(
            impute="median",
            scale="standard",
            encode="onehot",
            select="univariate",
            select_k=10,
        ),
        description="Impute/scale/one-hot plus fold-local SelectKBest.",
    ),
    RecipeStrategy(
        name="impute_scale_variance",
        recipe=PreprocessRecipe(
            impute="median",
            scale="standard",
            encode="onehot",
            select="variance",
            select_threshold=0.0,
        ),
        description="Impute/scale/one-hot plus fold-local variance threshold.",
    ),
)


def families_for_task(
    task: TaskName,
    *,
    names: tuple[str, ...] | list[str] | None = None,
    max_families: int | None = None,
) -> list[ModelFamily]:
    catalog = CLASSIFICATION_FAMILIES if task == "classification" else REGRESSION_FAMILIES
    if names is None:
        chosen = list(catalog)
    else:
        wanted = {str(n).strip().lower() for n in names}
        chosen = [f for f in catalog if f.name in wanted]
        missing = wanted - {f.name for f in chosen}
        if missing:
            available = sorted(f.name for f in catalog)
            raise ValidationError(
                f"Unknown AutoML families for {task}: {sorted(missing)}. "
                f"Available: {available}"
            )
        if not chosen:
            raise ValidationError(f"No AutoML families selected for {task}.")
    if max_families is not None:
        if max_families < 1:
            raise ValidationError("max_families must be >= 1")
        chosen = chosen[: int(max_families)]
    return chosen


def recipe_strategies(
    *,
    include_recipe_search: bool,
    fixed: PreprocessRecipe | None = None,
    max_strategies: int | None = None,
) -> list[RecipeStrategy]:
    """Resolve the recipe catalog for a search run.

    When ``include_recipe_search`` is False, returns a single strategy from
    ``fixed`` (or passthrough). When True, returns the default discrete catalog
    (optionally capped), ignoring Session-global plans — fold-local only.
    """
    if not include_recipe_search:
        recipe = (
            fixed
            if fixed is not None
            else PreprocessRecipe(impute=None, scale=None, encode=None)
        )
        return [
            RecipeStrategy(
                name="fixed",
                recipe=recipe,
                description="Caller-fixed PreprocessRecipe (no strategy search).",
            )
        ]
    strategies = list(DEFAULT_RECIPE_STRATEGIES)
    if fixed is not None and not fixed.is_empty():
        # Keep caller recipe as an extra candidate alongside the catalog.
        strategies.insert(
            0,
            RecipeStrategy(
                name="caller_base",
                recipe=fixed,
                description="Caller-provided base PreprocessRecipe.",
            ),
        )
    if max_strategies is not None:
        if max_strategies < 1:
            raise ValidationError("max_recipe_strategies must be >= 1")
        strategies = strategies[: int(max_strategies)]
    return strategies


def family_by_name(task: TaskName, name: str) -> ModelFamily:
    for fam in families_for_task(task):
        if fam.name == name:
            return fam
    raise ValidationError(f"Unknown family {name!r} for task {task!r}")
