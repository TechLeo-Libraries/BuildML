"""Default model-family and preprocess-recipe catalogs for AutoML.

These catalogs are deliberately finite and disclosed: not neural architecture
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
        """Instantiate this family with optional hyperparameters applied.

        Calls the family factory, then merges ``params`` via ``set_params`` so
        AutoML trials can materialize concrete estimators.

        Parameters
        ----------
        random_state:
            Seed passed to the family factory when supported.
        **params:
            Hyperparameter overrides merged via ``estimator.set_params``.

        Returns
        -------
        estimator
            Fresh sklearn-compatible estimator ready for fold-local fitting.
        """
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
        name="impute_mean",
        recipe=PreprocessRecipe(impute="mean", scale=None, encode=None),
        description="Mean impute only.",
    ),
    RecipeStrategy(
        name="impute_scale",
        recipe=PreprocessRecipe(impute="median", scale="standard", encode=None),
        description="Median impute + standard scale.",
    ),
    RecipeStrategy(
        name="impute_most_frequent_scale",
        recipe=PreprocessRecipe(impute="most_frequent", scale="standard", encode=None),
        description="Most-frequent impute + standard scale.",
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
        name="impute_encode_onehot",
        recipe=PreprocessRecipe(impute="median", scale=None, encode="onehot"),
        description="Median impute + one-hot (no scale).",
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
        name="impute_scale_select_k20",
        recipe=PreprocessRecipe(
            impute="median",
            scale="standard",
            encode="onehot",
            select="univariate",
            select_k=20,
        ),
        description="Impute/scale/one-hot plus SelectKBest(k=20).",
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
    RecipeStrategy(
        name="impute_minmax_onehot",
        recipe=PreprocessRecipe(impute="median", scale="minmax", encode="onehot"),
        description="Median impute + minmax + one-hot.",
    ),
)


def families_for_task(
    task: TaskName,
    *,
    names: tuple[str, ...] | list[str] | None = None,
    max_families: int | None = None,
    include_industry: bool = True,
) -> list[ModelFamily]:
    """Return searchable model families for a classification or regression task.

    Optionally filters to named families, caps the catalog size, and extends
    with industry GBDT families when installed.

    Parameters
    ----------
    task:
        ``classification`` or ``regression``.
    names:
        Optional subset of family names to include; ``None`` uses the full catalog.
    max_families:
        Optional cap on the number of families returned (preserves catalog order).
    include_industry:
        When True, append LightGBM/XGBoost/CatBoost families when discoverable.

    Returns
    -------
    list[ModelFamily]
        Resolved family objects ready for AutoML candidate generation.

    Raises
    ------
    ValidationError
        When ``names`` contains unknown families or ``max_families`` is invalid.
    """
    catalog = list(
        CLASSIFICATION_FAMILIES if task == "classification" else REGRESSION_FAMILIES
    )
    if include_industry:
        industry = (
            _industry_classification_families()
            if task == "classification"
            else _industry_regression_families()
        )
        catalog.extend(industry)
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
    """Resolve the fold-local recipe catalog for one AutoML search run.

    When ``include_recipe_search`` is False, returns a single strategy from
    ``fixed`` (or passthrough). When True, returns the default discrete catalog
    (optionally capped), ignoring Session-global plans: fold-local only.

    Parameters
    ----------
    include_recipe_search:
        When False, skip strategy search and use ``fixed`` or passthrough.
    fixed:
        Caller-provided base recipe included when recipe search is enabled.
    max_strategies:
        Optional cap on the number of strategies returned.

    Returns
    -------
    list[RecipeStrategy]
        Named recipe strategies for joint family+recipe AutoML search.

    Raises
    ------
    ValidationError
        When ``max_strategies`` is less than one.
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
    """Look up one model family by name for a task.

    Scans the full catalog from :func:`families_for_task` including industry
    GBDT families when installed.

    Parameters
    ----------
    task:
        ``classification`` or ``regression``.
    name:
        Family identifier (e.g. ``logistic``, ``lightgbm``).

    Returns
    -------
    ModelFamily
        Matching family from :func:`families_for_task`.

    Raises
    ------
    ValidationError
        When no family with ``name`` exists for ``task``.
    """
    for fam in families_for_task(task, include_industry=True):
        if fam.name == name:
            return fam
    raise ValidationError(f"Unknown family {name!r} for task {task!r}")


def _industry_classification_families() -> tuple[ModelFamily, ...]:
    from buildml.automl.extras import (
        catboost_available,
        lightgbm_available,
        xgboost_available,
    )

    out: list[ModelFamily] = []
    if lightgbm_available():
        out.append(
            ModelFamily(
                name="lightgbm",
                task="classification",
                factory=_lgb_clf,
                param_grid={"n_estimators": [80, 120], "learning_rate": [0.05, 0.1]},
                param_distributions={
                    "n_estimators": [60, 100, 140],
                    "learning_rate": [0.03, 0.05, 0.1],
                    "num_leaves": [15, 31, 63],
                },
            )
        )
    if xgboost_available():
        out.append(
            ModelFamily(
                name="xgboost",
                task="classification",
                factory=_xgb_clf,
                param_grid={"n_estimators": [80, 120], "max_depth": [3, 6]},
                param_distributions={
                    "n_estimators": [60, 100, 140],
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.03, 0.05, 0.1],
                },
            )
        )
    if catboost_available():
        out.append(
            ModelFamily(
                name="catboost",
                task="classification",
                factory=_cat_clf,
                param_grid={"iterations": [80, 120], "depth": [4, 6]},
                param_distributions={
                    "iterations": [60, 100, 140],
                    "depth": [4, 6, 8],
                    "learning_rate": [0.03, 0.05, 0.1],
                },
            )
        )
    return tuple(out)


def _industry_regression_families() -> tuple[ModelFamily, ...]:
    from buildml.automl.extras import (
        catboost_available,
        lightgbm_available,
        xgboost_available,
    )

    out: list[ModelFamily] = []
    if lightgbm_available():
        out.append(
            ModelFamily(
                name="lightgbm",
                task="regression",
                factory=_lgb_reg,
                param_grid={"n_estimators": [80, 120], "learning_rate": [0.05, 0.1]},
                param_distributions={
                    "n_estimators": [60, 100, 140],
                    "learning_rate": [0.03, 0.05, 0.1],
                    "num_leaves": [15, 31, 63],
                },
            )
        )
    if xgboost_available():
        out.append(
            ModelFamily(
                name="xgboost",
                task="regression",
                factory=_xgb_reg,
                param_grid={"n_estimators": [80, 120], "max_depth": [3, 6]},
                param_distributions={
                    "n_estimators": [60, 100, 140],
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.03, 0.05, 0.1],
                },
            )
        )
    if catboost_available():
        out.append(
            ModelFamily(
                name="catboost",
                task="regression",
                factory=_cat_reg,
                param_grid={"iterations": [80, 120], "depth": [4, 6]},
                param_distributions={
                    "iterations": [60, 100, 140],
                    "depth": [4, 6, 8],
                    "learning_rate": [0.03, 0.05, 0.1],
                },
            )
        )
    return tuple(out)


def _lgb_clf(rs: int | None) -> Any:
    from buildml.automl.extras import require_lightgbm

    lgb = require_lightgbm()
    return lgb.LGBMClassifier(
        n_estimators=100, random_state=rs, verbosity=-1, n_jobs=1
    )


def _lgb_reg(rs: int | None) -> Any:
    from buildml.automl.extras import require_lightgbm

    lgb = require_lightgbm()
    return lgb.LGBMRegressor(n_estimators=100, random_state=rs, verbosity=-1, n_jobs=1)


def _xgb_clf(rs: int | None) -> Any:
    from buildml.automl.extras import require_xgboost

    xgb = require_xgboost()
    return xgb.XGBClassifier(
        n_estimators=100, random_state=rs, verbosity=0, n_jobs=1, use_label_encoder=False
    )


def _xgb_reg(rs: int | None) -> Any:
    from buildml.automl.extras import require_xgboost

    xgb = require_xgboost()
    return xgb.XGBRegressor(n_estimators=100, random_state=rs, verbosity=0, n_jobs=1)


def _cat_clf(rs: int | None) -> Any:
    from buildml.automl.extras import require_catboost

    cb = require_catboost()
    return cb.CatBoostClassifier(
        iterations=100, random_state=rs, verbose=False, allow_writing_files=False
    )


def _cat_reg(rs: int | None) -> Any:
    from buildml.automl.extras import require_catboost

    cb = require_catboost()
    return cb.CatBoostRegressor(
        iterations=100, random_state=rs, verbose=False, allow_writing_files=False
    )
