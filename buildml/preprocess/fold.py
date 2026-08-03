"""Fold-local preprocess helpers for leakage-safe cross-validation."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
    f_classif,
    f_regression,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler

from buildml.core.errors import ValidationError
from buildml.preprocess.encode import INFREQUENT_LABEL, _smoothed_means
from buildml.preprocess.text import (
    _as_text,
    _build_vectorizer,
    _feature_names_for_column,
)

ImputeStrategy = Literal["mean", "median", "most_frequent", "constant"]
ScaleMethod = Literal["standard", "minmax"]
EncodeMethod = Literal["onehot", "ordinal", "infrequent", "target"]
SelectStrategy = Literal["variance", "univariate", "model"]
OutlierMethod = Literal["iqr", "zscore"]
OutlierAction = Literal["detect", "cap"]
BinStrategy = Literal["quantile", "uniform"]
BinEncodeAs = Literal["ordinal", "onehot"]
TextMethod = Literal["count", "tfidf", "hashing"]
ReduceMethod = Literal["pca"]

# Fold-local order mirrors score-time plan replay where feasible.
# Session-global-only steps (not in PreprocessRecipe): resample (train row
# rewrite), custom transforms, and any Session.* plan fitted on the full
# train partition before CV.
FOLD_LOCAL_ORDER = (
    "dates",
    "text",
    "outliers",
    "impute",
    "encode",
    "binning",
    "scale",
    "reduce",
    "select",
)
SESSION_GLOBAL_ONLY_STEPS = (
    "resample",
    "Session.apply_custom_transform (registered callables stay Session-global)",
    "Session.text_features / Session.reduce_dimensions when not expressed "
    "in PreprocessRecipe",
    "Session.extract_dates / Session.bin when not expressed in PreprocessRecipe",
    "Session.impute / encode / scale / handle_outliers / select_features "
    "fitted on the full train partition before CV",
)

# Fold-local scalar knobs safe to sweep inside nested CV / search.
# Strategy enums (impute/scale/encode/select/text/reduce/...) stay on the base
# recipe; only these numeric/categorical controls are first-class search knobs.
SAFE_RECIPE_KNOBS = frozenset(
    {
        "select_k",
        "select_threshold",
        "select_score_func",
        "n_bins",
        "binning_encode_as",
        "min_frequency",
        "iqr_multiplier",
        "zscore_threshold",
        "outlier_action",
        "target_smoothing",
        "fill_value",
        "date_include_time",
        "date_drop_original",
        "text_max_features",
        "reduce_n_components",
    }
)


@dataclass(slots=True)
class PreprocessRecipe:
    """Unfitted preprocess steps refit on each CV fold's training rows.

    Parameters
    ----------
    impute:
        Numeric imputation strategy, or ``None`` to skip.
    scale:
        Numeric scaling method, or ``None`` to skip.
    encode:
        Categorical encoding method, or ``None`` to skip. ``infrequent`` pools
        rare fold-train levels before one-hot. ``target`` fits smoothed category
        means on fold-train labels only (eval rows never contribute).
    select:
        Optional fold-local feature selection on the transformed numeric matrix.
        ``variance`` uses VarianceThreshold; ``univariate`` uses SelectKBest;
        ``model`` uses SelectFromModel fit on fold-train only.
    outliers:
        Optional fold-local outlier fences (``iqr`` / ``zscore``). When set,
        fences are fit on fold-train and applied to fold-train and fold-eval
        before later numeric steps. Only ``detect`` and ``cap`` are supported
        inside CV (row drops would rewrite fold membership).
    binning:
        Optional fold-local discretization (``quantile`` / ``uniform``). Edges
        are learned on fold-train finite values only.
    dates:
        When ``True``, expand datetime columns into calendar (and optional
        clock) parts before later steps. Expansion is row-wise deterministic;
        including it in the recipe avoids a Session-global ``extract_dates``
        before CV.
    text:
        Optional fold-local text vectorizer (``count`` / ``tfidf`` / ``hashing``).
        Vocabulary or hashing width is fit on fold-train documents only.
    reduce:
        Optional fold-local dimensionality reduction (``pca``). The rotation is
        fit on fold-train numeric columns after scale (when scale is set).
    impute_columns / scale_columns / encode_columns / select_columns /
    outlier_columns / binning_columns / date_columns / text_columns /
    reduce_columns:
        Optional explicit column lists. When omitted, columns are inferred
        from dtypes of the fold-train frame (select applies after encode/bin/
        reduce).
    fill_value:
        Constant fill when ``impute='constant'``.
    min_frequency:
        Infrequent-pooling threshold (fraction in (0, 1) or absolute count).
    select_threshold / select_k / select_score_func / select_estimator:
        Feature-selection controls. ``select_estimator`` is used only for
        ``select='model'``; when omitted, a small default classifier/regressor
        is chosen from fold-train label cardinality.
    outlier_action / iqr_multiplier / zscore_threshold:
        Outlier controls (``outlier_action`` defaults to ``cap``).
    n_bins / binning_encode_as:
        Binning controls (defaults: 5 bins, ordinal codes).
    date_include_time / date_drop_original:
        Date-expansion controls.
    text_max_features / text_ngram_range / text_drop_input:
        Text vectorizer controls (defaults: 128 features, unigrams, drop input).
    reduce_n_components / reduce_prefix / reduce_drop_input:
        PCA controls. ``reduce_n_components`` may be an int, a float variance
        target in (0, 1], or ``None`` for ``min(n_samples, n_features)``.
    target_smoothing:
        Additive smoothing strength for fold-local target encoding.
    Notes
    -----
    Session-global ``encode(method='target')`` writes out-of-fold values on the
    full train partition. Fold-local target encoding instead fits means on each
    fold's training rows and applies those frozen means to fold-train and
    fold-eval — the CV eval fold never supplies label statistics.

    **Still Session-global only:** ``resample`` (train-row rewrite),
    ``Session.apply_custom_transform`` (registered callables), and any Session
    plan fitted on the full train partition before CV. Prefer putting
    dates/text/binning/impute/encode/scale/reduce/select/outliers in this
    recipe for selection-time honesty.
    """

    impute: ImputeStrategy | None = "median"
    scale: ScaleMethod | None = None
    encode: EncodeMethod | None = None
    select: SelectStrategy | None = None
    outliers: OutlierMethod | None = None
    binning: BinStrategy | None = None
    dates: bool = False
    text: TextMethod | None = None
    reduce: ReduceMethod | None = None
    impute_columns: tuple[str, ...] | None = None
    scale_columns: tuple[str, ...] | None = None
    encode_columns: tuple[str, ...] | None = None
    select_columns: tuple[str, ...] | None = None
    outlier_columns: tuple[str, ...] | None = None
    binning_columns: tuple[str, ...] | None = None
    date_columns: tuple[str, ...] | None = None
    text_columns: tuple[str, ...] | None = None
    reduce_columns: tuple[str, ...] | None = None
    fill_value: Any | None = None
    min_frequency: float | int = 0.05
    select_threshold: float = 0.0
    select_k: int = 10
    select_score_func: Literal["f_classif", "f_regression"] = "f_classif"
    select_estimator: Any | None = None
    outlier_action: OutlierAction = "cap"
    iqr_multiplier: float = 1.5
    zscore_threshold: float = 3.0
    n_bins: int = 5
    binning_encode_as: BinEncodeAs = "ordinal"
    date_include_time: bool = False
    date_drop_original: bool = False
    text_max_features: int | None = 128
    text_ngram_range: tuple[int, int] = (1, 1)
    text_drop_input: bool = True
    reduce_n_components: int | float | None = None
    reduce_prefix: str = "pc"
    reduce_drop_input: bool = True
    target_smoothing: float = 10.0

    def to_dict(self) -> dict[str, Any]:
        """Return the recipe's settings as plain JSON-safe values.

        Belongs in a model card and in search results: the recipe is part of
        what produced a score, so a score quoted without it is not
        reproducible.

        Returns
        -------
        dict
            Every setting in plain-data form. ``select_estimator`` is reduced
            to its class name, since an estimator instance does not serialise.
        """
        estimator_name = None
        if self.select_estimator is not None:
            estimator_name = type(self.select_estimator).__name__
        return {
            "impute": self.impute,
            "scale": self.scale,
            "encode": self.encode,
            "select": self.select,
            "outliers": self.outliers,
            "binning": self.binning,
            "dates": self.dates,
            "text": self.text,
            "reduce": self.reduce,
            "impute_columns": None if self.impute_columns is None else list(self.impute_columns),
            "scale_columns": None if self.scale_columns is None else list(self.scale_columns),
            "encode_columns": None if self.encode_columns is None else list(self.encode_columns),
            "select_columns": None if self.select_columns is None else list(self.select_columns),
            "outlier_columns": None if self.outlier_columns is None else list(self.outlier_columns),
            "binning_columns": (
                None if self.binning_columns is None else list(self.binning_columns)
            ),
            "date_columns": None if self.date_columns is None else list(self.date_columns),
            "text_columns": None if self.text_columns is None else list(self.text_columns),
            "reduce_columns": None if self.reduce_columns is None else list(self.reduce_columns),
            "fill_value": self.fill_value,
            "min_frequency": self.min_frequency,
            "select_threshold": self.select_threshold,
            "select_k": self.select_k,
            "select_score_func": self.select_score_func,
            "select_estimator": estimator_name,
            "outlier_action": self.outlier_action,
            "iqr_multiplier": self.iqr_multiplier,
            "zscore_threshold": self.zscore_threshold,
            "n_bins": self.n_bins,
            "binning_encode_as": self.binning_encode_as,
            "date_include_time": self.date_include_time,
            "date_drop_original": self.date_drop_original,
            "text_max_features": self.text_max_features,
            "text_ngram_range": list(self.text_ngram_range),
            "text_drop_input": self.text_drop_input,
            "reduce_n_components": self.reduce_n_components,
            "reduce_prefix": self.reduce_prefix,
            "reduce_drop_input": self.reduce_drop_input,
            "target_smoothing": self.target_smoothing,
            "fold_local_order": list(FOLD_LOCAL_ORDER),
            "session_global_only": list(SESSION_GLOBAL_ONLY_STEPS),
        }

    def is_empty(self) -> bool:
        """Report whether this recipe would do anything at all.

        Callers use this to skip building a preprocessor when every step is
        switched off, which avoids wrapping the estimator in a pipeline that
        only passes data through.

        Returns
        -------
        bool
            ``True`` when no step is enabled. Column lists and tuning knobs do
            not count — a recipe naming ``scale_columns`` but leaving ``scale``
            as ``None`` is still empty.
        """
        return (
            self.impute is None
            and self.scale is None
            and self.encode is None
            and self.select is None
            and self.outliers is None
            and self.binning is None
            and not self.dates
            and self.text is None
            and self.reduce is None
        )

    def requires_target(self) -> bool:
        """Report whether fitting this recipe needs the labels.

        Most preprocessing looks only at the features, but target encoding and
        the supervised selection strategies read the labels — which is exactly
        why they must be fitted per fold rather than once up front. Callers
        check this before fitting so a missing label vector produces a clear
        error rather than a confusing one deeper in.

        Returns
        -------
        bool
            ``True`` when the recipe uses target encoding or univariate or
            model-based selection.
        """
        return self.encode == "target" or self.select in {"univariate", "model"}

    def with_knobs(self, knobs: dict[str, Any]) -> PreprocessRecipe:
        """Return a copy with tuning knobs overridden, leaving this one unchanged.

        This is how a hyperparameter search explores preprocessing settings.
        Each candidate configuration produces a fresh recipe, so the search can
        try ``select_k=10`` and ``select_k=30`` without the two runs
        interfering — and, because the copy is made per fold, the choice stays
        fold-local and does not leak across the search.

        Parameters
        ----------
        knobs:
            Settings to override, named from :data:`SAFE_RECIPE_KNOBS`. Only
            numeric and categorical knobs are permitted, not the strategy
            fields themselves — you can search over ``n_bins``, but switching
            ``binning`` from quantile to uniform is a change to the base recipe.
            An unknown key raises rather than being ignored, so a typo in a
            search space is caught immediately instead of silently doing
            nothing.

        Returns
        -------
        PreprocessRecipe
            A new recipe with the overrides applied. This instance is
            unmodified; the same base recipe is reused across every fold and
            candidate.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            A key is not in :data:`SAFE_RECIPE_KNOBS`. The message lists what
            is allowed.

        Examples
        --------
        >>> tuned = recipe.with_knobs({"select_k": 25, "n_bins": 8})  # doctest: +SKIP

        See Also
        --------
        buildml.session.Session.nested_cv_score : Searches over these knobs honestly.
        """
        if not knobs:
            return self
        unknown = sorted(set(knobs) - SAFE_RECIPE_KNOBS)
        if unknown:
            raise ValidationError(
                f"Unsupported recipe knobs: {unknown}. "
                f"Allowed: {sorted(SAFE_RECIPE_KNOBS)}"
            )
        field_names = {f.name for f in fields(self)}
        updates = {k: v for k, v in knobs.items() if k in field_names}
        return replace(self, **updates)


class FoldLocalPreprocessor:
    """A whole preprocessing sequence, refitted from scratch inside a single fold.

    This is what makes cross-validated scores honest. If you impute, encode,
    and scale once over the training set and *then* cross-validate, every fold's
    evaluation rows contributed to the medians, vocabularies, and standard
    deviations that shaped the fold's training rows. The score comes out
    optimistic, and the effect is largest exactly when your dataset is
    small — when you most needed cross-validation to be trustworthy.

    An instance of this class is fitted separately for each fold, seeing only
    that fold's training rows, and then applied to both halves. It follows the
    scikit-learn fit-and-transform convention, so it drops into a
    :class:`~sklearn.pipeline.Pipeline` alongside the estimator.

    The steps run in a fixed order chosen so each one gets input it can handle:
    dates expand first, then text vectorises, outliers are fenced, values are
    imputed, numbers are binned, categories are encoded, features are scaled,
    dimensions are reduced, and selection happens last on the finished numeric
    matrix.

    Parameters
    ----------
    recipe:
        Which steps to run and how to configure them.

    Notes
    -----
    Not every session-level operation can be made fold-local. Resampling
    rewrites the row set, and registered custom transforms are arbitrary
    callables, so both remain session-wide. Everything else belongs in the
    recipe if you intend to cross-validate.

    See Also
    --------
    PreprocessRecipe : Describes the steps this executes.
    build_fold_preprocessor : Constructs and fits one in a single call.
    """

    def __init__(self, recipe: PreprocessRecipe) -> None:
        """Prepare an unfitted preprocessor for the given recipe.

        Nothing is learned here; every fitted attribute stays empty until
        :meth:`fit` runs. Construct one of these per fold rather than reusing a
        fitted instance, which would carry another fold's statistics.

        Parameters
        ----------
        recipe:
            The steps and settings to apply.
        """
        self.recipe = recipe
        self._imputer: Any | None = None
        self._scaler: Any | None = None
        self._impute_columns: list[str] = []
        self._scale_columns: list[str] = []
        self._encode_columns: list[str] = []
        self._passthrough_columns: list[str] = []
        self._encoder: Any | None = None
        self._infrequent_maps: dict[str, list[str]] = {}
        self._target_maps: dict[str, dict[str, float]] = {}
        self._target_prior: float | None = None
        self._feature_names_: list[str] = []
        self._selector: Any | None = None
        self._selected_features_: list[str] = []
        self._outlier_columns: list[str] = []
        self._outlier_lower: dict[str, float] = {}
        self._outlier_upper: dict[str, float] = {}
        self._date_columns: list[str] = []
        self._date_created: list[str] = []
        self._binning_columns: list[str] = []
        self._binning_edges: dict[str, list[float]] = {}
        self._binning_labels: dict[str, list[str]] = {}
        self._text_columns: list[str] = []
        self._text_feature_names: list[str] = []
        self._text_vectorizers: dict[str, Any] = {}
        self._text_n_features: dict[str, int] = {}
        self._reduce_columns: list[str] = []
        self._reduce_feature_names: list[str] = []
        self._reducer: Any | None = None

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series | None = None) -> FoldLocalPreprocessor:
        """Learn every step's parameters from this fold's training rows.

        Runs the recipe's steps in order, each fitting on the output of the
        last, and stores what each one learned. Only the rows passed here are
        seen, which is the whole point — the fold's evaluation rows must not
        influence any of it.

        Parameters
        ----------
        x_train:
            Feature rows for this fold's training half.
        y_train:
            Labels for those rows. Required when the recipe uses target
            encoding or supervised selection, and unused otherwise. Passing
            labels that include the evaluation rows would defeat the entire
            arrangement.

        Returns
        -------
        FoldLocalPreprocessor
            This instance, fitted, so calls can chain in the scikit-learn
            style.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            Labels are needed but absent; an outlier action of ``'drop'`` was
            requested, which cannot work inside a fold because removing rows
            would change fold membership; ``n_bins`` is below 2; text settings
            are malformed; or the reduce method or prefix is unsupported.

        Notes
        -----
        Fitting is per fold, so its cost multiplies by the fold count. A recipe
        with text vectorisation and model-based selection can dominate the
        runtime of a search — worth knowing before setting fifty candidates
        against ten folds.
        """
        recipe = self.recipe
        if recipe.requires_target() and y_train is None:
            raise ValidationError(
                "PreprocessRecipe encode='target' or select in "
                "{'univariate','model'} requires fold-train labels"
            )
        if recipe.outliers is not None and recipe.outlier_action not in {"detect", "cap"}:
            raise ValidationError(
                "Fold-local outliers support only action='detect' or 'cap' "
                "(row drops are not applied inside CV folds)."
            )
        if recipe.binning is not None and recipe.n_bins < 2:
            raise ValidationError("n_bins must be at least 2 for fold-local binning")
        if recipe.text is not None:
            if recipe.text_max_features is not None and recipe.text_max_features < 1:
                raise ValidationError("text_max_features must be >= 1 when provided")
            ngram = recipe.text_ngram_range
            if len(ngram) != 2 or ngram[0] < 1 or ngram[1] < ngram[0]:
                raise ValidationError(
                    "text_ngram_range must be a (min_n, max_n) pair with 1 <= min_n <= max_n"
                )
        if recipe.reduce is not None and recipe.reduce != "pca":
            raise ValidationError(f"Unsupported fold-local reduce method '{recipe.reduce}'")
        if recipe.reduce is not None:
            prefix = str(recipe.reduce_prefix)
            if not prefix or not prefix.replace("_", "").isalnum():
                raise ValidationError("reduce_prefix must be a non-empty alphanumeric token")
        work = x_train.copy()
        if recipe.dates:
            self._fit_dates(work)
            work = self._apply_dates(work)
        if recipe.text is not None:
            self._fit_text(work)
            work = self._apply_text(work)
        else:
            self._text_columns = []
            self._text_vectorizers = {}
            self._text_feature_names = []
            self._text_n_features = {}
        if recipe.outliers is not None:
            self._fit_outliers(work)
            work = self._apply_outliers(work)
        self._impute_columns = (
            _resolve_numeric(work, recipe.impute_columns) if recipe.impute is not None else []
        )
        if recipe.encode is not None:
            categorical = _resolve_categorical(work, recipe.encode_columns)
        else:
            categorical = []
        self._encode_columns = list(categorical)
        if self._impute_columns and recipe.impute is not None:
            self._imputer = SimpleImputer(strategy=recipe.impute, fill_value=recipe.fill_value)
            self._imputer.fit(work[self._impute_columns])
            work = self._apply_impute(work)
        else:
            self._imputer = None
        # Resolve scale targets before encode so one-hot dummies are not auto-scaled.
        pending_scale = (
            _resolve_numeric(work, recipe.scale_columns) if recipe.scale is not None else []
        )
        if categorical and recipe.encode is not None:
            self._fit_encoder(work, y_train)
            work = self._apply_encode_inplace(work)
        else:
            self._encoder = None
        if recipe.binning is not None:
            self._fit_binning(work)
            work = self._apply_binning(work)
        else:
            self._binning_columns = []
        self._scale_columns = _remap_scaled_columns(pending_scale, work, self._binning_columns)
        if self._scale_columns and recipe.scale is not None:
            self._scaler = _make_scaler(recipe.scale)
            self._scaler.fit(work[self._scale_columns])
            work = self._apply_scale(work)
        else:
            self._scaler = None
        if recipe.reduce is not None:
            self._fit_reduce(work)
            work = self._apply_reduce(work)
        else:
            self._reduce_columns = []
            self._reduce_feature_names = []
            self._reducer = None
        self._passthrough_columns = []
        transformed = self._transform_pipeline(x_train)
        self._feature_names_ = list(transformed.columns)
        if recipe.select is not None:
            if y_train is None and recipe.select in {"univariate", "model"}:
                raise ValidationError(f"{recipe.select} selection requires fold-train labels")
            select_frame = transformed
            if recipe.select_columns is not None:
                missing = [c for c in recipe.select_columns if c not in select_frame.columns]
                if missing:
                    raise ValidationError(f"select_columns missing after fold transform: {missing}")
                select_frame = select_frame[list(recipe.select_columns)]
            non_numeric = [
                c
                for c in select_frame.columns
                if not pd.api.types.is_numeric_dtype(select_frame[c])
            ]
            if non_numeric:
                raise ValidationError(
                    "Fold-local feature selection requires numeric columns after encode; "
                    f"non-numeric: {non_numeric[:12]}"
                )
            if select_frame.isna().any().any():
                raise ValidationError(
                    "Fold-local feature selection requires non-null features; impute first."
                )
            self._fit_selector(select_frame, y_train)
            support = self._selector.get_support()
            self._selected_features_ = [
                c for c, keep in zip(select_frame.columns, support, strict=True) if keep
            ]
            if not self._selected_features_:
                # Keep at least one column so estimators receive a matrix.
                variances = select_frame.var(axis=0).to_numpy(dtype=float)
                top = int(np.argmax(variances)) if len(variances) else 0
                self._selected_features_ = [str(select_frame.columns[top])]
        else:
            self._selector = None
            self._selected_features_ = list(self._feature_names_)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted steps to a frame, in the order they were fitted.

        Called on both halves of the fold: the training rows so the estimator
        can be fitted, and the evaluation rows so it can be scored. Both go
        through the same frozen parameters, which is what makes the comparison
        meaningful.

        Parameters
        ----------
        x:
            Rows to transform. Must carry the columns the recipe expects; the
            fitted steps decide what happens to each.

        Returns
        -------
        ~pandas.DataFrame
            A numeric frame with the fitted column layout, indexed as the input
            was so predictions can be joined back to their rows.

        Raises
        ------
        ~buildml.core.errors.ValidationError
            :meth:`fit` has not run, or selection retained no column that
            exists in this frame — which means the frame does not match what
            was fitted.
        """
        if not self._feature_names_ and self._selector is None:
            raise ValidationError("FoldLocalPreprocessor must be fitted before transform")
        base = self._transform_pipeline(x)
        if self._selector is None:
            return base
        keep = [c for c in self._selected_features_ if c in base.columns]
        if not keep:
            raise ValidationError("Fold-local selector retained no overlapping columns")
        return base[keep].copy()

    def _transform_pipeline(self, x: pd.DataFrame) -> pd.DataFrame:
        work = x.copy()
        if self._date_columns:
            work = self._apply_dates(work)
        if self._text_columns:
            work = self._apply_text(work)
        if self._outlier_columns:
            work = self._apply_outliers(work)
        if self._imputer is not None:
            work = self._apply_impute(work)
        if self._encode_columns and self.recipe.encode is not None:
            work = self._apply_encode_inplace(work)
        if self._binning_columns:
            work = self._apply_binning(work)
        if self._scaler is not None:
            work = self._apply_scale(work)
        if self._reducer is not None:
            work = self._apply_reduce(work)
        # Drop leftover datetime source columns when requested.
        if self.recipe.dates and self.recipe.date_drop_original and self._date_columns:
            drop = [c for c in self._date_columns if c in work.columns]
            if drop:
                work = work.drop(columns=drop)
        # Prefer numeric/encoded matrix; drop raw non-numeric leftovers that
        # estimators cannot consume (e.g. unexpanded object columns).
        keep_cols = [
            c
            for c in work.columns
            if pd.api.types.is_numeric_dtype(work[c])
            or pd.api.types.is_bool_dtype(work[c])
        ]
        if not keep_cols:
            raise ValidationError(
                "PreprocessRecipe could not resolve any columns for the requested steps"
            )
        return work[keep_cols].copy()

    def _fit_text(self, x_train: pd.DataFrame) -> None:
        recipe = self.recipe
        assert recipe.text is not None
        cols = _resolve_text(x_train, recipe.text_columns)
        vectorizers: dict[str, Any] = {}
        feature_names: list[str] = []
        n_features: dict[str, int] = {}
        for column in cols:
            documents = _as_text(x_train[column])
            vectorizer = _build_vectorizer(
                recipe.text,
                max_features=recipe.text_max_features,
                ngram_range=recipe.text_ngram_range,
            )
            matrix = vectorizer.fit_transform(documents)
            width = int(matrix.shape[1])
            n_features[column] = width
            names = _feature_names_for_column(column, vectorizer, width, recipe.text)
            feature_names.extend(names)
            vectorizers[column] = vectorizer
        self._text_columns = list(cols)
        self._text_vectorizers = vectorizers
        self._text_feature_names = feature_names
        self._text_n_features = n_features

    def _apply_text(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self._text_columns:
            return frame
        recipe = self.recipe
        assert recipe.text is not None
        missing = [c for c in self._text_columns if c not in frame.columns]
        if missing:
            raise ValidationError(f"Fold-local text columns missing: {missing}")
        blocks: list[pd.DataFrame] = []
        for column in self._text_columns:
            documents = _as_text(frame[column])
            matrix = self._text_vectorizers[column].transform(documents)
            dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
            width = self._text_n_features[column]
            names = _feature_names_for_column(
                column,
                self._text_vectorizers[column],
                width,
                recipe.text,
            )
            blocks.append(pd.DataFrame(dense, columns=names, index=frame.index))
        feature_frame = pd.concat(blocks, axis=1)
        if list(feature_frame.columns) != list(self._text_feature_names):
            feature_frame.columns = list(self._text_feature_names)
        out = frame
        if recipe.text_drop_input:
            out = out.drop(columns=list(self._text_columns))
        return pd.concat([out, feature_frame], axis=1)

    def _fit_reduce(self, x_train: pd.DataFrame) -> None:
        recipe = self.recipe
        assert recipe.reduce == "pca"
        cols = _resolve_numeric(x_train, recipe.reduce_columns)
        if not cols:
            raise ValidationError("Fold-local PCA requires numeric columns")
        values = x_train[cols].to_numpy(dtype=float)
        if np.isnan(values).any():
            raise ValidationError(
                "Fold-local PCA requires non-null features; impute before reduce."
            )
        n_samples, n_features = values.shape
        max_components = min(n_samples, n_features)
        if max_components < 1:
            raise ValidationError("Not enough fold-train rows/columns for PCA")
        n_components = recipe.reduce_n_components
        if n_components is None:
            pca_n: int | float = max_components
        elif isinstance(n_components, float):
            if not (0.0 < n_components <= 1.0):
                raise ValidationError("Float reduce_n_components must be in (0, 1]")
            pca_n = n_components
        else:
            if int(n_components) < 1:
                raise ValidationError("Integer reduce_n_components must be >= 1")
            pca_n = min(int(n_components), max_components)
        reducer = PCA(n_components=pca_n, svd_solver="full")
        reducer.fit(values)
        n_out = int(np.asarray(reducer.explained_variance_ratio_).shape[0])
        self._reduce_columns = list(cols)
        self._reduce_feature_names = [f"{recipe.reduce_prefix}_{i + 1}" for i in range(n_out)]
        self._reducer = reducer

    def _apply_reduce(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self._reducer is None:
            return frame
        missing = [c for c in self._reduce_columns if c not in frame.columns]
        if missing:
            raise ValidationError(f"Fold-local reduce columns missing: {missing}")
        values = frame[self._reduce_columns].to_numpy(dtype=float)
        if np.isnan(values).any():
            raise ValidationError(
                "Fold-local PCA transform found nulls; impute before reduce."
            )
        transformed = self._reducer.transform(values)
        component_frame = pd.DataFrame(
            transformed,
            columns=list(self._reduce_feature_names),
            index=frame.index,
        )
        out = frame
        if self.recipe.reduce_drop_input:
            out = out.drop(columns=list(self._reduce_columns))
        return pd.concat([out, component_frame], axis=1)

    def _fit_outliers(self, x_train: pd.DataFrame) -> None:
        recipe = self.recipe
        assert recipe.outliers is not None
        cols = _resolve_numeric(x_train, recipe.outlier_columns)
        if not cols:
            raise ValidationError("Fold-local outliers require numeric columns")
        if recipe.iqr_multiplier <= 0 or recipe.zscore_threshold <= 0:
            raise ValidationError("Outlier fence multipliers/thresholds must be positive")
        lower: dict[str, float] = {}
        upper: dict[str, float] = {}
        for column in cols:
            series = pd.to_numeric(x_train[column], errors="coerce").dropna()
            if series.empty:
                raise ValidationError(
                    f"Column '{column}' has no finite fold-train values for outlier fences"
                )
            if recipe.outliers == "iqr":
                q1 = float(series.quantile(0.25))
                q3 = float(series.quantile(0.75))
                iqr = q3 - q1
                lower[column] = q1 - recipe.iqr_multiplier * iqr
                upper[column] = q3 + recipe.iqr_multiplier * iqr
            else:
                mean = float(series.mean())
                std = float(series.std(ddof=0))
                if std == 0.0:
                    lower[column] = mean
                    upper[column] = mean
                else:
                    lower[column] = mean - recipe.zscore_threshold * std
                    upper[column] = mean + recipe.zscore_threshold * std
        self._outlier_columns = list(cols)
        self._outlier_lower = lower
        self._outlier_upper = upper

    def _apply_outliers(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self._outlier_columns:
            return frame
        if self.recipe.outlier_action == "detect":
            return frame
        missing = [c for c in self._outlier_columns if c not in frame.columns]
        if missing:
            raise ValidationError(f"Fold-local outlier columns missing: {missing}")
        out = frame.copy()
        for column in self._outlier_columns:
            values = pd.to_numeric(out[column], errors="coerce")
            out[column] = values.clip(
                lower=self._outlier_lower[column],
                upper=self._outlier_upper[column],
            )
        return out

    def _fit_encoder(self, x_train: pd.DataFrame, y_train: pd.Series | None) -> None:
        recipe = self.recipe
        cols = self._encode_columns
        method = recipe.encode
        assert method is not None
        if method in {"onehot", "ordinal"}:
            self._encoder = _make_encoder(method)
            self._encoder.fit(x_train[cols].astype(str))
            return
        if method == "infrequent":
            maps: dict[str, list[str]] = {}
            collapsed = x_train[cols].astype(str).copy()
            for column in cols:
                counts = collapsed[column].value_counts(dropna=False)
                threshold = _frequency_threshold(recipe.min_frequency, len(collapsed))
                rare = [str(level) for level, count in counts.items() if float(count) < threshold]
                maps[column] = rare
                collapsed[column] = collapsed[column].where(
                    ~collapsed[column].isin(rare),
                    INFREQUENT_LABEL,
                )
            self._infrequent_maps = maps
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            encoder.fit(collapsed)
            self._encoder = encoder
            return
        if method == "target":
            assert y_train is not None
            y = _numeric_target(y_train)
            prior = float(np.mean(y))
            self._target_prior = prior
            maps_target: dict[str, dict[str, float]] = {}
            for column in cols:
                maps_target[column] = _smoothed_means(
                    x_train[column].astype(str),
                    y,
                    prior=prior,
                    smoothing=recipe.target_smoothing,
                )
            self._target_maps = maps_target
            self._encoder = "target"
            return
        raise ValidationError(f"Unsupported encode method '{method}'")

    def _fit_selector(self, frame: pd.DataFrame, y_train: pd.Series | None) -> None:
        recipe = self.recipe
        assert recipe.select is not None
        if recipe.select == "variance":
            selector = VarianceThreshold(threshold=recipe.select_threshold)
            selector.fit(frame)
            self._selector = selector
            return
        assert y_train is not None
        y = _numeric_target(y_train)
        if recipe.select == "univariate":
            score_func = f_regression if recipe.select_score_func == "f_regression" else f_classif
            # Heuristic: discrete few-level targets prefer classification scores.
            if recipe.select_score_func == "f_classif":
                unique = pd.unique(y)
                if len(unique) > 20:
                    score_func = f_regression
            k_eff = min(max(1, recipe.select_k), frame.shape[1])
            selector = SelectKBest(score_func=score_func, k=k_eff)
            selector.fit(frame, y)
            self._selector = selector
            return
        # model-based: fit estimator on fold-train only, then SelectFromModel.
        model = recipe.select_estimator
        if model is None:
            n_classes = int(pd.Series(y).nunique())
            if n_classes > 20:
                model = RandomForestRegressor(n_estimators=40, random_state=0)
            elif n_classes > 2:
                model = RandomForestClassifier(n_estimators=40, random_state=0)
            else:
                model = LogisticRegression(max_iter=400)
        model.fit(frame, y_train if y_train is not None else y)
        selector = SelectFromModel(model, prefit=True)
        # Ensure get_support works even when all features are below threshold.
        if not any(selector.get_support()):
            if hasattr(model, "feature_importances_"):
                importances = np.asarray(model.feature_importances_, dtype=float)
            elif hasattr(model, "coef_"):
                coef = np.asarray(model.coef_, dtype=float)
                importances = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
            else:
                importances = np.ones(frame.shape[1], dtype=float)
            keep = np.zeros(frame.shape[1], dtype=bool)
            keep[int(np.argmax(importances))] = True

            class _Support:
                def get_support(self_inner) -> np.ndarray:
                    return keep

            self._selector = _Support()
            return
        self._selector = selector

    def _fit_dates(self, x_train: pd.DataFrame) -> None:
        cols = _resolve_datetime(x_train, self.recipe.date_columns)
        if not cols:
            raise ValidationError("Fold-local dates require datetime-like columns")
        self._date_columns = list(cols)
        created: list[str] = []
        for col in cols:
            created.extend(
                [
                    f"{col}_year",
                    f"{col}_month",
                    f"{col}_day",
                    f"{col}_dayofweek",
                    f"{col}_dayofyear",
                    f"{col}_quarter",
                    f"{col}_is_month_start",
                    f"{col}_is_month_end",
                ]
            )
            if self.recipe.date_include_time:
                created.extend([f"{col}_hour", f"{col}_minute", f"{col}_second"])
        self._date_created = created

    def _apply_dates(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self._date_columns:
            return frame
        missing = [c for c in self._date_columns if c not in frame.columns]
        if missing:
            raise ValidationError(f"Fold-local date columns missing: {missing}")
        out = frame.copy()
        for col in self._date_columns:
            parsed = pd.to_datetime(out[col], errors="coerce", utc=False)
            out[f"{col}_year"] = parsed.dt.year
            out[f"{col}_month"] = parsed.dt.month
            out[f"{col}_day"] = parsed.dt.day
            out[f"{col}_dayofweek"] = parsed.dt.dayofweek
            out[f"{col}_dayofyear"] = parsed.dt.dayofyear
            out[f"{col}_quarter"] = parsed.dt.quarter
            out[f"{col}_is_month_start"] = parsed.dt.is_month_start.astype("Int64")
            out[f"{col}_is_month_end"] = parsed.dt.is_month_end.astype("Int64")
            if self.recipe.date_include_time:
                out[f"{col}_hour"] = parsed.dt.hour
                out[f"{col}_minute"] = parsed.dt.minute
                out[f"{col}_second"] = parsed.dt.second
            if not self.recipe.date_drop_original:
                out[col] = parsed
        return out

    def _apply_impute(self, frame: pd.DataFrame) -> pd.DataFrame:
        assert self._imputer is not None
        missing = [c for c in self._impute_columns if c not in frame.columns]
        if missing:
            raise ValidationError(f"Fold-local impute columns missing: {missing}")
        out = frame.copy()
        values = self._imputer.transform(out[self._impute_columns])
        out[self._impute_columns] = values
        return out

    def _apply_scale(self, frame: pd.DataFrame) -> pd.DataFrame:
        assert self._scaler is not None
        missing = [c for c in self._scale_columns if c not in frame.columns]
        if missing:
            raise ValidationError(f"Fold-local scale columns missing: {missing}")
        out = frame.copy()
        values = self._scaler.transform(out[self._scale_columns])
        out[self._scale_columns] = values
        return out

    def _apply_encode_inplace(self, frame: pd.DataFrame) -> pd.DataFrame:
        encoded = self._transform_encoded(frame)
        out = frame.drop(columns=[c for c in self._encode_columns if c in frame.columns])
        return pd.concat([out, encoded], axis=1)

    def _fit_binning(self, x_train: pd.DataFrame) -> None:
        recipe = self.recipe
        assert recipe.binning is not None
        cols = _resolve_numeric(x_train, recipe.binning_columns)
        if not cols:
            raise ValidationError("Fold-local binning requires numeric columns")
        edges: dict[str, list[float]] = {}
        labels: dict[str, list[str]] = {}
        for column in cols:
            series = pd.to_numeric(x_train[column], errors="coerce").dropna()
            if series.empty:
                raise ValidationError(
                    f"Column '{column}' has no finite fold-train values for binning"
                )
            unique = int(series.nunique(dropna=True))
            bins = min(recipe.n_bins, unique) if unique >= 2 else 2
            if recipe.binning == "quantile":
                quantiles = np.linspace(0.0, 1.0, bins + 1)
                raw_edges = np.unique(np.quantile(series.to_numpy(), quantiles))
            else:
                raw_edges = np.unique(
                    np.linspace(float(series.min()), float(series.max()), bins + 1)
                )
            if len(raw_edges) < 3:
                center = float(series.iloc[0])
                raw_edges = np.array([center - 0.5, center + 0.5], dtype=float)
            raw_edges = raw_edges.astype(float)
            raw_edges[0] = float("-inf")
            raw_edges[-1] = float("inf")
            edge_list = [float(v) for v in raw_edges]
            edges[column] = edge_list
            labels[column] = [f"{column}_bin_{i}" for i in range(len(edge_list) - 1)]
        self._binning_columns = list(cols)
        self._binning_edges = edges
        self._binning_labels = labels

    def _apply_binning(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self._binning_columns:
            return frame
        missing = [c for c in self._binning_columns if c not in frame.columns]
        if missing:
            raise ValidationError(f"Fold-local binning columns missing: {missing}")
        out = frame.copy()
        for column in self._binning_columns:
            values = pd.to_numeric(out[column], errors="coerce")
            edges = self._binning_edges[column]
            codes = pd.cut(
                values,
                bins=edges,
                labels=False,
                include_lowest=True,
                right=True,
            )
            if self.recipe.binning_encode_as == "ordinal":
                out[f"{column}_bin"] = codes.astype("float")
            else:
                for level, name in enumerate(self._binning_labels[column]):
                    out[name] = (codes == level).astype("float")
            del out[column]
        return out

    def _transform_encoded(self, x: pd.DataFrame) -> pd.DataFrame:
        cols = self._encode_columns
        missing = [c for c in cols if c not in x.columns]
        if missing:
            raise ValidationError(f"Categorical recipe columns missing: {missing}")
        method = self.recipe.encode
        work = x[cols].astype(str)
        if method == "target":
            prior = float(self._target_prior if self._target_prior is not None else 0.0)
            encoded = pd.DataFrame(index=x.index)
            for column in cols:
                mapping = self._target_maps.get(column, {})
                encoded[f"{column}_target"] = work[column].map(mapping).fillna(prior).astype(float)
            return encoded
        if method == "infrequent":
            for column in cols:
                rare = set(self._infrequent_maps.get(column, ()))
                work[column] = work[column].where(~work[column].isin(rare), INFREQUENT_LABEL)
        assert self._encoder is not None and self._encoder != "target"
        matrix = self._encoder.transform(work)
        if method == "onehot" or method == "infrequent":
            names = list(self._encoder.get_feature_names_out(cols))
        else:
            names = list(cols)
        return pd.DataFrame(matrix, index=x.index, columns=names)


def build_fold_preprocessor(
    x_train: pd.DataFrame,
    recipe: PreprocessRecipe,
    y_train: pd.Series | None = None,
) -> FoldLocalPreprocessor:
    """Build and fit a fold-local preprocessor in one call.

    The convenience entry point used by cross-validation and the search
    methods: construct a :class:`FoldLocalPreprocessor` for the recipe and fit
    it to this fold's training rows.

    Parameters
    ----------
    x_train:
        Feature rows for this fold's training half.
    recipe:
        The steps to run.
    y_train:
        Labels for those rows, required when the recipe uses target encoding or
        supervised selection.

    Returns
    -------
    FoldLocalPreprocessor
        A fitted preprocessor, ready to transform both halves of the fold.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        The recipe has no steps enabled, or fitting failed — see
        :meth:`FoldLocalPreprocessor.fit` for the specific conditions.

    See Also
    --------
    transform_fold_features : Applies the result to a frame.
    """
    if recipe.is_empty():
        raise ValidationError("PreprocessRecipe has no steps to fit")
    preprocessor = FoldLocalPreprocessor(recipe)
    preprocessor.fit(x_train, y_train)
    return preprocessor


def transform_fold_features(preprocessor: Any, x: pd.DataFrame) -> pd.DataFrame:
    """Transform a frame and guarantee the result is a labelled DataFrame.

    A :class:`FoldLocalPreprocessor` already returns a DataFrame, but a
    user-supplied scikit-learn transformer may hand back a bare NumPy array
    with no column names. This normalises both cases, recovering names from
    ``get_feature_names_out`` when the transformer offers it, so downstream
    code — feature importance, error analysis, explanations — always has
    something to refer to columns by.

    Parameters
    ----------
    preprocessor:
        Any fitted object with a ``transform`` method.
    x:
        The frame to transform. Its index is carried onto the output so
        predictions stay joinable to their source rows.

    Returns
    -------
    ~pandas.DataFrame
        The transformed features. Column names come from the transformer when
        it provides them, and are positional integers otherwise.

    See Also
    --------
    build_fold_preprocessor : Produces the preprocessor this consumes.
    """
    transformed = preprocessor.transform(x)
    if isinstance(transformed, pd.DataFrame):
        return transformed
    names = None
    if hasattr(preprocessor, "get_feature_names_out"):
        names = list(preprocessor.get_feature_names_out())
    return pd.DataFrame(transformed, index=x.index, columns=names)


def _make_scaler(method: ScaleMethod) -> Any:
    if method == "standard":
        return StandardScaler()
    if method == "minmax":
        return MinMaxScaler()
    raise ValidationError(f"Unsupported scale method '{method}'")


def _make_encoder(method: EncodeMethod) -> Any:
    if method == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if method == "ordinal":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    raise ValidationError(f"Unsupported encode method '{method}' for sklearn encoder factory")


def _resolve_numeric(frame: pd.DataFrame, columns: tuple[str, ...] | None) -> list[str]:
    if columns is not None:
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise ValidationError(f"Numeric recipe columns missing: {missing}")
        return list(columns)
    return [str(c) for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]


def _resolve_categorical(frame: pd.DataFrame, columns: tuple[str, ...] | None) -> list[str]:
    if columns is not None:
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise ValidationError(f"Categorical recipe columns missing: {missing}")
        return list(columns)
    return [
        str(c)
        for c in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[c])
        and not pd.api.types.is_datetime64_any_dtype(frame[c])
    ]


def _remap_scaled_columns(
    pending: list[str],
    frame: pd.DataFrame,
    binned: list[str],
) -> list[str]:
    """Keep pre-encode scale targets; follow ``col`` → ``col_bin`` after binning."""
    resolved: list[str] = []
    binned_set = set(binned)
    for column in pending:
        if column in frame.columns:
            resolved.append(column)
        elif column in binned_set and f"{column}_bin" in frame.columns:
            resolved.append(f"{column}_bin")
    return resolved


def _resolve_datetime(frame: pd.DataFrame, columns: tuple[str, ...] | None) -> list[str]:
    if columns is not None:
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise ValidationError(f"Datetime recipe columns missing: {missing}")
        return list(columns)
    resolved: list[str] = []
    for column in frame.columns:
        name = str(column)
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            resolved.append(name)
            continue
        if pd.api.types.is_object_dtype(frame[column]) or pd.api.types.is_string_dtype(
            frame[column]
        ):
            parsed = pd.to_datetime(frame[column], errors="coerce")
            if parsed.notna().mean() >= 0.8:
                resolved.append(name)
    return resolved


def _resolve_text(frame: pd.DataFrame, columns: tuple[str, ...] | None) -> list[str]:
    if columns is not None:
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise ValidationError(f"Text recipe columns missing: {missing}")
        for column in columns:
            if pd.api.types.is_numeric_dtype(frame[column]):
                raise ValidationError(
                    f"Column '{column}' is numeric; fold-local text expects string-like values."
                )
        return list(columns)
    names = [
        str(c)
        for c in frame.columns
        if (
            pd.api.types.is_string_dtype(frame[c]) or pd.api.types.is_object_dtype(frame[c])
        )
        and not pd.api.types.is_datetime64_any_dtype(frame[c])
        and not pd.api.types.is_numeric_dtype(frame[c])
    ]
    if not names:
        raise ValidationError(
            "No text/object columns available for fold-local text. "
            "Pass text_columns=... explicitly."
        )
    return names


def _frequency_threshold(min_frequency: float | int, n_rows: int) -> float:
    if isinstance(min_frequency, float):
        if not 0.0 < min_frequency < 1.0:
            raise ValidationError("float min_frequency must be in (0, 1)")
        return min_frequency * n_rows
    if int(min_frequency) < 1:
        raise ValidationError("integer min_frequency must be >= 1")
    return float(min_frequency)


def _numeric_target(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
        if values.isna().any():
            raise ValidationError("Fold-local target steps require a non-null target")
        return values.to_numpy(dtype=float)
    codes, _ = pd.factorize(series.astype(str), sort=True)
    if (codes < 0).any():
        raise ValidationError("Fold-local target steps cannot proceed with null labels")
    return codes.astype(float)
