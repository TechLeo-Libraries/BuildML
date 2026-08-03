"""Train-fitted tabular generators (bootstrap, Gaussian copula, SMOTE wrap)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from buildml.core.errors import MissingExtraError, ValidationError
from buildml.synthetic.types import ColumnKind, ColumnSchemaSpec


def infer_column_kind(series: pd.Series) -> ColumnKind:
    """Heuristic column kind for mixed-type tabular synthesis.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
series:
    series (pd.Series).

Returns
-------
ColumnKind
    Return value (ColumnKind) produced by this operation.
    """
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_integer_dtype(series):
        nunique = int(series.nunique(dropna=True))
        # Low-cardinality integers behave like categoricals for copula care
        if nunique <= min(20, max(3, int(0.05 * max(len(series), 1)))):
            return "categorical"
        return "integer"
    if pd.api.types.is_float_dtype(series) or pd.api.types.is_numeric_dtype(series):
        return "continuous"
    return "categorical"


def build_column_specs(frame: pd.DataFrame) -> tuple[ColumnSchemaSpec, ...]:
    """Construct a column specs ready for fit or scoring.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
frame:
    Partition or full DataFrame slice used for this operation.

Returns
-------
tuple[ColumnSchemaSpec, ...]
    Tuple of results (tuple[ColumnSchemaSpec, ...]) for downstream Session steps.
    """
    specs: list[ColumnSchemaSpec] = []
    for name in frame.columns:
        series = frame[name]
        kind = infer_column_kind(series)
        cats: tuple[str, ...] = ()
        if kind == "categorical":
            cats = tuple(str(v) for v in series.dropna().astype(str).unique().tolist())
        specs.append(
            ColumnSchemaSpec(
                name=str(name),
                kind=kind,
                n_unique=int(series.nunique(dropna=True)),
                n_null=int(series.isna().sum()),
                categories=cats,
            )
        )
    return tuple(specs)


def _empirical_cdf_values(values: np.ndarray) -> np.ndarray:
    """Map finite values to (0, 1) via mid-rank empirical CDF."""
    n = len(values)
    if n == 0:
        return values
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    # Mid-rank → open unit interval
    return (ranks - 0.5) / n


def _nearest_ppf(sorted_vals: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Inverse empirical CDF via nearest order statistic."""
    if len(sorted_vals) == 0:
        return np.full(len(u), np.nan)
    idx = np.clip(np.floor(u * len(sorted_vals)).astype(int), 0, len(sorted_vals) - 1)
    return sorted_vals[idx]


@dataclass
class BootstrapGenerator:
    """Row bootstrap with optional Gaussian smoothing on continuous columns."""

    frame: pd.DataFrame
    continuous_cols: tuple[str, ...]
    integer_cols: tuple[str, ...]
    smooth_sigma: float = 0.0
    col_std: dict[str, float] = field(default_factory=dict)
    random_state: int = 42

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        specs: tuple[ColumnSchemaSpec, ...],
        *,
        smooth_sigma: float = 0.0,
        random_state: int = 42,
    ) -> BootstrapGenerator:
        """Run fit on input data using the fitted internal state.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
frame:
    Partition or full DataFrame slice used for this operation.
specs:
    specs (tuple[ColumnSchemaSpec, ...]).
smooth_sigma:
    smooth sigma (float).
random_state:
    Seed for stochastic steps (sampling, initialization, bagging).

Returns
-------
BootstrapGenerator
    Return value (BootstrapGenerator) produced by this operation.

Raises
------
ValidationError
    When preconditions for this operation are not met.
        """
        if frame.empty:
            raise ValidationError("Bootstrap synthesizer requires non-empty train rows.")
        continuous = tuple(s.name for s in specs if s.kind == "continuous")
        integer = tuple(s.name for s in specs if s.kind == "integer")
        stds: dict[str, float] = {}
        for col in continuous + integer:
            series = pd.to_numeric(frame[col], errors="coerce")
            std = float(series.std(ddof=1)) if len(series) > 1 else 0.0
            if not np.isfinite(std):
                std = 0.0
            stds[col] = std
        return cls(
            frame=frame.reset_index(drop=True).copy(),
            continuous_cols=continuous,
            integer_cols=integer,
            smooth_sigma=float(smooth_sigma),
            col_std=stds,
            random_state=int(random_state),
        )

    def sample(self, n: int, *, random_state: int | None = None) -> pd.DataFrame:
        """Run sample on input data using the fitted internal state.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
n:
    n (int).
random_state:
    Seed for stochastic steps (sampling, initialization, bagging).

Returns
-------
pd.DataFrame
    Return value (pd.DataFrame) produced by this operation.

Raises
------
ValidationError
    When preconditions for this operation are not met.
        """
        if n < 1:
            raise ValidationError("sample_synthetic n must be >= 1.")
        rng = np.random.default_rng(
            self.random_state if random_state is None else int(random_state)
        )
        idx = rng.integers(0, len(self.frame), size=int(n))
        out = self.frame.iloc[idx].reset_index(drop=True).copy()
        if self.smooth_sigma > 0:
            for col in self.continuous_cols:
                sigma = self.smooth_sigma * self.col_std.get(col, 0.0)
                if sigma > 0:
                    out[col] = (
                        pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
                        + rng.normal(0.0, sigma, size=len(out))
                    )
            for col in self.integer_cols:
                sigma = self.smooth_sigma * self.col_std.get(col, 0.0)
                if sigma > 0:
                    noisy = (
                        pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
                        + rng.normal(0.0, sigma, size=len(out))
                    )
                    out[col] = np.rint(noisy).astype(np.int64)
        return out


@dataclass
class _CatTransform:
    categories: tuple[str, ...]
    thresholds: np.ndarray  # cumulative probs length = n_cats


@dataclass
class GaussianCopulaGenerator:
    """Gaussian-copula tabular generator with mixed-type care.

    Continuous / integer columns use empirical CDFs. Categorical columns use
    frequency bins mapped through Φ⁻¹ so category proportions participate in
    the joint correlation (not independent marginal draws).
    """

    columns: tuple[str, ...]
    kinds: dict[str, ColumnKind]
    corr: np.ndarray
    sorted_cont: dict[str, np.ndarray]
    cat_transforms: dict[str, _CatTransform]
    null_rates: dict[str, float]
    random_state: int = 42
    correlation_ridge: float = 1e-3

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        specs: tuple[ColumnSchemaSpec, ...],
        *,
        correlation_ridge: float = 1e-3,
        random_state: int = 42,
    ) -> GaussianCopulaGenerator:
        """Run fit on input data using the fitted internal state.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
frame:
    Partition or full DataFrame slice used for this operation.
specs:
    specs (tuple[ColumnSchemaSpec, ...]).
correlation_ridge:
    correlation ridge (float).
random_state:
    Seed for stochastic steps (sampling, initialization, bagging).

Returns
-------
GaussianCopulaGenerator
    Return value (GaussianCopulaGenerator) produced by this operation.

Raises
------
ValidationError
    When preconditions for this operation are not met.
        """
        if frame.empty:
            raise ValidationError("Gaussian copula requires non-empty train rows.")
        if len(frame) < 3:
            raise ValidationError(
                "Gaussian copula needs ≥3 train rows to estimate a joint."
            )
        columns = tuple(str(s.name) for s in specs)
        kinds = {s.name: s.kind for s in specs}
        null_rates = {
            s.name: float(s.n_null) / float(max(len(frame), 1)) for s in specs
        }
        sorted_cont: dict[str, np.ndarray] = {}
        cat_transforms: dict[str, _CatTransform] = {}
        z_cols: list[np.ndarray] = []

        for spec in specs:
            col = spec.name
            series = frame[col]
            if spec.kind in {"continuous", "integer"}:
                vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
                finite = vals[np.isfinite(vals)]
                if len(finite) < 2:
                    raise ValidationError(
                        f"Column {col!r} has insufficient finite values for copula fit."
                    )
                sorted_vals = np.sort(finite)
                sorted_cont[col] = sorted_vals
                # Transform observed (impute nulls with median for latent only)
                filled = vals.copy()
                med = float(np.median(finite))
                filled[~np.isfinite(filled)] = med
                u = _empirical_cdf_values(filled)
                u = np.clip(u, 1e-6, 1.0 - 1e-6)
                z_cols.append(stats.norm.ppf(u))
            else:
                as_str = series.astype("string")
                observed = as_str.dropna()
                if observed.empty:
                    raise ValidationError(
                        f"Categorical column {col!r} has no observed train values."
                    )
                counts = observed.value_counts(normalize=True)
                categories = tuple(str(c) for c in counts.index.tolist())
                probs = counts.to_numpy(dtype=float)
                thresholds = np.cumsum(probs)
                thresholds[-1] = 1.0
                cat_transforms[col] = _CatTransform(
                    categories=categories, thresholds=thresholds
                )
                # Mid-bin CDF for each row
                cat_to_u = {}
                prev = 0.0
                for cat, thr in zip(categories, thresholds, strict=True):
                    mid = 0.5 * (prev + float(thr))
                    cat_to_u[cat] = mid
                    prev = float(thr)
                mode_cat = categories[0]
                u_rows = []
                for val in as_str.tolist():
                    if val is pd.NA or val is None or (isinstance(val, float) and np.isnan(val)):
                        u_rows.append(cat_to_u[mode_cat])
                    else:
                        u_rows.append(cat_to_u.get(str(val), cat_to_u[mode_cat]))
                u_arr = np.clip(np.asarray(u_rows, dtype=float), 1e-6, 1.0 - 1e-6)
                z_cols.append(stats.norm.ppf(u_arr))

        z = np.column_stack(z_cols)
        # Correlation with ridge for PSD
        if z.shape[1] == 1:
            corr = np.array([[1.0]], dtype=float)
        else:
            corr = np.corrcoef(z, rowvar=False)
            if not np.isfinite(corr).all():
                corr = np.eye(z.shape[1], dtype=float)
            ridge = float(correlation_ridge)
            corr = corr + ridge * np.eye(corr.shape[0])
            # Project to nearest PSD via eigenvalue clip
            eigvals, eigvecs = np.linalg.eigh(corr)
            eigvals = np.clip(eigvals, 1e-8, None)
            corr = (eigvecs * eigvals) @ eigvecs.T
            # Rescale diagonal to 1
            d = np.sqrt(np.clip(np.diag(corr), 1e-12, None))
            corr = corr / np.outer(d, d)
            np.fill_diagonal(corr, 1.0)

        return cls(
            columns=columns,
            kinds=kinds,
            corr=corr.astype(float),
            sorted_cont=sorted_cont,
            cat_transforms=cat_transforms,
            null_rates=null_rates,
            random_state=int(random_state),
            correlation_ridge=float(correlation_ridge),
        )

    def sample(
        self,
        n: int,
        *,
        random_state: int | None = None,
        condition: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Run sample on input data using the fitted internal state.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
n:
    n (int).
random_state:
    Seed for stochastic steps (sampling, initialization, bagging).
condition:
    condition (dict[str, Any] | None).

Returns
-------
pd.DataFrame
    Return value (pd.DataFrame) produced by this operation.

Raises
------
ValidationError
    When preconditions for this operation are not met.
        """
        if n < 1:
            raise ValidationError("sample_synthetic n must be >= 1.")
        rng = np.random.default_rng(
            self.random_state if random_state is None else int(random_state)
        )
        dim = len(self.columns)
        mean = np.zeros(dim)
        # Oversample when conditioning via rejection
        need = int(n)
        batch = need
        if condition:
            batch = max(need * 8, need + 16)
        collected: list[pd.DataFrame] = []
        remaining = need
        attempts = 0
        while remaining > 0 and attempts < 20:
            attempts += 1
            z = rng.multivariate_normal(mean, self.corr, size=batch)
            u = stats.norm.cdf(z)
            frame = self._inverse_transform(u, rng)
            if condition:
                mask = np.ones(len(frame), dtype=bool)
                for key, value in condition.items():
                    if key not in frame.columns:
                        raise ValidationError(
                            f"condition key {key!r} is not a synthesizer column."
                        )
                    mask &= frame[key].astype(str).to_numpy() == str(value)
                frame = frame.loc[mask].reset_index(drop=True)
            if len(frame) > remaining:
                frame = frame.iloc[:remaining].reset_index(drop=True)
            if len(frame):
                collected.append(frame)
                remaining -= len(frame)
            batch = max(remaining * 8, remaining + 16)
        if remaining > 0:
            raise ValidationError(
                "Could not satisfy categorical condition with rejection sampling; "
                "relax condition or increase n / train diversity."
            )
        out = pd.concat(collected, ignore_index=True) if collected else pd.DataFrame(
            columns=list(self.columns)
        )
        return out[list(self.columns)]

    def _inverse_transform(self, u: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
        data: dict[str, Any] = {}
        for j, col in enumerate(self.columns):
            uj = np.clip(u[:, j], 1e-6, 1.0 - 1e-6)
            kind = self.kinds[col]
            if kind in {"continuous", "integer"}:
                vals = _nearest_ppf(self.sorted_cont[col], uj)
                if kind == "integer":
                    vals = np.rint(vals).astype(np.int64)
                # Re-introduce nulls at train rate
                rate = self.null_rates.get(col, 0.0)
                if rate > 0:
                    null_mask = rng.random(len(vals)) < rate
                    if kind == "integer":
                        vals = vals.astype(object)
                    vals = vals.copy()
                    vals[null_mask] = np.nan
                data[col] = vals
            else:
                transform = self.cat_transforms[col]
                idxs = np.searchsorted(transform.thresholds, uj, side="left")
                idxs = np.clip(idxs, 0, len(transform.categories) - 1)
                cats = np.asarray(transform.categories, dtype=object)[idxs]
                rate = self.null_rates.get(col, 0.0)
                if rate > 0:
                    null_mask = rng.random(len(cats)) < rate
                    cats = cats.copy()
                    cats[null_mask] = pd.NA
                data[col] = cats
        return pd.DataFrame(data)


@dataclass
class SmoteGenerator:
    """SMOTE-backed synthesizer (optional ``buildml[imbalanced]``).

    Distinct from ``Session.resample``: this path fits a reusable generator
    plan and exposes ``sample_synthetic`` without mutating Session train until
    an explicit merge is requested.
    """

    feature_columns: tuple[str, ...]
    target_column: str
    x_train: np.ndarray
    y_train: np.ndarray
    feature_frame_columns: tuple[str, ...]
    non_feature_template: dict[str, Any]
    k_neighbors: int
    sampling_strategy: Any
    random_state: int
    class_dtype: str

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        specs: tuple[ColumnSchemaSpec, ...],
        *,
        target_column: str,
        feature_columns: tuple[str, ...] | None = None,
        k_neighbors: int = 5,
        sampling_strategy: Any = "auto",
        random_state: int = 42,
    ) -> SmoteGenerator:
        """Run fit on input data using the fitted internal state.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
frame:
    Partition or full DataFrame slice used for this operation.
specs:
    specs (tuple[ColumnSchemaSpec, ...]).
target_column:
    Name of the supervised target column.
feature_columns:
    feature columns (tuple[str, ...] | None).
k_neighbors:
    k neighbors (int).
sampling_strategy:
    sampling strategy (Any).
random_state:
    Seed for stochastic steps (sampling, initialization, bagging).

Returns
-------
SmoteGenerator
    Return value (SmoteGenerator) produced by this operation.

Raises
------
ValidationError
    When preconditions for this operation are not met.
        """
        try:
            from imblearn.over_sampling import SMOTE  # noqa: F401
        except ImportError as exc:
            raise MissingExtraError(
                "imbalanced",
                "SMOTE synthesizer (method='smote')",
            ) from exc

        if target_column not in frame.columns:
            raise ValidationError(
                f"SMOTE synthesizer requires target_column {target_column!r} in train."
            )
        if feature_columns is None:
            feature_columns = tuple(
                s.name for s in specs if s.name != target_column and s.kind != "categorical"
            )
        feature_columns = tuple(feature_columns)
        if not feature_columns:
            raise ValidationError(
                "SMOTE synthesizer needs ≥1 numeric feature column "
                "(encode categoricals first, or use bootstrap / gaussian_copula)."
            )
        feat_list = list(feature_columns)
        for col in feat_list:
            if not pd.api.types.is_numeric_dtype(frame[col]):
                raise ValidationError(
                    f"SMOTE feature {col!r} must be numeric; got dtype={frame[col].dtype}."
                )
        if frame[feat_list].isna().any().any():
            raise ValidationError(
                "SMOTE synthesizer cannot fit with NaNs in numeric features. "
                "Impute train first."
            )
        y = frame[target_column]
        if y.nunique(dropna=True) < 2:
            raise ValidationError("SMOTE synthesizer requires ≥2 classes in train.")
        minority = int(y.astype(str).value_counts().min())
        k = int(k_neighbors)
        if minority <= k:
            raise ValidationError(
                f"SMOTE needs minority_count > k_neighbors ({k}); "
                f"found minority_count={minority}."
            )
        x = frame[feat_list].to_numpy(dtype=float)
        # Preserve non-feature columns via train-mode fill when sampling
        template: dict[str, Any] = {}
        for col in frame.columns:
            if col in feature_columns or col == target_column:
                continue
            mode = frame[col].mode(dropna=True)
            template[col] = mode.iloc[0] if len(mode) else pd.NA

        # Smoke-fit to validate imblearn accepts the data
        smote = SMOTE(
            random_state=int(random_state),
            k_neighbors=k,
            sampling_strategy=sampling_strategy,
        )
        smote.fit_resample(x, y)

        return cls(
            feature_columns=feature_columns,
            target_column=target_column,
            x_train=x,
            y_train=np.asarray(y),
            feature_frame_columns=tuple(frame.columns),
            non_feature_template=template,
            k_neighbors=k,
            sampling_strategy=sampling_strategy,
            random_state=int(random_state),
            class_dtype=str(y.dtype),
        )

    def sample(self, n: int, *, random_state: int | None = None) -> pd.DataFrame:
        """Run sample on input data using the fitted internal state.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
n:
    n (int).
random_state:
    Seed for stochastic steps (sampling, initialization, bagging).

Returns
-------
pd.DataFrame
    Return value (pd.DataFrame) produced by this operation.

Raises
------
ValidationError
    When preconditions for this operation are not met.
        """
        from imblearn.over_sampling import SMOTE

        if n < 1:
            raise ValidationError("sample_synthetic n must be >= 1.")
        seed = self.random_state if random_state is None else int(random_state)
        rng = np.random.default_rng(seed)
        synthetic_parts: list[pd.DataFrame] = []
        remaining = int(n)
        round_i = 0
        while remaining > 0 and round_i < 30:
            smote = SMOTE(
                random_state=int(seed + round_i),
                k_neighbors=self.k_neighbors,
                sampling_strategy=self.sampling_strategy,
            )
            x_res, y_res = smote.fit_resample(self.x_train, self.y_train)
            n_orig = len(self.x_train)
            if len(x_res) <= n_orig:
                raise ValidationError(
                    "SMOTE produced no synthetic rows; check class balance / strategy."
                )
            x_syn = x_res[n_orig:]
            y_syn = np.asarray(y_res)[n_orig:]
            part = pd.DataFrame(x_syn, columns=list(self.feature_columns))
            part[self.target_column] = y_syn
            for col, value in self.non_feature_template.items():
                part[col] = value
            # Align column order
            ordered = [c for c in self.feature_frame_columns if c in part.columns]
            part = part[ordered]
            if len(part) > remaining:
                take = rng.choice(len(part), size=remaining, replace=False)
                part = part.iloc[take].reset_index(drop=True)
            synthetic_parts.append(part)
            remaining -= len(part)
            round_i += 1
        if remaining > 0:
            # Fallback: sample with replacement from accumulated synthetics
            pool = pd.concat(synthetic_parts, ignore_index=True)
            extra_idx = rng.integers(0, len(pool), size=remaining)
            synthetic_parts.append(pool.iloc[extra_idx].reset_index(drop=True))
        out = pd.concat(synthetic_parts, ignore_index=True)
        return out.iloc[: int(n)].reset_index(drop=True)
