"""Session-global preprocess and resample orchestration."""

from __future__ import annotations

from buildml.session._imports import *  # noqa: F403


def drop_columns(session, columns: list[str] | tuple[str, ...]) -> Session:
    """Drop columns from the current dataset.

    Records the operation on Session history and returns the result for downstream chaining.

    Parameters
    ----------
    columns:
        Column names to remove.
    session:
        Active Session with dataset and optional split plan attached.

    Returns
    -------
    Session
    ``self`` for fluent chaining.

    Notes
    -----
    Split membership is preserved (row order unchanged). Roles for dropped
    columns are removed.
    """
    session._dataset = drop_columns_transform(session.dataset, columns)
    session._record("drop_columns", {"columns": list(columns)})
    return session


def impute(
    session,
    *,
    columns: list[str] | None = None,
    strategy: Literal['mean', 'median', 'most_frequent', 'constant'] = "median",
    fill_value: Any | None = None,
) -> Session:
    """Fit imputation on train and transform the full dataset.

    Records the operation on Session history and returns the result for downstream chaining.

    Parameters
    ----------
    columns:
        Columns to impute. Defaults to numeric ``feature``-role columns
        (skips ``ignore`` / ``id`` / ``target`` / ``group`` / ``time`` /
        ``weight``). Pass ``columns=[...]`` to force-include any column.
    strategy:
        Imputation strategy.
    fill_value:
        Constant fill when ``strategy='constant'``.
    session:
        Active Session with dataset and optional split plan attached.

    Returns
    -------
    Session
        ``self`` for fluent chaining.

    Notes
    -----
    **Leakage:** Requires an existing split. Statistics are learned from
    the train partition only, then applied to all rows.
    """
    session.assert_can_fit("train")
    plan = fit_simple_imputer(
        session.dataset,
        session._split_plan,
        columns=columns,
        strategy=strategy,
        fill_value=fill_value,
    )
    session._dataset = transform_simple_imputer(session.dataset, plan)
    session._impute_plan = plan
    session._record("impute", plan.to_dict())
    return session


def encode(
    session,
    *,
    columns: list[str] | None = None,
    method: Literal['onehot', 'ordinal', 'infrequent', 'target'] = "onehot",
    min_frequency: float | int = 0.05,
    n_folds: int = 5,
    random_state: int = 0,
    smoothing: float = 10.0,
) -> Session:
    """Fit categorical encoding on train and transform the full dataset.

    Records the operation on Session history and returns the result for downstream chaining.

    Parameters
    ----------
    columns:
        Columns to encode. Defaults to categorical ``feature``-role columns
        (skips ``ignore`` / ``id`` / ``target`` / ``group`` / ``time`` /
        ``weight``). Pass ``columns=[...]`` to force-include any column.
    method:
        ``onehot`` / ``ordinal`` for standard encodings; ``infrequent`` to
        pool rare train levels before one-hot; ``target`` for smoothed mean
        target encoding with out-of-fold values on train rows.
    min_frequency:
        For ``infrequent``: float in (0, 1) as a train fraction, or an
        absolute integer count threshold.
        n_folds / random_state / smoothing:
        Target-encoding controls (ignored for other methods).
    session:
        Active Session with dataset and optional split plan attached.

    Returns
    -------
    Session
        ``self`` for fluent chaining.

    Notes
    -----
    **Leakage:** Requires a split. Vocabularies and target means are learned
    on train only. Target encoding writes out-of-fold values on train and
    full-train means on holdouts.
    """
    session.assert_can_fit("train")
    plan = fit_encoder(
        session.dataset,
        session._split_plan,
        columns=columns,
        method=method,
        min_frequency=min_frequency,
        n_folds=n_folds,
        random_state=random_state,
        smoothing=smoothing,
    )
    session._dataset, result = transform_encoder(
        session.dataset, plan, split_plan=session._split_plan
    )
    session._encode_plan = plan
    session._last_preprocess = result
    session._record(
        "encode", plan.to_dict(), warnings=result.warnings, result_summary=result.to_dict()
    )
    return session


def handle_outliers(
    session,
    *,
    columns: list[str] | None = None,
    method: Literal['iqr', 'zscore'] = "iqr",
    action: Literal['detect', 'cap', 'drop'] = "cap",
    iqr_multiplier: float = 1.5,
    zscore_threshold: float = 3.0,
) -> Session:
    """Screen or treat numeric outliers using train-fitted fences.

    Records the operation on Session history and returns the result for downstream chaining.

    Parameters
    ----------
    method:
        ``iqr`` (Tukey fences) or ``zscore``.
    action:
        ``detect`` records the screen without mutating values; ``cap``
        winsorizes to the fences; ``drop`` removes flagged rows and rebuilds
        split membership.
    session:
        Active Session with dataset and optional split plan attached.
    columns:
        Column names to include or transform.
    iqr_multiplier:
        Controls ``iqr_multiplier``; see the function signature for type and default.
    zscore_threshold:
        Controls ``zscore_threshold``; see the function signature for type and default.

    Returns
    -------
    Session
        ``self`` for fluent chaining.

    Notes
    -----
    **Leakage:** Fence statistics are learned on train only, then applied
    with the frozen bounds. Heuristic screens are not proof of error.
    """
    session.assert_can_fit("train")
    assert session._split_plan is not None
    plan = fit_outlier_plan(
        session.dataset,
        session._split_plan,
        columns=columns,
        method=method,
        action=action,
        iqr_multiplier=iqr_multiplier,
        zscore_threshold=zscore_threshold,
    )
    dataset, split_plan, plan, result = apply_outlier_plan(
        session.dataset, session._split_plan, plan
    )
    session._dataset = dataset
    session._split_plan = split_plan
    session._outlier_plan = plan
    session._last_preprocess = result
    session._record(
        "handle_outliers", plan.to_dict(), warnings=result.warnings, result_summary=result.to_dict()
    )
    return session


def bin(
    session,
    *,
    columns: list[str] | None = None,
    strategy: Literal['quantile', 'uniform'] = "quantile",
    n_bins: int = 5,
    encode_as: Literal['ordinal', 'onehot'] = "ordinal",
) -> Session:
    """Discretize numeric columns with train-fitted bin edges.

    Records the operation on Session history and returns the result for downstream chaining.

    Parameters
    ----------
    session:
        Active Session with dataset and optional split plan attached.
    columns:
        Column names to include or transform.
    strategy:
        Controls ``strategy``; see the function signature for type and default.
    n_bins:
        Controls ``n_bins``; see the function signature for type and default.
    encode_as:
        Controls ``encode_as``; see the function signature for type and default.

    Returns
    -------
    Session
        ``self`` for fluent chaining.

    Notes
    -----
    **Leakage:** Edges are learned on train only. End bins use open
    ``±inf`` edges so score-time extremes remain defined.
    """
    session.assert_can_fit("train")
    plan = fit_binning(
        session.dataset,
        session._split_plan,
        columns=columns,
        strategy=strategy,
        n_bins=n_bins,
        encode_as=encode_as,
    )
    session._dataset, result = transform_binning(session.dataset, plan)
    session._binning_plan = plan
    session._last_preprocess = result
    session._record(
        "bin", plan.to_dict(), warnings=result.warnings, result_summary=result.to_dict()
    )
    return session


def select_features(
    session,
    *,
    strategy: Literal['variance', 'univariate', 'model'] = "variance",
    columns: list[str] | None = None,
    threshold: float = 0.0,
    k: int = 10,
    score_func: Literal['f_classif', 'f_regression', 'mutual_info'] = "f_classif",
    estimator: Any | None = None,
) -> Session:
    """Select a feature subset using train-only scores or model reliance.

    Records the operation on Session history and returns the result for downstream chaining.

    Parameters
    ----------
    strategy:
        ``variance`` (VarianceThreshold), ``univariate`` (SelectKBest), or
        ``model`` (SelectFromModel).
        threshold / k / score_func / estimator:
        Strategy-specific controls. Non-feature roles (target, id, group,
        time, weight) are preserved.
    session:
        Active Session with dataset and optional split plan attached.
    columns:
        Column names to include or transform.

    Returns
    -------
    Session
        ``self`` for fluent chaining.

    Notes
    -----
    **Leakage:** Selection fits on train only. Encode categoricals and
    impute before calling when features are non-numeric or contain nulls.
    """
    session.assert_can_fit("train")
    plan = fit_feature_selector(
        session.dataset,
        session._split_plan,
        strategy=strategy,
        columns=columns,
        threshold=threshold,
        k=k,
        score_func=score_func,
        estimator=estimator,
    )
    session._dataset, result = transform_feature_selector(session.dataset, plan)
    session._feature_select_plan = plan
    session._last_preprocess = result
    session._record(
        "select_features", plan.to_dict(), warnings=result.warnings, result_summary=result.to_dict()
    )
    return session


def scale(
    session, *, columns: list[str] | None = None, method: Literal['standard', 'minmax'] = "standard"
) -> Session:
    """Fit scaling on train and transform the full dataset.

    Records the operation on Session history and returns the result for downstream chaining.

    Parameters
    ----------
    columns:
        Columns to scale. Defaults to numeric ``feature``-role columns
        (skips ``ignore`` / ``id`` / ``target`` / ``group`` / ``time`` /
        ``weight`` — so costs and identifiers stay unmutated). Pass
        ``columns=[...]`` to force-include any column.
    method:
        ``standard`` or ``minmax``.
    session:
        Active Session with dataset and optional split plan attached.

    Returns
    -------
    Session
        ``self`` for fluent chaining.

    Notes
    -----
    **Leakage:** Requires a split. Scaler is fit on train only.
    """
    session.assert_can_fit("train")
    plan = fit_scaler(session.dataset, session._split_plan, columns=columns, method=method)
    session._dataset = transform_scaler(session.dataset, plan)
    session._scale_plan = plan
    session._record("scale", plan.to_dict())
    return session


def text_features(
    session,
    *,
    columns: list[str] | None = None,
    method: Literal['count', 'tfidf', 'hashing'] = "tfidf",
    max_features: int | None = 128,
    ngram_range: tuple[int, int] = (1, 1),
    drop_input_columns: bool = True,
) -> Session:
    """Fit text vectorizers on train and expand columns into numeric features.

    Records the operation on Session history and returns the result for downstream chaining.

    Parameters
    ----------
    method:
        ``tfidf`` (default), ``count``, or ``hashing``.
    max_features:
        Vocabulary width for count/TF-IDF, or hashing output width.
    ngram_range:
        Inclusive n-gram bounds passed to the sklearn vectorizer.
    session:
        Active Session with dataset and optional split plan attached.
    columns:
        Column names to include or transform.
    drop_input_columns:
        Controls ``drop_input_columns``; see the function signature for type and default.

    Returns
    -------
    Session
        ``self`` for fluent chaining.

    Notes
    -----
    **Leakage:** Requires a split. Vocabularies and IDF weights are learned
    from train documents only. Missing text becomes empty strings.
    """
    session.assert_can_fit("train")
    plan = fit_text_features(
        session.dataset,
        session._split_plan,
        columns=columns,
        method=method,
        max_features=max_features,
        ngram_range=ngram_range,
        drop_input_columns=drop_input_columns,
    )
    session._dataset, result = transform_text_features(session.dataset, plan)
    session._text_plan = plan
    session._last_preprocess = result
    session._record(
        "text_features", plan.to_dict(), warnings=result.warnings, result_summary=result.to_dict()
    )
    return session


def reduce_dimensions(
    session,
    *,
    columns: list[str] | None = None,
    method: Literal["pca", "umap", "tsne"] = "pca",
    n_components: int | float | None = None,
    drop_input_columns: bool = True,
    prefix: str = "pc",
    random_state: int | None = 0,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    tsne_perplexity: float = 30.0,
    tsne_learning_rate: str | float = "auto",
) -> Session:
    """Fit dimensionality reduction on train and replace numeric columns.

    Records the operation on Session history and returns the result for downstream chaining.

    Parameters
    ----------
    session:
        Active Session with dataset and optional split plan attached.
    columns:
        Column names to include or transform.
    method:
        Algorithm or method identifier for the resolved backend.
    n_components:
        Controls ``n_components``; see the function signature for type and default.
    drop_input_columns:
        Controls ``drop_input_columns``; see the function signature for type and default.
    prefix:
        Controls ``prefix``; see the function signature for type and default.
    random_state:
        Controls ``random_state``; see the function signature for type and default.
    umap_n_neighbors:
        Controls ``umap_n_neighbors``; see the function signature for type and default.
    umap_min_dist:
        Controls ``umap_min_dist``; see the function signature for type and default.
    tsne_perplexity:
        Controls ``tsne_perplexity``; see the function signature for type and default.
    tsne_learning_rate:
        Controls ``tsne_learning_rate``; see the function signature for type and default.

    Returns
    -------
    Session
        ``self`` for fluent chaining.
    """
    session.assert_can_fit("train")
    plan = fit_reducer(
        session.dataset,
        session._split_plan,
        columns=columns,
        method=method,
        n_components=n_components,
        drop_input_columns=drop_input_columns,
        prefix=prefix,
        random_state=random_state,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        tsne_perplexity=tsne_perplexity,
        tsne_learning_rate=tsne_learning_rate,
    )
    session._dataset, result = transform_reducer(session.dataset, plan)
    session._reduce_plan = plan
    session._last_preprocess = result
    session._record(
        "reduce_dimensions",
        plan.to_dict(),
        warnings=result.warnings,
        result_summary=result.to_dict(),
    )
    return session


def register_transform(
    session_cls,
    name: str,
    *,
    fit: Any,
    transform: Any,
    description: str = "",
    output_columns: Any | None = None,
    drop_input_columns: bool = False,
    serializable: bool = True,
    overwrite: bool = False,
) -> CustomTransformSpec:
    """Register a custom train-fit transform for :meth:`apply_custom_transform`.

    The ``fit`` callable receives only train rows for the selected columns.

    See :func:`buildml.preprocess.register_transform` for the full contract.

    Parameters
    ----------
    session_cls:
        Session class constructor used by module-level factory helpers.
    name:
        Registered transform or bundle identifier.
    fit:
        Controls ``fit``; see the function signature for type and default.
    transform:
        Controls ``transform``; see the function signature for type and default.
    description:
        Controls ``description``; see the function signature for type and default.
    output_columns:
        Controls ``output_columns``; see the function signature for type and default.
    drop_input_columns:
        Controls ``drop_input_columns``; see the function signature for type and default.
    serializable:
        Controls ``serializable``; see the function signature for type and default.
    overwrite:
        Controls ``overwrite``; see the function signature for type and default.

    Returns
    -------
    CustomTransformSpec
        Registered transform specification for reuse.
    """
    return register_custom_transform(
        name,
        fit=fit,
        transform=transform,
        description=description,
        output_columns=output_columns,
        drop_input_columns=drop_input_columns,
        serializable=serializable,
        overwrite=overwrite,
    )


def list_transforms(session_cls) -> tuple[CustomTransformSpec, ...]:
    """Return registered custom transforms in name order.

    Records the operation on Session history and returns the result for downstream chaining.

    Parameters
    ----------
    session_cls:
        Session class constructor used by module-level factory helpers.

    Returns
    -------
    tuple[CustomTransformSpec, ...]
        Registered transform specification for reuse.
    """
    return list_registered_transforms()


def apply_custom_transform(
    session, name: str, *, columns: list[str], params: Mapping[str, Any] | None = None
) -> Session:
    """Fit a registered custom transform on train and apply it to all rows.

    Records the operation on Session history and returns the result for downstream chaining.

    Parameters
    ----------
    name:
        Name previously passed to :meth:`register_transform`.
    columns:
        Input columns passed to fit/transform.
    params:
        Optional parameters forwarded to the registered ``fit`` callable.
    session:
        Active Session with dataset and optional split plan attached.

    Returns
    -------
    Session
        ``self`` for fluent chaining.

    Notes
    -----
    **Leakage:** Requires a split. Fit sees train rows only. Score-time
    replay requires the same name to remain registered in-process.
    """
    session.assert_can_fit("train")
    plan = fit_custom_transform(
        session.dataset, session._split_plan, name=name, columns=columns, params=params
    )
    session._dataset, result = transform_custom(session.dataset, plan)
    session._custom_plan = plan
    session._last_preprocess = result
    session._record(
        "apply_custom_transform",
        plan.to_dict(),
        warnings=result.warnings,
        result_summary=result.to_dict(),
    )
    return session


def extract_dates(
    session,
    columns: list[str] | tuple[str, ...] | None = None,
    *,
    include_time: bool = False,
    drop_original: bool = False,
) -> Session:
    """Expand datetime columns into calendar/time parts (``.dt``-correct).

    Records the operation on Session history and returns the result for downstream chaining.

    Parameters
    ----------
    session:
        Active Session with dataset and optional split plan attached.
    columns:
        Column names to include or transform.
    include_time:
        Controls ``include_time``; see the function signature for type and default.
    drop_original:
        Controls ``drop_original``; see the function signature for type and default.

    Returns
    -------
    Session
        ``self`` for fluent chaining.
    """
    session._dataset, plan = extract_date_features(
        session.dataset, columns=columns, include_time=include_time, drop_original=drop_original
    )
    session._date_plan = plan
    session._record("extract_dates", plan.to_dict())
    return session


def apply_preprocess_plans(
    session,
    data: Dataset | pd.DataFrame | None = None,
    plans: dict[str, Any] | None = None,
    *,
    inplace: bool = True,
    use_session_plans: bool = True,
) -> ApplyPlansResult:
    """Re-apply fitted preprocess plans in score-time order.

    Records the operation on Session history and returns the result for downstream chaining.

    Parameters
    ----------
    data:
        Optional Dataset or DataFrame to transform. Defaults to this
        session's dataset.
    plans:
        Optional plan mapping (checkpoint/pipeline ``plans.joblib`` payload
        or short keys). When omitted and ``use_session_plans=True``, uses
        plans currently attached to the session.
    inplace:
        When ``True`` and ``data`` is omitted (or is this session's
        dataset), replace the session dataset and update the split plan if
        outlier drop rewrote membership.
    use_session_plans:
        Merge session-attached plans under any explicit ``plans`` mapping.
    session:
        Active Session with dataset and optional split plan attached.

    Returns
    -------
    ApplyPlansResult
    Transformed dataset plus applied/skipped steps and warnings.

    Notes
    -----
    **Order:** dates → impute → outliers → encode → binning → scale →
    feature_select. Resample plans are lineage-only and are never
    reapplied at score time.
    **Leakage:** Plans must already be train-fitted; this method does not
    fit. Missing columns raise :class:`~buildml.core.errors.ValidationError`.

    Raises
    ------
    ValidationError
        When prerequisites are missing or inputs are invalid.
    """
    if session._dataset is None and data is None:
        raise ValidationError("No dataset attached. Ingest data or pass data=...")
    resolved_plans = dict(plans or {})
    if use_session_plans:
        for key, value in session._plan_objects().items():
            resolved_plans.setdefault(key, value)
    target = session.dataset if data is None else data
    result = run_apply_preprocess_plans(
        target, resolved_plans, split_plan=session._split_plan
    )
    mutating = inplace and (
        data is None or (isinstance(data, Dataset) and data is session._dataset)
    )
    if mutating:
        session._dataset = result.dataset
        if result.split_plan is not None:
            session._split_plan = result.split_plan
    session._record(
        "apply_preprocess_plans",
        {"inplace": mutating, "applied": list(result.applied), "skipped": list(result.skipped)},
        warnings=result.warnings,
        result_summary=result.to_dict(),
    )
    return result


def resample(
    session,
    *,
    sampler: Literal['smote', 'random_oversample', 'random_undersample', 'adasyn', 'borderline_smote'] = "smote",
    random_state: int = 42,
    sampling_strategy: str | float | dict[str, float] = "auto",
) -> Session:
    """Resample the **train** partition only (requires ``buildml[imbalanced]``).

    Validation/test rows are never altered. See

    :meth:`resample_strategies` for strategy guidance.

    Parameters
    ----------
    session:
        Active Session with dataset and optional split plan attached.
    sampler:
        Controls ``sampler``; see the function signature for type and default.
    random_state:
        Controls ``random_state``; see the function signature for type and default.
    sampling_strategy:
        Controls ``sampling_strategy``; see the function signature for type and default.

    Returns
    -------
    Session
        ``self`` for fluent chaining.
    """
    dataset, plan, resample_plan = resample_train(
        session.dataset,
        session._split_plan,
        sampler=sampler,
        random_state=random_state,
        sampling_strategy=sampling_strategy,
    )
    session._dataset = dataset
    session._split_plan = plan
    session._resample_plan = resample_plan
    session._record("resample", resample_plan.to_dict())
    return session


def resample_strategies(session) -> list[dict[str, Any]]:
    """List imbalance resampling strategies and when to use them.

    Records the operation on Session history and returns the result for downstream chaining.

    Parameters
    ----------
    session:
        Active Session with dataset and optional split plan attached.

    Returns
    -------
    list[dict[str, Any]]
        Domain result object from the underlying ``buildml`` module.
    """
    return list_resample_strategies()
